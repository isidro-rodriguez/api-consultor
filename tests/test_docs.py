import zipfile
from pathlib import Path

import pytest

from api_consultor import docs as docs_module
from api_consultor.docs import (
    CHUNK_CHARS,
    DocsError,
    collect,
    extract_doc_chunks,
    materialize,
    sections,
    split_text,
)

SPHINX = """\
"os" --- Interfaces del sistema
*******************************

Intro.

Archivos
========

os.listdir(path='.')

   Devuelve una lista.
"""


def test_sections_detect_setext_headings() -> None:
    found = list(sections(SPHINX))
    assert [title for title, _ in found] == [
        "",
        '"os" --- Interfaces del sistema',
        "Archivos",
    ]
    assert "os.listdir" in found[2][1]


def test_sections_detect_atx_headings_but_not_inside_fences() -> None:
    text = (
        "# Titulo\n\ncuerpo\n\n"
        "```python\n# comentario, no titulo\n```\n\n"
        "## Sub\n\nfin\n"
    )
    titles = [title for title, _ in sections(text)]
    assert titles == ["", "Titulo", "Sub"]


def test_sections_ignore_table_borders() -> None:
    text = "Nombre   Valor\n=======  =====\nuno      1\n"
    assert [title for title, _ in sections(text)] == [""]


def test_split_respects_limit_without_cutting_paragraphs() -> None:
    paragraphs = [f"parrafo {i} " + "x" * 390 for i in range(10)]
    chunks = list(split_text("\n\n".join(paragraphs), 1000))
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert all(paragraph in "\n\n".join(chunks) for paragraph in paragraphs)


def test_split_wraps_giant_paragraphs_by_lines() -> None:
    giant = "\n".join(f"linea {i}" for i in range(600))
    chunks = list(split_text(giant, 500))
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "linea 599" in chunks[-1]


def test_labels_are_unique_for_repeated_titles_and_long_sections() -> None:
    text = "Ejemplos\n--------\n\nuno\n\nEjemplos\n--------\n\ndos\n"
    chunks = extract_doc_chunks("notas", [("a.txt", text)])
    labels = [chunk.qualname for chunk in chunks]
    assert labels == ["a.txt#Ejemplos", "a.txt#Ejemplos [2]"]
    long_body = "T\n=\n\n" + "\n\n".join("p" * 600 for _ in range(6))
    labels = [c.qualname for c in extract_doc_chunks("n", [("b.txt", long_body)])]
    assert len(labels) == len(set(labels)) > 1


def test_chunks_are_limited_and_carry_label_and_version() -> None:
    chunks = extract_doc_chunks("python-3.15-docs-text", [("library/os.txt", SPHINX)])
    assert all(chunk.version == "3.15" for chunk in chunks)
    assert all(len(chunk.content) <= CHUNK_CHARS + 200 for chunk in chunks)
    assert chunks[0].content.startswith(chunks[0].qualname)


def test_version_is_unknown_without_number_in_name() -> None:
    chunks = extract_doc_chunks("mis-notas", [("a.md", "# T\n\ncuerpo\n")])
    assert chunks[0].version == "desconocida"


def test_exclude_patterns_skip_files() -> None:
    files = [
        ("whatsnew/3.15.txt", "T\n=\n\nx\n"),
        ("library/os.txt", "T\n=\n\ny\n"),
    ]
    chunks = extract_doc_chunks("py-3.15", files, exclude=["whatsnew/*"])
    assert {chunk.qualname.split("#")[0] for chunk in chunks} == {"library/os.txt"}


def test_collect_reads_zip_and_strips_common_root(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    with zipfile.ZipFile(docs / "python-3.15-docs-text.zip", "w") as zf:
        zf.writestr("python-3.15-docs-text/library/os.txt", "T\n=\n\nx\n")
        zf.writestr("python-3.15-docs-text/imagen.png", "no es texto")
    sources = collect(docs)
    assert list(sources) == ["python-3.15-docs-text"]
    assert [path for path, _ in sources["python-3.15-docs-text"]] == ["library/os.txt"]


def test_collect_groups_directories_and_loose_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    (docs / "mis-notas" / "sub").mkdir(parents=True)
    (docs / "mis-notas" / "sub" / "a.md").write_text("# A\n\nx\n", encoding="utf-8")
    (docs / "suelto.rst").write_text("T\n=\n\nx\n", encoding="utf-8")
    (docs / "datos.csv").write_text("a,b\n", encoding="utf-8")
    (docs / ".oculto.txt").write_text("x", encoding="utf-8")
    sources = collect(docs)
    assert set(sources) == {"mis-notas", "docs"}
    assert sources["mis-notas"][0][0] == "sub/a.md"
    assert [path for path, _ in sources["docs"]] == ["suelto.rst"]


def test_doc_chunks_remember_file_and_heading_line(tmp_path: Path) -> None:
    text = "# Uno\n\ncuerpo\n\n# Dos\n\notro\n\n# Uno\n\nrepetido\n"
    chunks = extract_doc_chunks("n", [("a.md", text)], root=tmp_path)
    assert {c.path for c in chunks} == {str(tmp_path / "a.md")}
    assert [c.line for c in chunks] == [1, 5, 9]


def test_doc_chunks_from_zip_have_no_location() -> None:
    chunks = extract_doc_chunks("n", [("a.md", "# T\n\nx\n")])
    assert [(c.path, c.line) for c in chunks] == [("", 0)]


def make_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)


def test_materialize_folders_and_loose_files_stay_in_place(tmp_path: Path) -> None:
    (tmp_path / "carpeta").mkdir()
    cache = tmp_path / "cache"
    assert materialize(tmp_path, "carpeta", cache) == (tmp_path / "carpeta").resolve()
    assert materialize(tmp_path, "docs", cache) == tmp_path.resolve()
    assert not cache.exists()


def test_materialize_extracts_zip_texts_without_the_common_root(
    tmp_path: Path,
) -> None:
    make_zip(
        tmp_path / "py-3.15.zip",
        {
            "py-3.15/library/os.txt": "T\n=\n\nx\n",
            "py-3.15/whatsnew/3.15.txt": "N\n=\n\ny\n",
            "py-3.15/imagen.png": "no es texto",
        },
    )
    root = materialize(tmp_path, "py-3.15", tmp_path / "cache", ["whatsnew/*"])
    assert root == (tmp_path / "cache" / "py-3.15").resolve()
    assert sorted(
        f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()
    ) == ["library/os.txt"]


def test_materialize_replaces_a_previous_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "n.zip"
    make_zip(archive, {"viejo.md": "# V\n\nx\n"})
    materialize(tmp_path, "n", tmp_path / "cache")
    make_zip(archive, {"nuevo.md": "# N\n\ny\n"})
    root = materialize(tmp_path, "n", tmp_path / "cache")
    assert [f.name for f in root.rglob("*") if f.is_file()] == ["nuevo.md"]


def test_materialize_ignores_members_that_escape_the_target(tmp_path: Path) -> None:
    make_zip(tmp_path / "n.zip", {"../fuera.md": "x", "dentro.md": "# D\n\ny\n"})
    root = materialize(tmp_path, "n", tmp_path / "cache")
    assert not (tmp_path / "fuera.md").exists()
    assert not (tmp_path / "cache" / "fuera.md").exists()
    assert [f.name for f in root.rglob("*") if f.is_file()] == ["dentro.md"]


def _zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return path


def test_a_folder_and_a_zip_with_the_same_name_collide(tmp_path: Path) -> None:
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "a.md").write_text("# A\n", encoding="utf-8")
    _zip(tmp_path / "foo.zip", {"b.md": "# B\n"})
    with pytest.raises(DocsError, match=r"«foo»: foo\.zip"):
        collect(tmp_path)


def test_loose_files_collide_with_a_docs_folder(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "suelto.md").write_text("# S\n", encoding="utf-8")
    with pytest.raises(DocsError, match="«docs»"):
        collect(tmp_path)


def test_a_zip_bomb_is_rejected_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Muy comprimible: pocos bytes en el zip, mucho texto al descomprimir.
    _zip(tmp_path / "bomba.zip", {"a.txt": "0" * 4096, "b.txt": "0" * 4096})
    monkeypatch.setattr(docs_module, "MAX_SOURCE_BYTES", 6000)
    with pytest.raises(DocsError, match=r"bomba\.zip ocupa"):
        collect(tmp_path)


def test_oversized_files_and_too_many_files_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "muchos"
    folder.mkdir()
    for index in range(3):
        (folder / f"{index}.md").write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(docs_module, "MAX_FILES", 2)
    with pytest.raises(DocsError, match="3 ficheros"):
        collect(tmp_path)
    monkeypatch.setattr(docs_module, "MAX_FILES", 10)
    monkeypatch.setattr(docs_module, "MAX_FILE_BYTES", 50)
    with pytest.raises(DocsError, match=r"0\.md ocupa"):
        collect(tmp_path)


def test_a_damaged_zip_is_reported(tmp_path: Path) -> None:
    (tmp_path / "roto.zip").write_bytes(b"esto no es un zip")
    with pytest.raises(DocsError, match=r"roto\.zip no es un \.zip válido"):
        collect(tmp_path)


def test_invalid_utf8_is_replaced_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "latin.md").write_bytes("# Año\n".encode("latin-1"))
    [(name, text)] = collect(tmp_path)["docs"]
    assert name == "latin.md"
    assert "\ufffd" in text
