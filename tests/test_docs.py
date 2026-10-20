import zipfile
from pathlib import Path

from api_consultor.docs import (
    CHUNK_CHARS,
    _sections,
    _split,
    collect,
    extract_doc_chunks,
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
    sections = list(_sections(SPHINX))
    assert [title for title, _ in sections] == [
        "",
        '"os" --- Interfaces del sistema',
        "Archivos",
    ]
    assert "os.listdir" in sections[2][1]


def test_sections_detect_atx_headings_but_not_inside_fences() -> None:
    text = (
        "# Titulo\n\ncuerpo\n\n"
        "```python\n# comentario, no titulo\n```\n\n"
        "## Sub\n\nfin\n"
    )
    titles = [title for title, _ in _sections(text)]
    assert titles == ["", "Titulo", "Sub"]


def test_sections_ignore_table_borders() -> None:
    text = "Nombre   Valor\n=======  =====\nuno      1\n"
    assert [title for title, _ in _sections(text)] == [""]


def test_split_respects_limit_without_cutting_paragraphs() -> None:
    paragraphs = [f"parrafo {i} " + "x" * 390 for i in range(10)]
    chunks = list(_split("\n\n".join(paragraphs), 1000))
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert all(paragraph in "\n\n".join(chunks) for paragraph in paragraphs)


def test_split_wraps_giant_paragraphs_by_lines() -> None:
    giant = "\n".join(f"linea {i}" for i in range(600))
    chunks = list(_split(giant, 500))
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
