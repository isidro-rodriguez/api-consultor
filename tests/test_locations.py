import shutil
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from api_consultor import render
from api_consultor.cli import app
from api_consultor.config import SETTINGS, default_docs_dir, user_data_dir
from api_consultor.records import Chunk, relativize
from api_consultor.store import FORMAT_VERSION, Store

from .conftest import EMBED_DIM, StubOllama

runner = CliRunner()
VECTOR = [1.0] + [0.0] * (EMBED_DIM - 1)


def test_relativize_only_touches_paths_under_the_root(tmp_path: Path) -> None:
    inside = Chunk("a", "p", "1", "x", str(tmp_path / "pkg" / "a.py"), 3)
    outside = Chunk("b", "p", "1", "x", str(tmp_path.parent / "otro.py"))
    empty = Chunk("c", "p", "1", "x")
    result = relativize([inside, outside, empty], tmp_path)
    assert [chunk.path for chunk in result] == [
        "pkg/a.py",
        str(tmp_path.parent / "otro.py"),
        "",
    ]
    assert result[0].line == 3


def test_default_docs_dir_lives_in_the_user_data_dir() -> None:
    assert default_docs_dir() == user_data_dir() / "docs"


@pytest.fixture
def v3_index(tmp_path: Path) -> Path:
    """Un índice del formato 3: rutas absolutas y sin `root`/`python`."""
    path = tmp_path / "old.db"
    site = tmp_path / "venv" / "site-packages"
    notes = tmp_path / "docs" / "notas"
    with Store(path, EMBED_DIM) as store:
        store.replace(
            "pkg",
            [Chunk("pkg.f", "pkg", "1", "x", str(site / "pkg" / "__init__.py"), 2)],
            [VECTOR],
        )
        store.replace(
            "mod", [Chunk("mod", "mod", "1", "x", str(site / "mod.py"))], [VECTOR]
        )
        store.replace(
            "notas",
            [
                Chunk("a/b.md#Uso", "notas", "1", "x", str(notes / "a" / "b.md"), 1),
                Chunk("c.md [2]", "notas", "1", "y", str(notes / "c.md"), 1),
            ],
            [VECTOR, VECTOR],
            kind="docs",
        )
        store.replace("raro", [Chunk("r", "raro", "1", "x", "relativa.py")], [VECTOR])
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE sources DROP COLUMN root")
    db.execute("ALTER TABLE sources DROP COLUMN python")
    db.execute("UPDATE meta SET value = '3' WHERE key = 'format_version'")
    db.commit()
    db.close()
    return path


def test_v3_index_is_migrated_to_relative_paths(v3_index: Path, tmp_path: Path) -> None:
    with Store(v3_index, EMBED_DIM) as store:
        roots = {s.package: s.root for s in store.sources()}
        paths = {name: store.paths(name) for name in roots}
        assert store.meta()["format_version"] == str(FORMAT_VERSION)
    site = str(tmp_path / "venv" / "site-packages")
    notes = str(tmp_path / "docs" / "notas")
    normalized = {name: Path(root) if root else None for name, root in roots.items()}
    assert normalized == {
        "pkg": Path(site),
        "mod": Path(site),
        "notas": Path(notes),
        "raro": None,
    }
    assert paths["pkg"] == ["pkg/__init__.py"]
    assert paths["mod"] == ["mod.py"]
    assert paths["notas"] == ["a/b.md", "c.md"]
    assert paths["raro"] == ["relativa.py"]


def test_relocate_changes_only_the_root(tmp_path: Path) -> None:
    with Store(tmp_path / "x.db", EMBED_DIM) as store:
        store.replace("a", [Chunk("a", "a", "1", "x", "f.md")], [VECTOR], root="/viejo")
        assert store.relocate("a", "/nuevo")
        assert not store.relocate("no-existe", "/nuevo")
        assert [s.root for s in store.sources()] == ["/nuevo"]
        assert store.paths("a") == ["f.md"]


@pytest.fixture
def other_docs(isolated_cwd: Path) -> Path:
    folder = isolated_cwd / "otras" / "notas"
    folder.mkdir(parents=True)
    (folder / "os.md").write_text("# os.listdir\n\nLista un directorio.\n", "utf-8")
    return folder.parent


BOX_CHARS = str.maketrans(dict.fromkeys("│─╭╮╰╯", " "))


def flat(text: str) -> str:
    """Une las líneas que Rich parte por el ancho de la consola, sin bordes de panel."""
    return " ".join(text.translate(BOX_CHARS).split())


def source_root() -> str:
    with Store(SETTINGS.db_path, EMBED_DIM) as store:
        return store.sources()[0].root


def test_index_docs_keeps_the_original_location(
    other_docs: Path, stub_ollama: StubOllama
) -> None:
    result = runner.invoke(app, ["index-docs", str(other_docs)])
    assert result.exit_code == 0, result.output
    assert not SETTINGS.docs_dir.exists()
    assert Path(source_root()) == (other_docs / "notas").resolve()


def test_index_docs_defaults_to_the_user_folder(
    isolated_cwd: Path, stub_ollama: StubOllama
) -> None:
    (SETTINGS.docs_dir / "notas").mkdir(parents=True)
    (SETTINGS.docs_dir / "notas" / "a.md").write_text("# A\n\ntexto\n", "utf-8")
    result = runner.invoke(app, ["index-docs"])
    assert result.exit_code == 0, result.output
    assert Path(source_root()) == (SETTINGS.docs_dir / "notas").resolve()


def test_missing_default_folder_explains_where_to_put_the_docs(
    isolated_cwd: Path,
) -> None:
    result = runner.invoke(app, ["index-docs"])
    assert result.exit_code == 1
    assert "index-docs <carpeta>" in flat(result.output)


@pytest.fixture
def notes_index(other_docs: Path, stub_ollama: StubOllama) -> Path:
    runner.invoke(app, ["index-docs", str(other_docs)])
    return other_docs / "notas"


def test_ask_flags_sources_whose_file_disappeared(
    notes_index: Path, stub_ollama: StubOllama
) -> None:
    ok = runner.invoke(app, ["ask", "os.listdir"])
    assert "no disponible" not in ok.output
    shutil.rmtree(notes_index)
    gone = runner.invoke(app, ["ask", "os.listdir"])
    assert gone.exit_code == 0, gone.output
    assert "fichero no disponible" in gone.output
    assert "api sources" in gone.output


def test_sources_shows_moved_folders_and_relocate_fixes_them(
    notes_index: Path, isolated_cwd: Path
) -> None:
    moved = isolated_cwd / "movida" / "notas"
    moved.parent.mkdir()
    shutil.move(notes_index, moved)
    assert "no encontrada" in runner.invoke(app, ["sources"]).output
    result = runner.invoke(app, ["sources", "--relocate", "notas", str(moved)])
    assert result.exit_code == 0, result.output
    assert "1 de 1 ficheros" in flat(result.output)
    assert "no encontrada" not in result.output


def test_relocate_rejects_wrong_targets(notes_index: Path, isolated_cwd: Path) -> None:
    empty = isolated_cwd / "vacia"
    empty.mkdir()
    wrong = runner.invoke(app, ["sources", "--relocate", "notas", str(empty)])
    assert wrong.exit_code == 1
    assert "ningún fichero" in flat(wrong.output)
    unknown = runner.invoke(app, ["sources", "--relocate", "nada", str(empty)])
    assert unknown.exit_code == 1
    assert "no indexada" in unknown.output
    not_a_dir = runner.invoke(app, ["sources", "--relocate", "notas", str(empty / "x")])
    assert not_a_dir.exit_code == 1
    assert "No existe la carpeta" in not_a_dir.output


def test_panel_marks_unavailable_files_without_a_link(
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunk = Chunk("a.md#T", "notas", "1", "x", "a.md")
    render.show_sources([chunk], unavailable=lambda _chunk: True)
    output = capsys.readouterr().out
    assert "fichero no disponible" in output
    assert "Ctrl+clic" not in output
