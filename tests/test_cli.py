from pathlib import Path

import ollama
import pytest
from typer.testing import CliRunner

from api_consultor import __version__
from api_consultor.cli import app

from .conftest import StubOllama

runner = CliRunner()

NOTES = "# os.listdir\n\nDevuelve los nombres de las entradas de un directorio.\n"


@pytest.fixture
def docs(isolated_cwd: Path) -> Path:
    folder = isolated_cwd / "docs" / "mis-notas"
    folder.mkdir(parents=True)
    (folder / "os.md").write_text(NOTES, encoding="utf-8")
    return folder


@pytest.fixture
def indexed(docs: Path, stub_ollama: StubOllama) -> StubOllama:
    result = runner.invoke(app, ["index-docs"])
    assert result.exit_code == 0, result.output
    return stub_ollama


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "index-docs", "ask", "sources", "--version"):
        assert command in result.output


def test_version_flag_shows_panel() -> None:
    for flag in ("--version", "-V"):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0
        assert f"v{__version__}" in result.output
        assert "Embeddings" in result.output


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_index_docs_then_sources(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    assert "mis-notas" in result.output


def test_sources_without_index_says_so() -> None:
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    assert "Nada indexado" in result.output


def test_ask_answers_and_lists_sources(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["ask", "¿qué hace os.listdir?"])
    assert result.exit_code == 0, result.output
    assert "Respuesta de prueba." in result.output
    assert "Fuentes" in result.output
    assert "os.md#os.listdir" in result.output
    assert len(indexed.messages) == 1


def test_ask_interactive_mode(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["ask"], input="os.listdir\nsalir\n")
    assert result.exit_code == 0, result.output
    assert "Modo interactivo" in result.output
    assert len(indexed.messages) == 1


def test_ask_blank_question_enters_interactive_mode(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["ask", "   "], input="salir\n")
    assert "Modo interactivo" in result.output
    assert indexed.messages == []


def test_ask_without_index_fails_with_hint() -> None:
    result = runner.invoke(app, ["ask", "hola"])
    assert result.exit_code == 1
    assert "nada indexado" in result.output.lower()
    assert "api index" in result.output


def test_ask_reports_unreachable_ollama(
    indexed: StubOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(_texts: object) -> None:
        raise ConnectionError

    monkeypatch.setattr("api_consultor.consultor.embed", down)
    result = runner.invoke(app, ["ask", "hola"])
    assert result.exit_code == 1
    assert "No se pudo conectar con Ollama" in result.output


def test_ask_reports_missing_model(
    indexed: StubOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_texts: object) -> None:
        raise ollama.ResponseError("model 'bge-m3' not found", 404)

    monkeypatch.setattr("api_consultor.consultor.embed", missing)
    result = runner.invoke(app, ["ask", "hola"])
    assert result.exit_code == 1
    assert "Falta un modelo en Ollama" in result.output
    assert "ollama pull" in result.output


def test_index_docs_rejects_missing_directory() -> None:
    result = runner.invoke(app, ["index-docs", "no_existe"])
    assert result.exit_code == 2


def test_index_docs_exclude_patterns(docs: Path, stub_ollama: StubOllama) -> None:
    result = runner.invoke(app, ["index-docs", "-x", "*.md"])
    assert result.exit_code == 0
    assert "nada indexable" in result.output


def test_index_unknown_module_warns_and_continues(stub_ollama: StubOllama) -> None:
    result = runner.invoke(app, ["index", "modulo_inexistente_xyz", "json"])
    assert result.exit_code == 0, result.output
    assert "módulo no encontrado" in result.output
    assert "json" in result.output
