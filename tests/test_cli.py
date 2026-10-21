import platform
from pathlib import Path

import ollama
import pytest
from typer.testing import CliRunner

from api_consultor import __version__
from api_consultor.cli import app
from api_consultor.config import SETTINGS
from api_consultor.store import Chunk, Store

from .conftest import EMBED_DIM, StubOllama, fake_embed

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


DB = Path("data/consultor.db")


def seed_json_index(version: str, model: str = "bge-m3") -> None:
    """Índice con un único paquete (`json`) indexado con la versión indicada."""
    chunks = [Chunk("json.dumps", "json", version, "def json.dumps(obj)\n\nserializa")]
    with Store(DB, EMBED_DIM, model) as store:
        store.replace("json", chunks, fake_embed([c.content for c in chunks]))


def test_sources_shows_state_versions_and_embedding_metadata(
    indexed: StubOllama,
) -> None:
    seed_json_index("3.0.0")
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0, result.output
    assert "obsoleta" in result.output
    assert "3.0.0" in result.output
    assert "docs" in result.output
    assert "Embeddings: bge-m3" in result.output
    assert "1024 dimensiones" in result.output


def test_ask_warns_about_stale_sources_and_marks_them(stub_ollama: StubOllama) -> None:
    seed_json_index("3.0.0")
    result = runner.invoke(app, ["ask", "json.dumps"])
    assert result.exit_code == 0, result.output
    assert "posiblemente obsoleta" in result.output
    assert "api update" in result.output
    assert "⚠ obsoleta" in result.output  # marca en el panel de fuentes
    assert len(stub_ollama.messages) == 1


def test_ask_does_not_warn_when_everything_is_current(stub_ollama: StubOllama) -> None:
    seed_json_index(platform.python_version())
    result = runner.invoke(app, ["ask", "json.dumps"])
    assert "obsoleta" not in result.output


def test_update_reindexes_only_stale_packages(stub_ollama: StubOllama) -> None:
    seed_json_index("3.0.0")
    result = runner.invoke(app, ["update", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Reindexado" in result.output
    after = runner.invoke(app, ["sources"])
    assert "obsoleta" not in after.output
    assert platform.python_version() in after.output


def test_update_asks_for_confirmation(stub_ollama: StubOllama) -> None:
    seed_json_index("3.0.0")
    result = runner.invoke(app, ["update"], input="n\n")
    assert result.exit_code == 1
    assert "3.0.0" in runner.invoke(app, ["sources"]).output


def test_update_reports_when_everything_is_current(stub_ollama: StubOllama) -> None:
    seed_json_index(platform.python_version())
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "al día" in result.output


def test_a_different_embedding_model_blocks_ask_but_not_sources(
    stub_ollama: StubOllama,
) -> None:
    seed_json_index("3.0.0", model="otro-modelo")
    blocked = runner.invoke(app, ["ask", "hola"])
    assert blocked.exit_code == 1
    assert "otro-modelo" in blocked.output
    assert "reconstruye" in blocked.output
    assert runner.invoke(app, ["sources"]).exit_code == 0


def test_indexing_fails_early_when_the_model_dimension_does_not_match(
    docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "api_consultor.cli.embed", lambda texts: [[0.0] * 8 for _ in texts]
    )
    result = runner.invoke(app, ["index-docs"])
    assert result.exit_code == 1
    assert "8" in result.output
    assert "dimensiones" in result.output


def test_index_records_the_installed_version(stub_ollama: StubOllama) -> None:
    assert runner.invoke(app, ["index", "json"]).exit_code == 0
    assert platform.python_version() in runner.invoke(app, ["sources"]).output


def test_ask_abstains_without_generating_when_evidence_is_weak(
    indexed: StubOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    from api_consultor.config import SETTINGS

    patched = dataclasses.replace(SETTINGS, max_distance=0.1)
    monkeypatch.setattr("api_consultor.consultor.SETTINGS", patched)
    result = runner.invoke(app, ["ask", "zzzz qqqq xxxx"])
    assert result.exit_code == 0, result.output
    assert "No encuentro documentación suficiente" in result.output
    assert "Motivo" in result.output
    assert indexed.messages == []


def test_ask_debug_shows_candidates_signals_and_context_size(
    indexed: StubOllama,
) -> None:
    result = runner.invoke(app, ["ask", "--debug", "os.listdir"])
    assert result.exit_code == 0, result.output
    assert "Candidatos" in result.output
    assert "Mejor distancia" in result.output
    assert "Umbral" in result.output
    assert "desactivado" in result.output
    assert "caracteres" in result.output
    assert "os.md#os.listdir" in result.output
    assert "Respuesta de prueba." in result.output


def test_ask_debug_also_explains_an_abstention(
    indexed: StubOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    from api_consultor.config import SETTINGS

    patched = dataclasses.replace(SETTINGS, max_distance=0.1)
    monkeypatch.setattr("api_consultor.consultor.SETTINGS", patched)
    result = runner.invoke(app, ["ask", "-d", "zzzz qqqq xxxx"])
    assert "Abstención" in result.output
    assert indexed.messages == []


def test_ask_verifies_citations_in_the_answer(indexed: StubOllama) -> None:
    indexed.tokens = ["Usa listdir [DOC-01] ", "y también [DOC-99]."]
    result = runner.invoke(app, ["ask", "os.listdir"])
    assert result.exit_code == 0, result.output
    assert "DOC-01 ✔" in result.output
    assert "inexistentes (DOC-99)" in result.output


def test_sources_remove_deletes_after_confirmation(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["sources", "--remove", "mis-notas"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Eliminada mis-notas" in result.output
    assert "Nada indexado" in result.output


def test_sources_remove_declined_keeps_the_index(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["sources", "-r", "mis-notas"], input="n\n")
    assert result.exit_code == 1
    assert "mis-notas" in runner.invoke(app, ["sources"]).output


def test_sources_remove_yes_skips_confirmation(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["sources", "-r", "mis-notas", "-r", "mis-notas", "-y"])
    assert result.exit_code == 0, result.output
    assert result.output.count("Eliminada mis-notas") == 1


def test_sources_remove_unknown_changes_nothing(indexed: StubOllama) -> None:
    result = runner.invoke(app, ["sources", "-r", "mis-notas", "-r", "fantasma", "-y"])
    assert result.exit_code == 1
    assert "fantasma" in result.output
    assert "Disponibles: mis-notas" in result.output
    assert "mis-notas" in runner.invoke(app, ["sources"]).output


class Launched:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, check: bool) -> None:
        assert check is False
        self.commands.append(command)


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> Launched:
    recorder = Launched()
    monkeypatch.setattr("api_consultor.cli.subprocess.run", recorder.run)
    return recorder


def write_user_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def seed(chunks: list[Chunk]) -> None:
    vector = [1.0] + [0.0] * (SETTINGS.embed_dim - 1)
    with Store(SETTINGS.db_path, SETTINGS.embed_dim, SETTINGS.embed_model) as store:
        store.replace("z", chunks, [vector] * len(chunks))


def test_open_launches_the_configured_editor_at_the_line(
    indexed: StubOllama, launched: Launched, isolated_user_config: Path
) -> None:
    write_user_config(
        isolated_user_config, '[editor]\ncommand = ["ed", "+{line}", "{file}"]\n'
    )
    result = runner.invoke(app, ["open", "os.md#os.listdir"])
    assert result.exit_code == 0, result.output
    command = launched.commands[0]
    assert command[:2] == ["ed", "+1"]
    assert command[2].endswith("os.md")


def test_open_accepts_a_short_unique_suffix(
    indexed: StubOllama, launched: Launched
) -> None:
    assert runner.invoke(app, ["open", "listdir"]).exit_code == 0
    assert len(launched.commands) == 1


def test_open_unknown_name_fails_cleanly(
    indexed: StubOllama, launched: Launched
) -> None:
    result = runner.invoke(app, ["open", "inexistente"])
    assert result.exit_code == 1
    assert "No hay nada indexado" in result.output
    assert not launched.commands


def test_open_reports_a_missing_editor(
    indexed: StubOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr("api_consultor.cli.subprocess.run", missing)
    result = runner.invoke(app, ["open", "os.md#os.listdir"])
    assert result.exit_code == 1
    assert "No se pudo ejecutar el editor" in result.output
    assert "api config" in result.output


def test_open_chunks_without_a_file_are_explained(
    stub_ollama: StubOllama, launched: Launched
) -> None:
    seed([Chunk("z.md#T", "z", "1", "x")])
    result = runner.invoke(app, ["open", "z.md#T"])
    assert result.exit_code == 1
    assert "no tiene fichero asociado" in result.output
    assert not launched.commands


def test_open_ambiguous_names_list_the_options(
    stub_ollama: StubOllama, launched: Launched
) -> None:
    seed(
        [
            Chunk("z.a.run", "z", "1", "x", "/p/a.py", 1),
            Chunk("z.b.run", "z", "1", "y", "/p/b.py", 1),
        ]
    )
    result = runner.invoke(app, ["open", "run"])
    assert result.exit_code == 1
    assert "ambiguo" in result.output
    assert "z.a.run" in result.output
    assert not launched.commands


def test_config_shows_defaults_and_path() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "(no existe)" in result.output
    assert "{file}" in result.output


def test_invalid_config_is_reported_by_open(
    indexed: StubOllama, isolated_user_config: Path
) -> None:
    write_user_config(isolated_user_config, "[editor]\nlinks = 1\n")
    result = runner.invoke(app, ["open", "os.md#os.listdir"])
    assert result.exit_code == 1
    assert "claves desconocidas" in result.output


def test_ask_reports_an_invalid_config(
    indexed: StubOllama, isolated_user_config: Path
) -> None:
    write_user_config(isolated_user_config, "[editor]\nlinks = 1\n")
    result = runner.invoke(app, ["ask", "os.listdir"])
    assert result.exit_code == 1
    assert "claves desconocidas" in result.output
