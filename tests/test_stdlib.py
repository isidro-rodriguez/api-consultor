from pathlib import Path

import pytest
from typer.testing import CliRunner

from api_consultor.cli import app
from api_consultor.config import SETTINGS
from api_consultor.records import Chunk, SourceInfo
from api_consultor.store import Store
from api_consultor.versions import VersionStatus, collapse_stdlib

from .conftest import EMBED_DIM, StubOllama

runner = CliRunner()


@pytest.fixture
def two_modules(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Simula una biblioteca estándar de dos módulos (la real tardaría minutos)."""
    modules = ["json", "shutil"]
    monkeypatch.setattr("api_consultor.cli.stdlib_modules", lambda: modules)
    return modules


def sources() -> list[SourceInfo]:
    with Store(SETTINGS.db_path, EMBED_DIM) as store:
        return store.sources()


def test_index_stdlib_records_the_kind(
    two_modules: list[str], stub_ollama: StubOllama
) -> None:
    result = runner.invoke(app, ["index", "--stdlib"])
    assert result.exit_code == 0, result.output
    assert {(s.package, s.kind) for s in sources()} == {
        ("json", "stdlib"),
        ("shutil", "stdlib"),
    }


def test_stdlib_modules_are_not_indexed_twice(
    two_modules: list[str], stub_ollama: StubOllama
) -> None:
    result = runner.invoke(app, ["index", "json", "--stdlib"])
    assert result.exit_code == 0, result.output
    assert {s.kind for s in sources()} == {"stdlib"}


def test_index_requires_packages_or_stdlib() -> None:
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 1
    assert "--stdlib" in result.output


def test_index_stdlib_in_another_environment(
    monkeypatch: pytest.MonkeyPatch, stub_ollama: StubOllama, tmp_path: Path
) -> None:
    fake = tmp_path / "python"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr("api_consultor.cli.remote_stdlib", lambda _python: [])
    result = runner.invoke(app, ["index", "--stdlib", "-p", str(fake)])
    assert result.exit_code == 0, result.output
    assert not sources()


def test_index_stdlib_reports_unusable_environments(tmp_path: Path) -> None:
    fake = tmp_path / "python"
    fake.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["index", "--stdlib", "-p", str(fake)])
    assert result.exit_code == 1


def flat(text: str) -> str:
    return " ".join(text.split())


def test_sources_summarises_the_stdlib_unless_asked(
    two_modules: list[str], stub_ollama: StubOllama
) -> None:
    runner.invoke(app, ["index", "--stdlib"])
    short = flat(runner.invoke(app, ["sources"]).output)
    assert "stdlib (2 módulos)" in short
    assert "shutil" not in short
    assert "--all" in short
    full = flat(runner.invoke(app, ["sources", "--all"]).output)
    assert "shutil" in full
    assert "stdlib (" not in full


def test_remove_stdlib_alias_removes_every_module(
    two_modules: list[str], stub_ollama: StubOllama
) -> None:
    runner.invoke(app, ["index", "--stdlib"])
    result = runner.invoke(app, ["sources", "-r", "stdlib", "-y"])
    assert result.exit_code == 0, result.output
    assert not sources()


def test_update_keeps_the_stdlib_kind(stub_ollama: StubOllama) -> None:
    with Store(SETTINGS.db_path, EMBED_DIM) as store:
        store.replace(
            "json",
            [Chunk("json.dumps", "json", "0.0.1", "x")],
            [[1.0] + [0.0] * (EMBED_DIM - 1)],
            kind="stdlib",
        )
    result = runner.invoke(app, ["update", "--yes"])
    assert result.exit_code == 0, result.output
    [source] = sources()
    assert (source.package, source.kind) == ("json", "stdlib")
    assert source.version != "0.0.1"


def status(name: str, state: str, python: str = "", chunks: int = 1) -> VersionStatus:
    info = SourceInfo(
        name, "3.13", chunks, "stdlib", "2026-01-0" + str(chunks), "", python
    )
    return VersionStatus(info, "3.13", state)


def test_collapse_stdlib_groups_by_interpreter_and_keeps_the_worst_state() -> None:
    docs = VersionStatus(SourceInfo("notas", "1", 4, "docs"), None, "docs")
    rows = [
        status("json", "ok", chunks=2),
        docs,
        status("shutil", "stale", chunks=3),
        status("re", "ok", python="/otro", chunks=1),
    ]
    summary = collapse_stdlib(rows)
    assert [r.source.package for r in summary] == [
        "stdlib (2 módulos)",
        "notas",
        "stdlib (1 módulos)",
    ]
    assert (summary[0].state, summary[0].source.chunks) == ("stale", 5)
    assert summary[0].source.indexed_at == "2026-01-03"


def test_setup_offers_the_stdlib_and_defaults_to_yes(
    two_modules: list[str],
    stub_ollama: StubOllama,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api_consultor.cli.shutil.which", lambda _: "/usr/bin/ollama")
    monkeypatch.setattr("api_consultor.cli.installed_models", lambda: set())
    monkeypatch.setattr("api_consultor.cli._ollama", lambda *_args: None)
    result = runner.invoke(app, ["setup"], input="\n")
    assert result.exit_code == 0, result.output
    assert "Indexar ahora la biblioteca estándar" in result.output
    assert len(sources()) == 2
    again = runner.invoke(app, ["setup"])  # ya indexada: no vuelve a preguntar
    assert again.exit_code == 0, again.output
    assert "Indexar ahora" not in again.output


@pytest.mark.parametrize(
    ("args", "input_", "expected"),
    [
        (["setup"], "n\n", 0),
        (["setup", "--no-stdlib"], "", 0),
        (["setup", "--stdlib"], "", 2),
    ],
)
def test_setup_stdlib_answers(
    two_modules: list[str],
    stub_ollama: StubOllama,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    input_: str,
    expected: int,
) -> None:
    monkeypatch.setattr("api_consultor.cli.shutil.which", lambda _: "/usr/bin/ollama")
    monkeypatch.setattr("api_consultor.cli.installed_models", lambda: set())
    monkeypatch.setattr("api_consultor.cli._ollama", lambda *_args: None)
    result = runner.invoke(app, args, input=input_)
    assert result.exit_code == 0, result.output
    assert len(sources()) == expected
