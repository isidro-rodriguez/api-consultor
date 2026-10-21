import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from api_consultor import models
from api_consultor.cli import app
from api_consultor.config import SETTINGS

runner = CliRunner()


def test_packaged_modelfile_has_base_model() -> None:
    assert models.modelfile().is_file()
    assert models.base_model() == "qwen3:8b"


def test_base_model_parses_text_and_rejects_missing_from() -> None:
    assert models.base_model("# nota\nfrom llama3:8b\nSYSTEM x") == "llama3:8b"
    with pytest.raises(ValueError, match="FROM"):
        models.base_model("SYSTEM x")


def test_has_model_defaults_to_latest_tag() -> None:
    installed = {"bge-m3:latest", "qwen3:8b"}
    assert models.has_model(installed, "bge-m3")
    assert models.has_model(installed, "qwen3:8b")
    assert not models.has_model(installed, "qwen3:14b")


def test_installed_models_reads_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = SimpleNamespace(
        models=[SimpleNamespace(model="a:latest"), SimpleNamespace(model=None)]
    )
    monkeypatch.setattr(models.ollama, "list", lambda: listing)
    assert models.installed_models() == {"a:latest"}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    recorded: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("api_consultor.cli.shutil.which", lambda _: "/usr/bin/ollama")
    monkeypatch.setattr("api_consultor.cli.subprocess.run", fake_run)
    return recorded


def test_setup_pulls_missing_and_creates_model(
    calls: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("api_consultor.cli.installed_models", lambda: {"bge-m3:latest"})
    result = runner.invoke(app, ["setup", "--no-stdlib"])
    assert result.exit_code == 0, result.output
    assert calls[0] == ["ollama", "pull", "qwen3:8b"]
    assert calls[1][:3] == ["ollama", "create", SETTINGS.chat_model]
    assert calls[1][3] == "-f"
    assert len(calls) == 2


def test_setup_without_ollama_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api_consultor.cli.shutil.which", lambda _: None)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code != 0
    assert "ollama.com" in result.output


def test_setup_reports_failing_command(
    monkeypatch: pytest.MonkeyPatch, calls: list[list[str]]
) -> None:
    monkeypatch.setattr("api_consultor.cli.installed_models", lambda: set())
    monkeypatch.setattr(
        "api_consultor.cli.subprocess.run",
        lambda command, *, check: subprocess.CompletedProcess(command, 3),
    )
    result = runner.invoke(app, ["setup"])
    assert result.exit_code != 0
    assert "código 3" in result.output
