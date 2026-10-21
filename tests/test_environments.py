import json
import shutil
import sys
import venv
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from api_consultor import external, probe
from api_consultor.cli import app
from api_consultor.config import SETTINGS
from api_consultor.external import (
    ProbeError,
    extract_remote,
    is_own,
    remote_versions,
    resolve_interpreter,
)
from api_consultor.records import Chunk
from api_consultor.store import Store
from api_consultor.versions import check_versions

from .conftest import EMBED_DIM, PackageFactory, StubOllama

runner = CliRunner()

CODE = '''
"""Paquete de prueba."""

def saluda():
    """Saluda a alguien."""
'''


@pytest.fixture(scope="session")
def venv_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Entorno virtual real y vacío (sin pip) para probar el Python ajeno."""
    folder = tmp_path_factory.mktemp("envs") / ".venv"
    venv.create(folder, with_pip=False)
    return folder


@pytest.fixture
def foreign(venv_dir: Path) -> Generator[Callable[[str, str], Path]]:
    """Instala un paquete en el entorno ajeno; devuelve su carpeta de paquetes."""
    created: list[Path] = []

    def install(name: str, code: str) -> Path:
        python = resolve_interpreter(venv_dir)
        purelib = next(venv_dir.glob("lib/python*/site-packages"), None) or (
            venv_dir / "Lib" / "site-packages"
        )
        assert python.is_file()
        (purelib / name).mkdir(exist_ok=True)
        (purelib / name / "__init__.py").write_text(code, encoding="utf-8")
        info = purelib / f"{name}-1.0.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
        )
        created.extend([purelib / name, info])
        return purelib

    yield install
    for folder in created:
        shutil.rmtree(folder, ignore_errors=True)


def test_resolve_interpreter_accepts_venv_and_executable(venv_dir: Path) -> None:
    python = resolve_interpreter(venv_dir)
    assert python.is_file()
    assert resolve_interpreter(python) == python


def test_resolve_interpreter_rejects_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No existe"):
        resolve_interpreter(tmp_path / "nada")
    with pytest.raises(FileNotFoundError, match="intérprete"):
        resolve_interpreter(tmp_path)


def test_is_own() -> None:
    assert is_own("")
    assert is_own(sys.executable)
    assert not is_own("/otro/python")


def test_extract_remote_returns_relative_paths(
    venv_dir: Path, foreign: Callable[[str, str], Path]
) -> None:
    purelib = foreign("ajeno_pkg", CODE)
    python = str(resolve_interpreter(venv_dir))
    extraction = extract_remote(python, "ajeno_pkg")
    assert extraction.python == python
    assert Path(extraction.root) == purelib.resolve()
    paths = {chunk.path for chunk in extraction.chunks}
    assert paths == {"ajeno_pkg/__init__.py"}
    assert {chunk.qualname for chunk in extraction.chunks} >= {"ajeno_pkg.saluda"}


def test_extract_remote_reports_missing_and_broken_packages(
    venv_dir: Path, foreign: Callable[[str, str], Path]
) -> None:
    python = str(resolve_interpreter(venv_dir))
    with pytest.raises(ModuleNotFoundError) as missing:
        extract_remote(python, "no_esta_instalado_xyz")
    assert missing.value.name == "no_esta_instalado_xyz"
    foreign("roto_pkg", "raise ImportError('roto a propósito')\n")
    with pytest.raises(ImportError, match="roto a propósito"):
        extract_remote(python, "roto_pkg")


def test_remote_versions(venv_dir: Path) -> None:
    python = str(resolve_interpreter(venv_dir))
    found = remote_versions(python, ["json", "no_esta_instalado_xyz"])
    assert found["json"] is not None
    assert found["no_esta_instalado_xyz"] is None


@pytest.mark.skipif(sys.platform == "win32", reason="guion con shebang")
def test_unusable_interpreters_raise_probe_error(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="No se pudo ejecutar"):
        remote_versions(str(tmp_path / "no-python"), ["json"])
    broken = tmp_path / "falso.py"
    broken.write_text("#!/bin/sh\necho no-json\n", encoding="utf-8")
    broken.chmod(0o755)
    with pytest.raises(ProbeError, match="no respondió"):
        remote_versions(str(broken), ["json"])


def test_probe_rejects_old_python(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe.sys, "version_info", (3, 11, 0, "final", 0))
    assert probe.main(["versions", "json"]) == probe.EXIT_UNSUPPORTED
    assert "3.12" in json.loads(capsys.readouterr().out)["error"]


def test_probe_commands_in_process(
    make_package: PackageFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    name = make_package("sonda_pkg", {"__init__.py": CODE})
    assert probe.main(["extract", name]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["chunks"][0]["path"] == "sonda_pkg/__init__.py"
    assert probe.main(["extract", "no_esta_instalado_xyz"]) == probe.EXIT_NOT_FOUND
    assert "no_esta_instalado_xyz" in capsys.readouterr().out
    assert probe.main(["versions", "json"]) == 0
    assert "json" in json.loads(capsys.readouterr().out)


def test_probe_reports_import_errors(
    make_package: PackageFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    name = make_package("sonda_rota", {"__init__.py": "raise ValueError('mal')\n"})
    assert probe.main(["extract", name]) == probe.EXIT_IMPORT
    assert "mal" in capsys.readouterr().out


def test_probe_uses_the_chunk_dataclass_fields() -> None:
    assert set(Chunk.__slots__) == {
        "qualname",
        "package",
        "version",
        "content",
        "path",
        "line",
    }
    assert external.PROBE.is_file()


def test_versions_are_checked_in_the_indexed_environment(
    venv_dir: Path, foreign: Callable[[str, str], Path], tmp_path: Path
) -> None:
    foreign("ajeno_pkg", CODE)
    python = str(resolve_interpreter(venv_dir))
    from api_consultor.records import SourceInfo

    sources = [
        SourceInfo("ajeno_pkg", "desconocida", 1, python=python),
        SourceInfo("ajeno_pkg", "9.9", 1, python=python),
        SourceInfo("no_esta_xyz", "1", 1, python=python),
        SourceInfo("ajeno_pkg", "1", 1, python=str(tmp_path / "borrado" / "python")),
        SourceInfo("notas", "1", 1, kind="docs", root=str(tmp_path / "movida")),
        SourceInfo("notas", "1", 1, kind="docs", root=str(tmp_path)),
    ]
    states = [status.state for status in check_versions(sources)]
    assert states == ["unknown", "stale", "missing", "gone", "gone", "docs"]


def test_a_failing_probe_marks_the_source_as_gone(
    monkeypatch: pytest.MonkeyPatch, venv_dir: Path
) -> None:
    from api_consultor import versions
    from api_consultor.records import SourceInfo

    def fail(_python: str, _names: object) -> dict[str, str | None]:
        raise ProbeError("x")

    monkeypatch.setattr(versions, "remote_versions", fail)
    python = str(resolve_interpreter(venv_dir))
    [status] = check_versions([SourceInfo("json", "1", 1, python=python)])
    assert status.state == "gone"


def test_index_with_python_records_environment_and_root(
    venv_dir: Path, foreign: Callable[[str, str], Path], stub_ollama: StubOllama
) -> None:
    purelib = foreign("ajeno_pkg", CODE)
    result = runner.invoke(app, ["index", "ajeno_pkg", "--python", str(venv_dir)])
    assert result.exit_code == 0, result.output
    with Store(SETTINGS.db_path, EMBED_DIM) as store:
        [source] = store.sources()
        chunk = store.by_name(["ajeno_pkg.saluda"])[0]
    assert source.python == str(resolve_interpreter(venv_dir))
    assert Path(source.root) == purelib.resolve()
    assert chunk.path == "ajeno_pkg/__init__.py"
    assert "al día" in runner.invoke(app, ["sources"]).output


def test_index_with_python_warns_about_missing_packages(
    venv_dir: Path, stub_ollama: StubOllama
) -> None:
    result = runner.invoke(app, ["index", "no_esta_xyz", "-p", str(venv_dir)])
    assert result.exit_code == 0
    assert "no encontrado" in result.output
    assert "--python" in result.output


def test_index_with_python_warns_about_unusable_interpreters(
    tmp_path: Path, stub_ollama: StubOllama
) -> None:
    fake = tmp_path / "falso"
    fake.write_text("", encoding="utf-8")
    fake.chmod(0o755)
    result = runner.invoke(app, ["index", "json", "-p", str(fake)])
    assert result.exit_code == 0
    assert "json:" in result.output


def test_index_with_a_wrong_environment_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", "json", "-p", str(tmp_path / "nada")])
    assert result.exit_code == 1
    assert "No existe" in result.output


def test_update_reindexes_in_the_original_environment(
    venv_dir: Path, foreign: Callable[[str, str], Path], stub_ollama: StubOllama
) -> None:
    foreign("ajeno_pkg", CODE)
    python = str(resolve_interpreter(venv_dir))
    with Store(SETTINGS.db_path, EMBED_DIM) as store:
        store.replace(
            "ajeno_pkg",
            [Chunk("ajeno_pkg.saluda", "ajeno_pkg", "0.5", "x", "ajeno_pkg/a.py")],
            [[1.0] + [0.0] * (EMBED_DIM - 1)],
            root="/x",
            python=python,
        )
    assert "obsoleta" in runner.invoke(app, ["sources"]).output
    result = runner.invoke(app, ["update", "--yes"])
    assert result.exit_code == 0, result.output
    with Store(SETTINGS.db_path, EMBED_DIM) as store:
        [source] = store.sources()
    assert (source.version, source.python) == ("1.0", python)
    assert "al día" in runner.invoke(app, ["sources"]).output


def test_stdlib_modules_are_public_and_importable_here() -> None:
    from api_consultor.indexer import STDLIB_SKIP, stdlib_modules

    modules = stdlib_modules()
    assert {"json", "pathlib", "shutil", "asyncio"} <= set(modules)
    assert not {"_thread", "this", "test", "antigravity"} & set(modules)
    assert not STDLIB_SKIP & set(modules)
    assert modules == sorted(modules)


def test_probe_lists_the_stdlib(capsys: pytest.CaptureFixture[str]) -> None:
    assert probe.main(["stdlib"]) == 0
    assert "json" in json.loads(capsys.readouterr().out)["modules"]


def test_remote_stdlib(venv_dir: Path, tmp_path: Path) -> None:
    from api_consultor.external import remote_stdlib

    assert "json" in remote_stdlib(str(resolve_interpreter(venv_dir)))
    with pytest.raises(ProbeError):
        remote_stdlib(str(tmp_path / "no-python"))


def test_remote_stdlib_rejects_unexpected_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from api_consultor.external import remote_stdlib

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 3, stdout='{"error": "viejo"}')

    monkeypatch.setattr(external.subprocess, "run", fake_run)
    with pytest.raises(ProbeError, match="viejo"):
        remote_stdlib("python")
