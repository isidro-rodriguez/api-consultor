from pathlib import Path

import pytest

from api_consultor import config
from api_consultor.config import Settings, default_db_path, user_data_dir


@pytest.fixture(autouse=True)
def real_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DATA_ENV_VAR, raising=False)


@pytest.mark.parametrize(
    ("platform", "env", "expected_tail"),
    [
        ("win32", "LOCALAPPDATA", ("api-consultor",)),
        ("linux", "XDG_DATA_HOME", ("api-consultor",)),
        ("darwin", "HOME", ("Application Support", "api-consultor")),
    ],
)
def test_user_data_dir_per_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    env: str,
    expected_tail: tuple[str, ...],
) -> None:
    monkeypatch.setattr(config.sys, "platform", platform)
    monkeypatch.setenv(env, str(tmp_path))
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    path = user_data_dir()
    assert path.parts[-len(expected_tail) :] == expected_tail
    assert path.is_relative_to(tmp_path)


def test_linux_falls_back_to_local_share(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    assert user_data_dir() == tmp_path / ".local" / "share" / "api-consultor"


def test_env_var_overrides_the_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(config.DATA_ENV_VAR, str(tmp_path))
    assert user_data_dir() == tmp_path


def test_default_database_lives_in_the_user_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(config.DATA_ENV_VAR, str(tmp_path))
    assert default_db_path() == tmp_path / "consultor.db"
    assert Settings().db_path == tmp_path / "consultor.db"
    assert not Settings().db_path.is_relative_to(Path.cwd()) or tmp_path == Path.cwd()
