from pathlib import Path

import pytest

from api_consultor import userconfig
from api_consultor.store import Chunk
from api_consultor.userconfig import (
    UserConfig,
    UserConfigError,
    chunk_link,
    config_path,
    default_command,
    editor_command,
    load,
)

CHUNK = Chunk("a.f", "a", "1", "x", str(Path("/tmp/mi proyecto/a.py").resolve()), 42)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("win32", ("edit", "{file}")),
        ("linux", ("nano", "+{line}", "{file}")),
        ("darwin", ("open", "-t", "{file}")),
    ],
)
def test_default_editor_per_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected: tuple[str, ...]
) -> None:
    monkeypatch.setattr(userconfig.sys, "platform", platform)
    assert default_command() == expected
    assert UserConfig().command == expected


def test_env_var_overrides_the_config_path(isolated_user_config: Path) -> None:
    assert config_path() == isolated_user_config


@pytest.mark.parametrize(
    ("platform", "env", "expected_tail"),
    [
        ("win32", "APPDATA", ("api-consultor", "config.toml")),
        ("linux", "XDG_CONFIG_HOME", ("api-consultor", "config.toml")),
        ("darwin", "HOME", ("Application Support", "api-consultor", "config.toml")),
    ],
)
def test_config_path_per_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    env: str,
    expected_tail: tuple[str, ...],
) -> None:
    monkeypatch.delenv("API_CONSULTOR_CONFIG")
    monkeypatch.setattr(userconfig.sys, "platform", platform)
    monkeypatch.setenv(env, str(tmp_path))
    monkeypatch.setattr(userconfig.Path, "home", lambda: tmp_path)
    path = config_path()
    assert path.parts[-len(expected_tail) :] == expected_tail
    assert path.is_relative_to(tmp_path)


def test_missing_file_gives_defaults(isolated_user_config: Path) -> None:
    config = load()
    assert config.source is None
    assert config.link == "{uri}"


def test_file_overrides_editor_and_link(isolated_user_config: Path) -> None:
    isolated_user_config.parent.mkdir(parents=True)
    isolated_user_config.write_text(
        '[editor]\ncommand = ["code", "--goto", "{file}:{line}"]\n'
        'link = "vscode://file/{path}:{line}"\n',
        encoding="utf-8",
    )
    config = load()
    assert config.command == ("code", "--goto", "{file}:{line}")
    assert config.source == isolated_user_config
    assert editor_command(config, CHUNK)[:2] == ["code", "--goto"]
    assert editor_command(config, CHUNK)[2].endswith("a.py:42")
    url = chunk_link(config, CHUNK)
    assert url is not None
    assert url.startswith("vscode://file/")
    assert url.endswith("a.py:42")
    assert "%20" in url  # los espacios van codificados


def test_default_link_is_a_file_uri() -> None:
    url = chunk_link(UserConfig(), CHUNK)
    assert url is not None
    assert url.startswith("file://")


def test_line_is_never_below_one() -> None:
    chunk = Chunk("a", "a", "1", "x", CHUNK.path, 0)
    assert editor_command(UserConfig(("nano", "+{line}")), chunk) == ["nano", "+1"]


def test_no_link_without_a_usable_path() -> None:
    assert chunk_link(UserConfig(), Chunk("a", "a", "1", "x")) is None
    assert chunk_link(UserConfig(), Chunk("a", "a", "1", "x", "relativa.py", 1)) is None
    assert chunk_link(UserConfig(link=""), CHUNK) is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[editor\n", "No se pudo leer"),
        ("editor = 3\n", "debe ser una tabla"),
        ("[editor]\ncomand = []\n", "claves desconocidas"),
        ("[editor]\ncommand = 'nano'\n", "lista de textos"),
        ("[editor]\ncommand = [1]\n", "lista de textos"),
        ("[editor]\nlink = 3\n", "`link` debe ser un texto"),
        ('[editor]\ncommand = ["x", "{nada}"]\n', "Plantilla inválida"),
        ('[editor]\nlink = "{0}"\n', "Plantilla inválida"),
    ],
)
def test_invalid_files_are_reported(
    isolated_user_config: Path, content: str, message: str
) -> None:
    isolated_user_config.parent.mkdir(parents=True)
    isolated_user_config.write_text(content, encoding="utf-8")
    with pytest.raises(UserConfigError, match=message):
        load()
