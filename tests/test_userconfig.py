import re
from pathlib import Path

import pytest

from api_consultor import userconfig
from api_consultor.config import Settings
from api_consultor.store import Chunk
from api_consultor.userconfig import (
    OPTIONS,
    UserConfig,
    UserConfigError,
    chunk_link,
    config_path,
    create,
    effective,
    load,
    template,
)

CHUNK = Chunk("a.f", "a", "1", "x", str(Path("/tmp/mi proyecto/a.py").resolve()), 42)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def test_missing_file_gives_defaults() -> None:
    config = load()
    assert config.source is None
    assert config.overrides == {}
    assert config.link == "{uri}"


def test_file_values_override_settings(isolated_user_config: Path) -> None:
    write(
        isolated_user_config,
        '[models]\nchat = "otro"\nembed_dim = 768\n[paths]\ndb = "x/y.db"\n'
        "[retrieval]\nk = 3\nmax_distance = 0.7\n[display]\nwidth = 100\n"
        '[links]\nurl = "vscode://file/{path}:{line}"\n',
    )
    config = load()
    settings = Settings()
    config.apply(settings)
    assert settings.chat_model == "otro"
    assert settings.embed_dim == 768
    assert settings.db_path == Path("x/y.db")
    assert settings.retrieval_k == 3
    assert settings.max_distance == 0.7
    assert settings.console_width == 100
    assert settings.embed_model == Settings().embed_model  # lo no indicado se conserva
    assert config.source == isolated_user_config
    assert config.link == "vscode://file/{path}:{line}"


def test_integers_are_accepted_where_floats_are_expected(
    isolated_user_config: Path,
) -> None:
    write(isolated_user_config, "[retrieval]\nmax_distance = 1\n")
    assert load().overrides == {"max_distance": 1}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[retrieval\n", "No se pudo leer"),
        ("[nada]\nx = 1\n", "secciones desconocidas"),
        ("retrieval = 3\n", "secciones desconocidas|debe ser una tabla"),
        ("[retrieval]\nkk = 1\n", "claves desconocidas en \\[retrieval\\]"),
        ("[retrieval]\nk = 0\n", "mayor que 0"),
        ("[retrieval]\nk = -2\n", "mayor que 0"),
        ("[retrieval]\nk = 2.5\n", "de tipo int"),
        ("[retrieval]\nk = true\n", "de tipo int"),
        ('[retrieval]\nk = "3"\n', "de tipo int"),
        ("[retrieval]\nmax_distance = 0\n", "mayor que 0"),
        ('[retrieval]\nmax_distance = "x"\n', "de tipo float"),
        ("[models]\nchat = 3\n", "texto no vacío"),
        ('[models]\nchat = " "\n', "texto no vacío"),
        ("[paths]\ndb = 1\n", "texto no vacío"),
        ("[links]\nurl = 3\n", "debe ser un texto"),
        ('[links]\nurl = "{nada}"\n', "Plantilla de enlace inválida"),
        ('[links]\nurl = "{0}"\n', "Plantilla de enlace inválida"),
        ("[links]\nopen = 1\n", "claves desconocidas en \\[links\\]"),
    ],
)
def test_invalid_files_are_reported(
    isolated_user_config: Path, content: str, message: str
) -> None:
    write(isolated_user_config, content)
    with pytest.raises(UserConfigError, match=message):
        load()


def test_table_that_is_not_a_table_is_reported(isolated_user_config: Path) -> None:
    write(isolated_user_config, "[links]\n")
    write(isolated_user_config, 'models = "x"\n')
    with pytest.raises(UserConfigError, match="debe ser una tabla"):
        load()


def test_default_link_is_a_file_uri() -> None:
    url = chunk_link(UserConfig(), CHUNK)
    assert url is not None
    assert url.startswith("file://")


def test_custom_link_encodes_spaces_and_line() -> None:
    config = UserConfig(link="vscode://file/{path}:{line}")
    url = chunk_link(config, CHUNK)
    assert url is not None
    assert url.startswith("vscode://file/")
    assert url.endswith("a.py:42")
    assert "%20" in url


def test_line_is_never_below_one() -> None:
    chunk = Chunk("a", "a", "1", "x", CHUNK.path, 0)
    url = chunk_link(UserConfig(link="{line}"), chunk)
    assert url == "1"


def test_no_link_without_a_usable_path() -> None:
    assert chunk_link(UserConfig(), Chunk("a", "a", "1", "x")) is None
    assert chunk_link(UserConfig(), Chunk("a", "a", "1", "x", "relativa.py", 1)) is None
    assert chunk_link(UserConfig(link=""), CHUNK) is None


def test_effective_lists_every_option_and_flags_custom_ones(
    isolated_user_config: Path,
) -> None:
    write(isolated_user_config, "[retrieval]\nk = 3\n")
    config = load()
    settings = Settings()
    config.apply(settings)
    rows = {
        name: (value, custom) for name, value, custom in effective(config, settings)
    }
    assert len(rows) == len(OPTIONS) + 1
    assert rows["retrieval.k"] == ("3", True)
    assert rows["display.width"] == ("90", False)
    assert rows["retrieval.max_distance"] == ("—", False)
    assert rows["links.url"] == ("{uri}", False)


def test_template_is_valid_and_documents_every_option() -> None:
    text = template()
    for option in OPTIONS:
        assert f"{option.key} = " in text
        assert option.help in text


def test_template_values_match_the_defaults(isolated_user_config: Path) -> None:
    """Descomentar toda la plantilla (salvo el umbral) no cambia nada."""
    lines = [
        re.sub(r"^# (?=\w+ = )", "", line)
        for line in template().splitlines()
        if not line.startswith("# max_distance")
    ]
    write(isolated_user_config, "\n".join(lines))
    config = load()
    settings = Settings()
    config.apply(settings)
    assert settings == Settings()
    assert config.link == "{uri}"


def test_create_writes_a_loadable_template(isolated_user_config: Path) -> None:
    assert create() == isolated_user_config
    assert isolated_user_config.read_text(encoding="utf-8") == template()
    assert load().overrides == {}


def test_paths_expand_the_home_directory(isolated_user_config: Path) -> None:
    write(isolated_user_config, '[paths]\ndb = "~/indice.db"\n')
    assert load().overrides["db_path"] == Path("~/indice.db").expanduser()
