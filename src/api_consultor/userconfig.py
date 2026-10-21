"""Preferencias del usuario (`config.toml`) y enlaces a los ficheros de las fuentes.

El fichero vive en el directorio de configuración del sistema y es opcional. Todo lo
que no indique conserva el valor por defecto de `config.Settings`.
"""

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from .config import Settings
from .store import Chunk

APP_DIR = "api-consultor"
ENV_VAR = "API_CONSULTOR_CONFIG"  # ruta alternativa al fichero (también para tests)
DEFAULT_LINK = "{uri}"
LINK_FIELDS = "{file} {path} {uri} {line}"


class UserConfigError(Exception):
    """El fichero de preferencias no se puede usar."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True, slots=True)
class Option:
    """Una opción de `config.toml` y el campo de `Settings` que gobierna."""

    table: str
    key: str
    setting: str
    kind: type
    help: str
    positive: bool = True  # números: deben ser > 0

    @property
    def name(self) -> str:
        return f"{self.table}.{self.key}"


OPTIONS = (
    Option("models", "chat", "chat_model", str, "Modelo de chat de Ollama."),
    Option("models", "embed", "embed_model", str, "Modelo de embeddings de Ollama."),
    Option("models", "embed_dim", "embed_dim", int, "Dimensión de sus vectores."),
    Option("paths", "db", "db_path", Path, "Base de datos del índice."),
    Option("paths", "docs", "docs_dir", Path, "Carpeta de documentación."),
    Option("retrieval", "k", "retrieval_k", int, "Fragmentos enviados al modelo."),
    Option(
        "retrieval", "candidates", "retrieval_candidates", int, "Candidatos por método."
    ),
    Option(
        "retrieval", "max_per_symbol", "max_per_symbol", int, "Máx. por mismo símbolo."
    ),
    Option(
        "retrieval",
        "max_distance",
        "max_distance",
        float,
        "Umbral de distancia para abstenerse (sin definir = desactivado).",
    ),
    Option("display", "width", "console_width", int, "Ancho máximo de la salida."),
)
_TABLES = {option.table for option in OPTIONS} | {"links"}


def config_path() -> Path:
    """Ruta de `config.toml` según el sistema (Windows, macOS o XDG en Linux)."""
    override = os.environ.get(ENV_VAR)
    if override is not None:
        return Path(override)
    home = Path.home()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return base / APP_DIR / "config.toml"


@dataclass(frozen=True, slots=True)
class UserConfig:
    """Valores leídos de `config.toml` (solo los que el usuario definió)."""

    overrides: dict[str, object] = field(default_factory=dict)  # campo de Settings
    link: str = DEFAULT_LINK
    source: Path | None = None  # fichero leído (None = no existe)

    def apply(self, settings: Settings) -> None:
        """Aplica los valores del usuario sobre `settings`."""
        for name, value in self.overrides.items():
            setattr(settings, name, value)


def _convert(option: Option, value: object, path: Path) -> object:
    where = f"{path}: [{option.table}] {option.key}"
    if option.kind in (str, Path):
        if not isinstance(value, str) or not value.strip():
            raise UserConfigError(f"{where} debe ser un texto no vacío.")
        return Path(value) if option.kind is Path else value
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if not number or (option.kind is int and not isinstance(value, int)):
        raise UserConfigError(f"{where} debe ser de tipo {option.kind.__name__}.")
    if value <= 0:  # type: ignore[operator]
        raise UserConfigError(f"{where} debe ser mayor que 0.")
    return value


def load(path: Path | None = None) -> UserConfig:
    """Lee las preferencias; sin fichero devuelve los valores por defecto."""
    path = path or config_path()
    if not path.is_file():
        return UserConfig()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UserConfigError(f"No se pudo leer {path}: {exc}") from exc
    if unknown := sorted(set(data) - _TABLES):
        raise UserConfigError(
            f"{path}: secciones desconocidas: {', '.join(unknown)}.",
            f"Válidas: {', '.join(sorted(_TABLES))}.",
        )
    for table in sorted(_TABLES):
        section = data.get(table, {})
        if not isinstance(section, dict):
            raise UserConfigError(f"{path}: `{table}` debe ser una tabla [{table}].")
        valid = {o.key for o in OPTIONS if o.table == table} or {"url"}
        if bad := sorted(set(section) - valid):
            raise UserConfigError(
                f"{path}: claves desconocidas en [{table}]: {', '.join(bad)}.",
                f"Válidas: {', '.join(sorted(valid))}.",
            )
    overrides = {
        option.setting: _convert(option, data[option.table][option.key], path)
        for option in OPTIONS
        if option.key in data.get(option.table, {})
    }
    link = data.get("links", {}).get("url", DEFAULT_LINK)
    if not isinstance(link, str):
        raise UserConfigError(f"{path}: [links] url debe ser un texto.")
    config = UserConfig(overrides, link, path)
    _validate_link(config)
    return config


def _fields(path: str, line: int) -> dict[str, str]:
    file = Path(path)
    return {
        "file": str(file),
        "path": quote(file.as_posix(), safe="/:"),
        "uri": file.as_uri() if file.is_absolute() else "",
        "line": str(max(line, 1)),
    }


def _validate_link(config: UserConfig) -> None:
    """Falla pronto si la plantilla usa campos que no existen."""
    try:
        config.link.format(**_fields(str(Path.cwd()), 1))
    except (KeyError, IndexError, ValueError) as exc:
        raise UserConfigError(
            f"Plantilla de enlace inválida en {config.source}: {exc!r}.",
            f"Campos disponibles: {LINK_FIELDS}.",
        ) from exc


def chunk_link(config: UserConfig, chunk: Chunk) -> str | None:
    """URL (hiperenlace de terminal) a la ubicación del fragmento, si la tiene."""
    if not chunk.path or not Path(chunk.path).is_absolute():
        return None
    url = config.link.format(**_fields(chunk.path, chunk.line))
    return url if url else None


def effective(config: UserConfig, settings: Settings) -> list[tuple[str, str, bool]]:
    """(opción, valor, definido por el usuario) de cada opción de `config.toml`."""
    rows = [
        (
            option.name,
            "—"
            if getattr(settings, option.setting) is None
            else str(getattr(settings, option.setting)),
            option.setting in config.overrides,
        )
        for option in OPTIONS
    ]
    rows.append(("links.url", config.link, config.link != DEFAULT_LINK))
    return rows


def template() -> str:
    """Contenido de un `config.toml` nuevo: valores por defecto, comentados."""
    defaults = Settings()
    lines = [
        "# Preferencias de ApiConsultor. Descomenta y cambia lo que necesites;",
        "# lo que no esté aquí usa el valor por defecto indicado.",
    ]
    for table in (*dict.fromkeys(o.table for o in OPTIONS), "links"):
        lines += ["", f"[{table}]"]
        if table == "links":
            lines += [
                f"# Hiperenlace de cada fuente. Campos: {LINK_FIELDS}",
                '# p. ej. "vscode://file/{path}:{line}" para abrir en VS Code.',
                f'# url = "{DEFAULT_LINK}"',
                "",
            ]
            continue
        for option in (o for o in OPTIONS if o.table == table):
            value = getattr(defaults, option.setting)
            lines.append(f"# {option.help}")
            if value is None:
                lines.append(f"# {option.key} = 0.8")
            else:
                shown = value.as_posix() if isinstance(value, Path) else value
                lines.append(f"# {option.key} = {_toml(shown)}")
    return "\n".join(lines).rstrip("\n") + "\n"


def _toml(value: object) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


def create(path: Path | None = None) -> Path:
    """Crea `config.toml` con la plantilla (y su carpeta) y devuelve la ruta."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template(), encoding="utf-8")
    return path
