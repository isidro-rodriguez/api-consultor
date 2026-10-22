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

from .config import APP_DIR, Settings
from .records import Chunk

ENV_VAR = "API_CONSULTOR_CONFIG"
"""Variable de entorno con una ruta alternativa a `config.toml` (también para tests)."""

DEFAULT_LINK = "{uri}"
"""Plantilla de enlace por defecto: un `file://` al fichero."""

LINK_FIELDS = "{file} {path} {uri} {line}"
"""Campos que admite la plantilla de enlace."""


class UserConfigError(Exception):
    """El fichero de preferencias no se puede usar.

    Args:
        message: Qué ocurre.
        hint: Cómo solucionarlo, si se sabe.

    Attributes:
        hint: Sugerencia para el usuario, o `None`.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True, slots=True)
class Option:
    """Una opción de `config.toml` y el campo de `Settings` que gobierna.

    Attributes:
        table: Sección del TOML (`retrieval`).
        key: Clave dentro de la sección (`k`).
        setting: Campo de `Settings` que sobrescribe (`retrieval_k`).
        kind: Tipo esperado: `str`, `int`, `float` o `Path`.
        help: Descripción que aparece en la plantilla.
        positive: Si los números deben ser mayores que 0.
    """

    table: str
    key: str
    setting: str
    kind: type
    help: str
    positive: bool = True  # números: deben ser > 0

    @property
    def name(self) -> str:
        """Nombre completo de la opción, p. ej. `retrieval.k`."""
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
    Option(
        "retrieval",
        "translate",
        "translate",
        bool,
        "Traduce al inglés las preguntas en español para buscar (una llamada al LLM).",
    ),
    Option("display", "width", "console_width", int, "Ancho máximo de la salida."),
)
"""Opciones que admite `config.toml` (además de `[links] url`)."""

_TABLES = {option.table for option in OPTIONS} | {"links"}


def config_path() -> Path:
    """Devuelve la ruta de `config.toml` según el sistema.

    Respeta `API_CONSULTOR_CONFIG`; si no, usa `%APPDATA%` (Windows),
    `~/Library/Application Support` (macOS) o `$XDG_CONFIG_HOME` / `~/.config`.

    Returns:
        Ruta del fichero (puede no existir).
    """
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
    """Valores leídos de `config.toml` (solo los que el usuario definió).

    Attributes:
        overrides: Valores del usuario por campo de `Settings`.
        link: Plantilla del hiperenlace de las fuentes.
        source: Fichero leído, o `None` si no existe.
    """

    overrides: dict[str, object] = field(default_factory=dict)
    link: str = DEFAULT_LINK
    source: Path | None = None

    def apply(self, settings: Settings) -> None:
        """Sobrescribe `settings` con los valores del usuario.

        Args:
            settings: Configuración a modificar en el sitio.
        """
        for name, value in self.overrides.items():
            setattr(settings, name, value)


def _convert(option: Option, value: object, path: Path) -> object:
    where = f"{path}: [{option.table}] {option.key}"
    if option.kind in (str, Path):
        if not isinstance(value, str) or not value.strip():
            raise UserConfigError(f"{where} debe ser un texto no vacío.")
        return Path(value).expanduser() if option.kind is Path else value
    if option.kind is bool:
        if not isinstance(value, bool):
            raise UserConfigError(f"{where} debe ser true o false.")
        return value
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or (option.kind is int and not isinstance(value, int))
    ):
        raise UserConfigError(f"{where} debe ser de tipo {option.kind.__name__}.")
    if value <= 0:
        raise UserConfigError(f"{where} debe ser mayor que 0.")
    return value


def load(path: Path | None = None) -> UserConfig:
    """Lee y valida `config.toml`.

    Args:
        path: Fichero a leer; por defecto, el de `config_path()`.

    Returns:
        Las preferencias del usuario (vacías si el fichero no existe).

    Raises:
        UserConfigError: Si el fichero no se puede leer, tiene secciones o claves
            desconocidas, valores de tipo incorrecto o una plantilla de enlace inválida.
    """
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


def chunk_file(root: str, chunk: Chunk) -> Path | None:
    """Calcula la ruta absoluta del fichero de un fragmento.

    Args:
        root: Carpeta raíz de la fuente (`SourceInfo.root`).
        chunk: Fragmento con la ruta relativa a esa raíz (o absoluta).

    Returns:
        La ruta, o `None` si el fragmento no tiene fichero o no se puede resolver.
    """
    if not chunk.path:
        return None
    path = Path(root) / chunk.path  # una ruta absoluta sustituye a la raíz
    return path if path.is_absolute() else None


def chunk_link(config: UserConfig, chunk: Chunk, root: str = "") -> str | None:
    """Construye el hiperenlace de terminal a la ubicación de un fragmento.

    Args:
        config: Preferencias con la plantilla de enlace.
        chunk: Fragmento del que se quiere el enlace.
        root: Carpeta raíz de su fuente.

    Returns:
        La URL, o `None` si el fragmento no tiene fichero o este ya no existe.
    """
    file = chunk_file(root, chunk)
    if file is None or not file.exists():
        return None
    url = config.link.format(**_fields(str(file), chunk.line))
    return url if url else None


def effective(config: UserConfig, settings: Settings) -> list[tuple[str, str, bool]]:
    """Lista los valores efectivos de cada opción de `config.toml`.

    Args:
        config: Preferencias del usuario.
        settings: Configuración ya con esas preferencias aplicadas.

    Returns:
        Tuplas (opción, valor, definido por el usuario); `—` si no hay valor.
    """
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
    """Genera el contenido de un `config.toml` nuevo.

    Returns:
        Una plantilla con cada opción documentada y comentada con su valor por defecto.
    """
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
    if isinstance(value, bool):
        return "true" if value else "false"
    return f'"{value}"' if isinstance(value, str) else str(value)


def create(path: Path | None = None) -> Path:
    """Escribe la plantilla de `config.toml`, creando la carpeta si hace falta.

    Args:
        path: Dónde escribirla; por defecto, `config_path()`.

    Returns:
        La ruta del fichero creado.
    """
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template(), encoding="utf-8")
    return path
