"""Preferencias del usuario (`config.toml`): editor y enlaces de las fuentes.

El fichero vive en el directorio de configuración del sistema y es opcional:

    [editor]
    command = ["code", "--goto", "{file}:{line}"]   # `api open`
    link = "vscode://file/{path}:{line}"            # enlaces del panel «Fuentes»
"""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .store import Chunk

APP_DIR = "api-consultor"
ENV_VAR = "API_CONSULTOR_CONFIG"  # ruta alternativa al fichero (también para tests)
_KNOWN_KEYS = frozenset({"command", "link"})
DEFAULT_LINK = "{uri}"


class UserConfigError(Exception):
    """El fichero de preferencias no se puede usar."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


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


def default_command() -> tuple[str, ...]:
    """Editor por defecto: `edit` en Windows, TextEdit en macOS, `nano` en el resto."""
    if sys.platform == "win32":
        return ("edit", "{file}")
    if sys.platform == "darwin":
        return ("open", "-t", "{file}")
    return ("nano", "+{line}", "{file}")


@dataclass(frozen=True, slots=True)
class UserConfig:
    command: tuple[str, ...] = ()
    link: str = DEFAULT_LINK
    source: Path | None = None  # fichero del que se leyó (None = valores por defecto)

    def __post_init__(self) -> None:
        if not self.command:
            object.__setattr__(self, "command", default_command())


def load(path: Path | None = None) -> UserConfig:
    """Lee las preferencias; sin fichero devuelve los valores por defecto."""
    path = path or config_path()
    if not path.is_file():
        return UserConfig()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UserConfigError(f"No se pudo leer {path}: {exc}") from exc
    editor = data.get("editor", {})
    if not isinstance(editor, dict):
        raise UserConfigError(f"{path}: `editor` debe ser una tabla [editor].")
    if unknown := sorted(set(editor) - _KNOWN_KEYS):
        raise UserConfigError(
            f"{path}: claves desconocidas en [editor]: {', '.join(unknown)}.",
            f"Válidas: {', '.join(sorted(_KNOWN_KEYS))}.",
        )
    command = editor.get("command", [])
    if not isinstance(command, list) or not all(isinstance(a, str) for a in command):
        raise UserConfigError(
            f"{path}: `command` debe ser una lista de textos, "
            'p. ej. ["nano", "{file}"].'
        )
    link = editor.get("link", DEFAULT_LINK)
    if not isinstance(link, str):
        raise UserConfigError(f"{path}: `link` debe ser un texto.")
    config = UserConfig(tuple(command), link, path)
    _validate(config)
    return config


def _fields(path: str, line: int) -> dict[str, str]:
    file = Path(path)
    return {
        "file": str(file),
        "path": quote(file.as_posix(), safe="/:"),
        "uri": file.as_uri() if file.is_absolute() else "",
        "line": str(max(line, 1)),
    }


def _validate(config: UserConfig) -> None:
    """Falla pronto si las plantillas usan campos que no existen."""
    sample = _fields(str(Path.cwd()), 1)
    try:
        for part in (*config.command, config.link):
            part.format(**sample)
    except (KeyError, IndexError, ValueError) as exc:
        raise UserConfigError(
            f"Plantilla inválida en {config.source}: {exc!r}.",
            "Campos disponibles: {file} {path} {uri} {line}.",
        ) from exc


def editor_command(config: UserConfig, chunk: Chunk) -> list[str]:
    """Comando que abre la ubicación del fragmento en el editor configurado."""
    fields = _fields(chunk.path, chunk.line)
    return [part.format(**fields) for part in config.command]


def chunk_link(config: UserConfig, chunk: Chunk) -> str | None:
    """URL (hiperenlace de terminal) a la ubicación del fragmento, si la tiene."""
    if not chunk.path or not Path(chunk.path).is_absolute():
        return None
    url = config.link.format(**_fields(chunk.path, chunk.line))
    return url if url else None
