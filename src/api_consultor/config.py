"""Configuración central."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = "api-consultor"
DATA_ENV_VAR = "API_CONSULTOR_DATA"  # carpeta de datos alternativa (también para tests)


def user_data_dir() -> Path:
    """Carpeta de datos del usuario: índice y documentación extraída de .zip."""
    override = os.environ.get(DATA_ENV_VAR)
    if override is not None:
        return Path(override)
    home = Path.home()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / APP_DIR


def default_db_path() -> Path:
    return user_data_dir() / "consultor.db"


@dataclass(slots=True)
class Settings:
    """Valores por defecto; `config.toml` los sobrescribe al arrancar la CLI."""

    chat_model: str = "api-consultor"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024  # dimensión de bge-m3; cámbiala si cambias de modelo
    db_path: Path = field(default_factory=default_db_path)
    docs_dir: Path = Path("docs")
    retrieval_k: int = 5  # fragmentos que se envían al modelo
    retrieval_candidates: int = 30  # candidatos por método antes de fusionar
    max_per_symbol: int = 2  # fragmentos de un mismo símbolo en el contexto
    # Distancia vectorial máxima para responder sin coincidencia de nombre; None
    # desactiva el umbral. Calíbrala con `api ask --debug` (depende del modelo).
    max_distance: float | None = None
    console_width: int = 90  # ancho máximo de la salida en consola


SETTINGS = Settings()
