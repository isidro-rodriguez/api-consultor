"""Configuración central."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = "api-consultor"
"""Nombre de la carpeta de la aplicación en los directorios del usuario."""

DATA_ENV_VAR = "API_CONSULTOR_DATA"
"""Variable de entorno con una carpeta de datos alternativa (también para tests)."""


def user_data_dir() -> Path:
    """Devuelve la carpeta de datos del usuario (índice y documentación extraída).

    Respeta `API_CONSULTOR_DATA`; si no, usa la ubicación estándar del sistema:
    `%LOCALAPPDATA%` (Windows), `~/Library/Application Support` (macOS) o
    `$XDG_DATA_HOME` / `~/.local/share` (resto).

    Returns:
        Ruta de la carpeta `api-consultor` (puede no existir todavía).
    """
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
    """Devuelve la ruta por defecto de la base de datos del índice."""
    return user_data_dir() / "consultor.db"


@dataclass(slots=True)
class Settings:
    """Parámetros de la aplicación.

    Son los valores por defecto; `config.toml` los sobrescribe al arrancar la CLI.

    Attributes:
        chat_model: Modelo de chat de Ollama (el definido en el `Modelfile`).
        embed_model: Modelo de embeddings de Ollama.
        embed_dim: Dimensión de los vectores de `embed_model`.
        db_path: Fichero de la base de datos del índice.
        docs_dir: Carpeta con la documentación local a indexar.
        retrieval_k: Fragmentos que se envían al modelo.
        retrieval_candidates: Candidatos por método antes de fusionar.
        max_per_symbol: Máximo de fragmentos de un mismo símbolo en el contexto.
        max_distance: Distancia vectorial máxima para responder sin coincidencia de
            nombre; `None` desactiva el umbral.
        console_width: Ancho máximo de la salida en consola.
    """

    chat_model: str = "api-consultor"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    db_path: Path = field(default_factory=default_db_path)
    docs_dir: Path = Path("docs")
    retrieval_k: int = 5
    retrieval_candidates: int = 30
    max_per_symbol: int = 2
    max_distance: float | None = None  # calíbralo con `api ask --debug`
    console_width: int = 90


SETTINGS = Settings()
"""Configuración única de la aplicación (la CLI le aplica `config.toml`)."""
