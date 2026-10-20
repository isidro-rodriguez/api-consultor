"""Configuración central."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    chat_model: str = "api-consultor"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024  # dimensión de bge-m3; cámbiala si cambias de modelo
    db_path: Path = Path("data/consultor.db")
    docs_dir: Path = Path("docs")
    retrieval_k: int = 5  # fragmentos que se envían al modelo
    retrieval_candidates: int = 30  # candidatos por método antes de fusionar
    console_width: int = 90  # ancho máximo de la salida en consola


SETTINGS = Settings()
