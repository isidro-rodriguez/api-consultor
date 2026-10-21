"""Configuración central."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """Valores por defecto; `config.toml` los sobrescribe al arrancar la CLI."""

    chat_model: str = "api-consultor"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024  # dimensión de bge-m3; cámbiala si cambias de modelo
    db_path: Path = Path("data/consultor.db")
    docs_dir: Path = Path("docs")
    retrieval_k: int = 5  # fragmentos que se envían al modelo
    retrieval_candidates: int = 30  # candidatos por método antes de fusionar
    max_per_symbol: int = 2  # fragmentos de un mismo símbolo en el contexto
    # Distancia vectorial máxima para responder sin coincidencia de nombre; None
    # desactiva el umbral. Calíbrala con `api ask --debug` (depende del modelo).
    max_distance: float | None = None
    console_width: int = 90  # ancho máximo de la salida en consola


SETTINGS = Settings()
