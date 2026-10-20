"""ApiConsultor: consultor de APIs Python offline con Ollama y RAG."""

from importlib import metadata

try:
    __version__ = metadata.version("api-consultor")
except metadata.PackageNotFoundError:  # ejecutado desde el código sin instalar
    __version__ = "0+desconocida"

__all__ = ["__version__"]
