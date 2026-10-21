"""Modelos de Ollama que necesita la aplicación y su `Modelfile` empaquetado."""

import re
from importlib import resources
from importlib.resources.abc import Traversable

import ollama


def modelfile() -> Traversable:
    """Devuelve el `Modelfile` que se distribuye con el paquete.

    Returns:
        El recurso del fichero; usar `importlib.resources.as_file` para tener una ruta.
    """
    return resources.files("api_consultor").joinpath("Modelfile")


def base_model(text: str | None = None) -> str:
    """Extrae el modelo base (`FROM`) de un `Modelfile`.

    Args:
        text: Contenido del `Modelfile`; por defecto, el empaquetado.

    Returns:
        El nombre del modelo base, p. ej. `qwen3:8b`.

    Raises:
        ValueError: Si el `Modelfile` no tiene una línea `FROM`.
    """
    content = modelfile().read_text(encoding="utf-8") if text is None else text
    found = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE | re.IGNORECASE)
    if found is None:
        message = "El Modelfile no tiene una línea FROM."
        raise ValueError(message)
    return str(found.group(1))


def installed_models() -> set[str]:
    """Lista los modelos descargados en Ollama.

    Returns:
        Los nombres completos, con etiqueta (`bge-m3:latest`).

    Raises:
        ConnectionError: Si no se puede contactar con Ollama.
    """
    return {model.model for model in ollama.list().models if model.model is not None}


def has_model(installed: set[str], name: str) -> bool:
    """Comprueba si un modelo está entre los instalados.

    Args:
        installed: Nombres devueltos por `installed_models`.
        name: Nombre buscado; sin etiqueta equivale a `:latest`.

    Returns:
        `True` si está descargado.
    """
    return (name if ":" in name else f"{name}:latest") in installed
