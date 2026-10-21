"""Capa fina sobre el cliente de Ollama."""

from collections.abc import Iterator, Sequence

import ollama

from .config import SETTINGS


def embed(texts: Sequence[str]) -> list[list[float]]:
    """Calcula los embeddings de unos textos con el modelo configurado.

    Args:
        texts: Textos a vectorizar.

    Returns:
        Un vector por texto, en el mismo orden.

    Raises:
        ConnectionError: Si no se puede contactar con Ollama.
        ollama.ResponseError: Si Ollama rechaza la petición (p. ej. falta el modelo).
    """
    response = ollama.embed(model=SETTINGS.embed_model, input=list(texts))
    return [list(vector) for vector in response.embeddings]


def stream_chat(messages: list[dict[str, str]]) -> Iterator[str]:
    """Genera la respuesta del modelo de chat, fragmento a fragmento.

    Args:
        messages: Conversación en el formato de Ollama (`role` y `content`).

    Yields:
        Los trozos de texto de la respuesta según llegan (sin bloques `<think>`).

    Raises:
        ConnectionError: Si no se puede contactar con Ollama.
        ollama.ResponseError: Si Ollama rechaza la petición (p. ej. falta el modelo).
    """
    stream = ollama.chat(
        model=SETTINGS.chat_model,
        messages=messages,
        stream=True,
        think=False,  # Qwen3: sin bloques <think> en la respuesta
    )
    for part in stream:
        yield part.message.content or ""
