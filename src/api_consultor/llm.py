"""Capa fina sobre el cliente de Ollama."""

from collections.abc import Iterator, Sequence

import ollama

from .config import SETTINGS


def embed(texts: Sequence[str]) -> list[list[float]]:
    response = ollama.embed(model=SETTINGS.embed_model, input=list(texts))
    return [list(vector) for vector in response.embeddings]


def stream_chat(messages: list[dict[str, str]]) -> Iterator[str]:
    stream = ollama.chat(
        model=SETTINGS.chat_model,
        messages=messages,
        stream=True,
        think=False,  # Qwen3: sin bloques <think> en la respuesta
    )
    for part in stream:
        yield part.message.content or ""
