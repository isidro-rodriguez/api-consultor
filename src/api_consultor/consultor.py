"""Flujo RAG: recuperar fragmentos y preguntar al modelo."""

import re
from collections.abc import Iterator

from .config import SETTINGS
from .llm import embed, stream_chat
from .store import Chunk, Store

# Nombres tipo `pathlib.Path.rglob` o `snake_case` dentro de la pregunta.
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|\w*_\w+")


def _format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "<docs/>"
    return "\n".join(
        f'<docs source="{c.qualname}" package="{c.package}" version="{c.version}">\n'
        f"{c.content}\n</docs>"
        for c in chunks
    )


class Consultor:
    def __init__(self, store: Store) -> None:
        self._store = store

    def retrieve(self, question: str, k: int | None = None) -> list[Chunk]:
        """Fragmentos más relevantes (por defecto `SETTINGS.top_k`)."""
        limit = SETTINGS.top_k if k is None else k
        exact = self._store.by_name(_IDENTIFIER.findall(question))
        semantic = self._store.search(embed([question])[0], limit)
        # dict: elimina duplicados conservando el orden (primero los exactos)
        merged = {chunk.qualname: chunk for chunk in (*exact, *semantic)}
        return list(merged.values())[:limit]

    def stream(self, question: str, chunks: list[Chunk]) -> Iterator[str]:
        """Genera la respuesta en streaming a partir de los fragmentos recuperados."""
        content = f"{_format_context(chunks)}\n\nPregunta: {question}"
        yield from stream_chat([{"role": "user", "content": content}])
