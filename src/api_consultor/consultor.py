"""Flujo RAG: recuperar fragmentos y preguntar al modelo."""

import re
from collections.abc import Iterator, Sequence

from .config import SETTINGS
from .llm import embed, stream_chat
from .store import Chunk, Store

# Nombres tipo `pathlib.Path.rglob` o `snake_case` dentro de la pregunta.
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|\w*_\w+")
_WORD = re.compile(r"[^\W\d]\w{2,}")  # palabras de 3+ letras (o con guion bajo)
_STOPWORDS_TEXT = """
que qué como cómo para por con sin sobre entre cuando cuándo donde dónde cual cuál
cuales los las una uno unos unas del este esta esto estos estas ese esa uso usar usa
hace hacer hago sirve funciona puedo quiero necesito devuelve parámetros parametros
the and how what does use with for from this that
"""
_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

# Fusión de rankings (Reciprocal Rank Fusion): puntuación = peso / (RRF_K + posición).
RRF_K = 60
WEIGHT_QUALIFIED = 3.0  # nombre cualificado en la pregunta (pathlib.Path.rglob)
WEIGHT_SHORT = 1.5  # nombre corto de un símbolo (rglob)
WEIGHT_LEXICAL = 1.0  # palabras en el contenido (BM25)
WEIGHT_SEMANTIC = 1.0  # similitud de embeddings


def _terms(question: str) -> list[str]:
    """Palabras significativas de la pregunta, en minúsculas y sin repetir."""
    words = (word.lower() for word in _WORD.findall(question))
    return list(dict.fromkeys(word for word in words if word not in _STOPWORDS))


def _lexical_query(question: str) -> str | None:
    """Consulta FTS5: frases para `a.b.c` y términos sueltos (OR; BM25 los pondera)."""
    parts: list[str] = []
    for name in _IDENTIFIER.findall(question):
        pieces = re.findall(r"\w+", name)
        if len(pieces) > 1:
            parts.append('"' + " ".join(pieces) + '"')
    parts.extend(f'"{term}"' for term in _terms(question))
    return " OR ".join(dict.fromkeys(parts)) or None


def _fuse(rankings: Sequence[tuple[float, Sequence[Chunk]]]) -> list[Chunk]:
    """Fusiona varios rankings ponderados; los empates conservan el orden de entrada."""
    scores: dict[tuple[str, str], float] = {}
    chunks: dict[tuple[str, str], Chunk] = {}
    for weight, ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            key = (chunk.package, chunk.qualname)
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + position)
            chunks.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunks[key] for key in ordered]


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
        """Fragmentos más relevantes (por defecto `SETTINGS.retrieval_k`).

        Combina cuatro señales: nombres cualificados, nombres cortos de símbolos,
        palabras del contenido (BM25) y similitud semántica, fusionadas con RRF.
        """
        limit = SETTINGS.retrieval_k if k is None else k
        pool = max(SETTINGS.retrieval_candidates, limit)
        terms = _terms(question)
        query = _lexical_query(question)
        rankings = [
            (
                WEIGHT_QUALIFIED,
                self._store.by_name(_IDENTIFIER.findall(question), limit_per_name=3),
            ),
            (WEIGHT_SHORT, self._store.by_short_name(terms)),
            (
                WEIGHT_LEXICAL,
                self._store.lexical(query, pool) if query is not None else [],
            ),
            (WEIGHT_SEMANTIC, self._store.search(embed([question])[0], pool)),
        ]
        return _fuse(rankings)[:limit]

    def stream(self, question: str, chunks: list[Chunk]) -> Iterator[str]:
        """Genera la respuesta en streaming a partir de los fragmentos recuperados."""
        content = f"{_format_context(chunks)}\n\nPregunta: {question}"
        yield from stream_chat([{"role": "user", "content": content}])
