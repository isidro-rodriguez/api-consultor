"""Flujo RAG: recuperar fragmentos y preguntar al modelo."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from html import escape

from .citations import INSTRUCTION, doc_id
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
"""Constante de suavizado de RRF: cuánto pesa la posición en cada ranking."""

WEIGHT_QUALIFIED = 3.0
"""Peso de la señal «nombre cualificado en la pregunta» (`pathlib.Path.rglob`)."""

WEIGHT_SHORT = 1.5
"""Peso de la señal «nombre corto de un símbolo» (`rglob`)."""

WEIGHT_LEXICAL = 1.0
"""Peso de la señal «palabras en el contenido» (BM25)."""

WEIGHT_SEMANTIC = 1.0
"""Peso de la señal «similitud de embeddings»."""


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


_CONTINUATION = re.compile(r" \[\d+\]$")


def _symbol(chunk: Chunk) -> tuple[str, str]:
    """Símbolo al que pertenece un fragmento (las continuaciones `x [2]` son de `x`)."""
    return chunk.package, _CONTINUATION.sub("", chunk.qualname)


def _fuse_detailed(
    rankings: Sequence[tuple[str, float, Sequence[Chunk]]],
) -> list[tuple[Chunk, float, dict[str, int]]]:
    """Fusiona rankings ponderados (RRF).

    Devuelve (fragmento, puntuación, posición en cada señal) de mayor a menor
    puntuación; los empates conservan el orden de entrada.
    """
    scores: dict[tuple[str, str], float] = {}
    chunks: dict[tuple[str, str], Chunk] = {}
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for name, weight, ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            key = (chunk.package, chunk.qualname)
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + position)
            chunks.setdefault(key, chunk)
            ranks.setdefault(key, {})[name] = position
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(chunks[key], scores[key], ranks[key]) for key in ordered]


def _fuse(rankings: Sequence[tuple[float, Sequence[Chunk]]]) -> list[Chunk]:
    named = [(str(i), weight, ranking) for i, (weight, ranking) in enumerate(rankings)]
    return [chunk for chunk, _score, _ranks in _fuse_detailed(named)]


def _diversify(ordered: Sequence[Chunk], limit: int, per_symbol: int) -> list[Chunk]:
    """Elige `limit` fragmentos como mucho `per_symbol` por símbolo.

    Si no alcanzan, rellena con los descartados (más contexto es mejor que menos).
    """
    chosen: list[int] = []
    counts: dict[tuple[str, str], int] = {}
    for index, chunk in enumerate(ordered):
        symbol = _symbol(chunk)
        if counts.get(symbol, 0) < per_symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
            chosen.append(index)
        if len(chosen) == limit:
            break
    if len(chosen) < limit:
        taken = set(chosen)
        extra = [i for i in range(len(ordered)) if i not in taken]
        chosen.extend(extra[: limit - len(chosen)])
    return [ordered[i] for i in sorted(chosen)]


@dataclass(frozen=True, slots=True)
class Candidate:
    """Un candidato de la recuperación con sus señales (para `--debug`).

    Attributes:
        chunk: El fragmento candidato.
        score: Puntuación RRF fusionada.
        ranks: Posición del fragmento en cada señal (`N`, `S`, `L`, `V`).
        distance: Distancia vectorial, si apareció en la búsqueda semántica.
        selected: Si se envía al modelo.
    """

    chunk: Chunk
    score: float
    ranks: dict[str, int]
    distance: float | None
    selected: bool


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Resultado de recuperar: elegidos, candidatos y motivo de abstención.

    Attributes:
        chunks: Fragmentos que se envían al modelo (vacío si hay abstención).
        candidates: Todos los candidatos fusionados, de mejor a peor.
        names: Nombres de símbolos detectados en la pregunta.
        terms: Palabras significativas de la pregunta.
        best_distance: Menor distancia vectorial encontrada, si hubo.
        abstain_reason: Por qué no se responde, o `None` si se responde.
    """

    chunks: list[Chunk]
    candidates: list[Candidate] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    best_distance: float | None = None
    abstain_reason: str | None = None


def format_context(chunks: list[Chunk]) -> str:
    """Da formato XML al contexto que se envía al modelo.

    Cada fragmento va en un bloque `<docs id="DOC-NN" ...>` con sus atributos
    escapados, para que el modelo pueda citarlo por su identificador.

    Args:
        chunks: Fragmentos elegidos, en el orden en que se numeran.

    Returns:
        El contexto, o `<docs/>` si no hay fragmentos.
    """
    if not chunks:
        return "<docs/>"
    return "\n".join(
        f'<docs id="{doc_id(i)}" source="{escape(c.qualname)}" '
        f'package="{escape(c.package)}" version="{escape(c.version)}">\n'
        f"{c.content}\n</docs>"
        for i, c in enumerate(chunks, start=1)
    )


class Consultor:
    """Flujo RAG sobre un índice: recupera fragmentos y pregunta al modelo.

    Args:
        store: Índice abierto en el que buscar.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def retrieve(self, question: str, k: int | None = None) -> list[Chunk]:
        """Devuelve los fragmentos más relevantes para una pregunta.

        Args:
            question: Pregunta del usuario.
            k: Cuántos fragmentos devolver; por defecto `SETTINGS.retrieval_k`.

        Returns:
            Los fragmentos elegidos (vacío si la recuperación se abstiene).
        """
        return self.search(question, k).chunks

    def search(self, question: str, k: int | None = None) -> Retrieval:
        """Recupera con detalle: candidatos, señales y posible abstención.

        Combina cuatro señales (nombres cualificados, nombres cortos, palabras del
        contenido con BM25 y similitud semántica) con RRF, limita los fragmentos por
        símbolo y, si hay umbral (`max_distance`) y no hay coincidencia de nombre ni
        vecinos suficientemente cercanos, se abstiene.

        Args:
            question: Pregunta del usuario.
            k: Cuántos fragmentos elegir; por defecto `SETTINGS.retrieval_k`.

        Returns:
            Los elegidos junto con todos los candidatos y sus señales.

        Raises:
            ConnectionError: Si no se puede contactar con Ollama para el embedding.
            IndexIntegrityError: Si el modelo de embeddings no encaja con el índice.
        """
        limit = SETTINGS.retrieval_k if k is None else k
        pool = max(SETTINGS.retrieval_candidates, limit)
        names = _IDENTIFIER.findall(question)
        terms = _terms(question)
        query = _lexical_query(question)
        qualified = self._store.by_name(names, limit_per_name=3)
        short = self._store.by_short_name(terms)
        lexical = self._store.lexical(query, pool) if query is not None else []
        scored = self._store.search_scored(embed([question])[0], pool)
        distances = {(c.package, c.qualname): d for c, d in scored}
        fused = _fuse_detailed(
            [
                ("N", WEIGHT_QUALIFIED, qualified),
                ("S", WEIGHT_SHORT, short),
                ("L", WEIGHT_LEXICAL, lexical),
                ("V", WEIGHT_SEMANTIC, [chunk for chunk, _ in scored]),
            ]
        )
        selected = _diversify(
            [chunk for chunk, _score, _ranks in fused], limit, SETTINGS.max_per_symbol
        )
        chosen = {(c.package, c.qualname) for c in selected}
        candidates = [
            Candidate(
                chunk,
                score,
                ranks,
                distances.get((chunk.package, chunk.qualname)),
                (chunk.package, chunk.qualname) in chosen,
            )
            for chunk, score, ranks in fused
        ]
        best = min(distances.values(), default=None)
        reason = self._abstain_reason(bool(qualified or short), best)
        return Retrieval(
            [] if reason is not None else selected,
            candidates,
            names,
            terms,
            best,
            reason,
        )

    @staticmethod
    def _abstain_reason(name_match: bool, best: float | None) -> str | None:
        limit = SETTINGS.max_distance
        if limit is None or name_match:
            return None
        if best is None:
            return "no hay nada recuperable"
        if best > limit:
            return f"ningún candidato es suficientemente cercano ({best:.3f} > {limit})"
        return None

    def stream(self, question: str, chunks: list[Chunk]) -> Iterator[str]:
        """Genera la respuesta en streaming a partir de los fragmentos recuperados.

        Args:
            question: Pregunta del usuario.
            chunks: Fragmentos que forman el contexto (los numera como `DOC-NN`).

        Yields:
            Los trozos de la respuesta según los genera el modelo.
        """
        content = f"{format_context(chunks)}\n\n{INSTRUCTION}\n\nPregunta: {question}"
        yield from stream_chat([{"role": "user", "content": content}])
