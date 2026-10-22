"""Evaluación objetiva de la recuperación: Recall@k, MRR y abstención."""

import math
import tomllib
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import pairwise
from pathlib import Path

from .store import Chunk

DEFAULT_KS = (1, 5, 10)
"""Valores de k con los que `api eval` informa Recall@k."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Pregunta de referencia y fuentes que deberían recuperarse.

    `expected` son nombres exactos o patrones `fnmatch` (`*`) de `Chunk.qualname`.
    Cada elemento admite alternativas separadas por `|` (basta con que acierte una),
    p. ej. `"json.dumps|library/json.txt#*"`: la API indexada o su página de docs.
    Vacío significa que el índice **no** responde a la pregunta (p. ej. una función
    inexistente): lo correcto es abstenerse.

    Attributes:
        question: Pregunta que se lanza a la recuperación.
        expected: Fuentes que deberían aparecer (nombres, patrones o alternativas).
        category: Etiqueta para agrupar las métricas.
    """

    question: str
    expected: tuple[str, ...]
    category: str = "general"

    @property
    def answerable(self) -> bool:
        """Indica si el índice contiene la respuesta (hay fuentes esperadas)."""
        return bool(self.expected)


@dataclass(frozen=True, slots=True)
class Hits:
    """Lo que devuelve la recuperación para evaluarla, abstención incluida.

    Attributes:
        chunks: Fragmentos elegidos (sin aplicar la abstención), de mejor a peor.
        distance: Distancia con la que se decide abstenerse: la del vecino más
            cercano (`math.inf` si no hay ninguno), o `None` si una coincidencia de
            nombre hace que se responda con cualquier umbral.
        forced: Si se abstiene con cualquier umbral (p. ej. pregunta por un símbolo
            que no existe).
    """

    chunks: Sequence[Chunk]
    distance: float | None = None
    forced: bool = False


def _matches(qualname: str, pattern: str) -> bool:
    """Coincide con alguna de las alternativas (`a|b`), exactas o con comodines."""
    return any(
        qualname == option or fnmatchcase(qualname, option)
        for option in pattern.split("|")
    )


def first_relevant_rank(ranked: Sequence[str], expected: Sequence[str]) -> int | None:
    """Busca el primer resultado relevante de un ranking.

    Args:
        ranked: Nombres cualificados recuperados, de mejor a peor.
        expected: Patrones esperados (ver `EvalCase`).

    Returns:
        La posición (desde 1), o `None` si ninguno aparece.
    """
    for position, qualname in enumerate(ranked, start=1):
        if any(_matches(qualname, pattern) for pattern in expected):
            return position
    return None


def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Calcula Recall@k: fracción de lo esperado que aparece entre los `k` primeros.

    Args:
        ranked: Nombres cualificados recuperados, de mejor a peor.
        expected: Patrones esperados; no puede estar vacío.
        k: Cuántos resultados se consideran.

    Returns:
        Un valor entre 0 y 1.
    """
    top = ranked[:k]
    found = sum(any(_matches(q, pattern) for q in top) for pattern in expected)
    return found / len(expected)


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    """Calcula el rango recíproco del primer resultado relevante.

    Args:
        ranked: Nombres cualificados recuperados, de mejor a peor.
        expected: Patrones esperados (ver `EvalCase`).

    Returns:
        `1 / posición`, o 0 si ninguno aparece.
    """
    rank = first_relevant_rank(ranked, expected)
    return 0.0 if rank is None else 1 / rank


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Lo que recuperó un caso de evaluación.

    Attributes:
        case: El caso evaluado.
        ranked: Nombres cualificados recuperados, de mejor a peor.
        distance: Distancia de abstención (ver `Hits`); `None` si nunca se abstiene.
        forced: Si se abstiene con cualquier umbral (ver `Hits`).
    """

    case: EvalCase
    ranked: tuple[str, ...]
    distance: float | None = None
    forced: bool = False

    def abstains(self, threshold: float | None) -> bool:
        """Indica si la recuperación se abstendría con ese umbral (`max_distance`).

        Args:
            threshold: Umbral de distancia; `None` desactiva la abstención.

        Returns:
            `True` si se abstiene siempre, o si no hay coincidencia de nombre y la
            distancia supera el umbral.
        """
        return self.forced or (
            threshold is not None
            and self.distance is not None
            and self.distance > threshold
        )

    @property
    def rank(self) -> int | None:
        """Posición del primer resultado relevante, o `None` si no aparece."""
        return first_relevant_rank(self.ranked, self.case.expected)

    def recall(self, k: int) -> float:
        """Devuelve Recall@k de este caso."""
        return recall_at_k(self.ranked, self.case.expected, k)

    @property
    def reciprocal_rank(self) -> float:
        """Rango recíproco de este caso."""
        return reciprocal_rank(self.ranked, self.case.expected)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Métricas medias de un conjunto de casos.

    Attributes:
        cases: Número de casos.
        recall: Recall@k medio para cada k.
        mrr: Rango recíproco medio (MRR).
    """

    cases: int
    recall: dict[int, float]
    mrr: float


@dataclass(frozen=True, slots=True)
class Abstention:
    """Cómo se comporta un umbral de abstención con los casos evaluados.

    Attributes:
        threshold: Umbral (`max_distance`) evaluado; `None` si está desactivado.
        unanswerable: Casos sin respuesta en el índice.
        false_answers: De ellos, cuántos se responderían igualmente (lo grave).
        by_name: De esos, cuántos pasan por coincidencia de nombre (ningún umbral
            los detiene).
        answerable: Casos con respuesta.
        lost: De ellos, cuántos se callarían habiendo recuperado la fuente correcta
            (callar cuando la recuperación falla no es perder una respuesta).
    """

    threshold: float | None
    unanswerable: int
    false_answers: int
    by_name: int
    answerable: int
    lost: int

    @property
    def false_answer_rate(self) -> float:
        """Fracción de preguntas sin respuesta que se responden (0 si no hay)."""
        return self.false_answers / self.unanswerable if self.unanswerable else 0.0

    @property
    def lost_rate(self) -> float:
        """Fracción de preguntas con respuesta que se callan (0 si no hay)."""
        return self.lost / self.answerable if self.answerable else 0.0


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Resultados de una evaluación y sus métricas.

    Attributes:
        results: Resultado de cada caso.
        ks: Valores de k para Recall@k, ordenados.
    """

    results: tuple[CaseResult, ...]
    ks: tuple[int, ...]

    @property
    def categories(self) -> list[str]:
        """Categorías con preguntas que tienen respuesta, ordenadas."""
        return sorted({r.case.category for r in self.results if r.case.answerable})

    def metrics(self, category: str | None = None) -> Metrics:
        """Calcula las métricas medias (MRR sobre el mayor de `ks`).

        Solo cuentan los casos con respuesta: los que no la tienen se miden con
        `abstention`.

        Args:
            category: Categoría a medir; `None` para todos los casos.

        Returns:
            Las métricas, o ceros si no hay casos.
        """
        selected = [
            r
            for r in self.results
            if r.case.answerable and (category is None or r.case.category == category)
        ]
        if not selected:
            return Metrics(0, dict.fromkeys(self.ks, 0.0), 0.0)
        total = len(selected)
        return Metrics(
            cases=total,
            recall={k: sum(r.recall(k) for r in selected) / total for k in self.ks},
            mrr=sum(r.reciprocal_rank for r in selected) / total,
        )

    def failures(self, within: int) -> list[CaseResult]:
        """Lista los casos cuya primera fuente relevante queda fuera del top.

        Args:
            within: Tamaño del top que se considera acierto.

        Returns:
            Los casos con respuesta cuyo rango supera `within` o que no aciertan.
        """
        return [
            r
            for r in self.results
            if r.case.answerable and (r.rank is None or r.rank > within)
        ]

    def abstention(self, threshold: float | None) -> Abstention:
        """Mide un umbral de abstención sobre los casos evaluados.

        Args:
            threshold: Umbral (`max_distance`); `None` para la abstención desactivada.

        Returns:
            Respuestas falsas (casos sin respuesta que se responden) y respuestas
            perdidas (casos con respuesta que se callan).
        """
        negative = [r for r in self.results if not r.case.answerable]
        positive = [r for r in self.results if r.case.answerable]
        answered = [r for r in negative if not r.abstains(threshold)]
        return Abstention(
            threshold,
            len(negative),
            len(answered),
            sum(r.distance is None for r in answered),
            len(positive),
            sum(r.rank is not None and r.abstains(threshold) for r in positive),
        )

    def suggest_threshold(self) -> float | None:
        """Propone el `max_distance` que mejor separa los casos con y sin respuesta.

        Minimiza respuestas falsas más respuestas perdidas; a igualdad, prefiere
        menos respuestas falsas (es peor responder mal que callar). Prueba los
        puntos medios entre distancias consecutivas, para dejar margen a ambos lados.

        Returns:
            El umbral (redondeado a milésimas), o `None` si faltan casos con o sin
            respuesta o no hay distancias que separar.
        """
        finite = sorted(
            {
                r.distance
                for r in self.results
                if not r.forced and r.distance is not None and math.isfinite(r.distance)
            }
        )
        has_both = {r.case.answerable for r in self.results} == {True, False}
        if not has_both or not finite:
            return None
        options = [
            *((a + b) / 2 for a, b in pairwise(finite)),
            finite[-1],  # todo lo finito responde (solo calla lo que no tiene vecino)
        ]

        def cost(threshold: float) -> tuple[int, int]:
            result = self.abstention(threshold)
            return result.false_answers + result.lost, result.false_answers

        return round(min(options, key=cost), 3)


def missing_sources(
    cases: Sequence[EvalCase], qualnames: Collection[str]
) -> list[tuple[EvalCase, str]]:
    """Busca fuentes esperadas que no coinciden con nada de lo indexado.

    Esos casos no pueden acertar: suelen indicar que falta indexar algo.

    Args:
        cases: Casos de evaluación.
        qualnames: Nombres cualificados presentes en el índice.

    Returns:
        Pares (caso, patrón) de cada patrón sin coincidencias.
    """
    return [
        (case, pattern)
        for case in cases
        for pattern in case.expected
        if not any(_matches(name, pattern) for name in qualnames)
    ]


def evaluate(
    retrieve: Callable[[str, int], Sequence[Chunk] | Hits],
    cases: Sequence[EvalCase],
    ks: Sequence[int] = DEFAULT_KS,
    on_result: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    """Ejecuta la recuperación para cada caso y agrega las métricas.

    Args:
        retrieve: Función `(pregunta, k) -> fragmentos`, o `Hits` para medir
            también la abstención.
        cases: Casos a evaluar.
        ks: Valores de k para Recall@k.
        on_result: Llamada opcional tras cada caso (p. ej. para una barra de progreso).

    Returns:
        El informe con los resultados de todos los casos.

    Raises:
        ValueError: Si `ks` está vacío o contiene valores menores que 1.
    """
    sorted_ks = tuple(sorted(set(ks)))
    if not sorted_ks or sorted_ks[0] < 1:
        message = "ks debe contener enteros positivos"
        raise ValueError(message)
    results: list[CaseResult] = []
    for case in cases:
        found = retrieve(case.question, sorted_ks[-1])
        hits = found if isinstance(found, Hits) else Hits(found)
        ranked = tuple(chunk.qualname for chunk in hits.chunks)
        result = CaseResult(case, ranked, hits.distance, hits.forced)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return EvalReport(tuple(results), sorted_ks)


def load_cases(path: Path) -> list[EvalCase]:
    """Lee casos de un TOML con tablas `[[case]]` (question, expected, category).

    `expected = []` marca una pregunta sin respuesta en el índice (ver `EvalCase`).

    Args:
        path: Fichero TOML con los casos.

    Returns:
        Los casos, en el orden del fichero.

    Raises:
        ValueError: Si algún caso está mal formado.
        tomllib.TOMLDecodeError: Si el fichero no es TOML válido.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = data.get("case", [])
    if not isinstance(raw, list):
        message = f"{path}: 'case' debe ser una lista de tablas ([[case]])"
        raise ValueError(message)
    cases: list[EvalCase] = []
    for index, entry in enumerate(raw, start=1):
        question = entry.get("question") if isinstance(entry, dict) else None
        expected = entry.get("expected") if isinstance(entry, dict) else None
        category = entry.get("category", "general") if isinstance(entry, dict) else None
        if not isinstance(question, str) or not question.strip():
            message = f"{path}: caso {index}: falta 'question' (texto)"
            raise ValueError(message)
        if not isinstance(expected, list) or not all(
            isinstance(item, str) for item in expected
        ):
            message = (
                f"{path}: caso {index}: 'expected' debe ser una lista de textos "
                "([] si el índice no tiene la respuesta)"
            )
            raise ValueError(message)
        if not isinstance(category, str):
            message = f"{path}: caso {index}: 'category' debe ser un texto"
            raise ValueError(message)
        cases.append(
            EvalCase(question, tuple(str(item) for item in expected), category)
        )
    return cases
