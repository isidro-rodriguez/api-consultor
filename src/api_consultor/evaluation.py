"""Evaluación objetiva de la recuperación: Recall@k y MRR sobre casos de referencia."""

import tomllib
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
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

    Attributes:
        question: Pregunta que se lanza a la recuperación.
        expected: Fuentes que deberían aparecer (nombres, patrones o alternativas).
        category: Etiqueta para agrupar las métricas.
    """

    question: str
    expected: tuple[str, ...]
    category: str = "general"


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
    """

    case: EvalCase
    ranked: tuple[str, ...]

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
        """Categorías presentes, ordenadas."""
        return sorted({result.case.category for result in self.results})

    def metrics(self, category: str | None = None) -> Metrics:
        """Calcula las métricas medias (MRR sobre el mayor de `ks`).

        Args:
            category: Categoría a medir; `None` para todos los casos.

        Returns:
            Las métricas, o ceros si no hay casos.
        """
        selected = [
            r for r in self.results if category is None or r.case.category == category
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
            Los casos con rango superior a `within` o sin acierto.
        """
        return [r for r in self.results if r.rank is None or r.rank > within]


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
    retrieve: Callable[[str, int], list[Chunk]],
    cases: Sequence[EvalCase],
    ks: Sequence[int] = DEFAULT_KS,
    on_result: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    """Ejecuta la recuperación para cada caso y agrega las métricas.

    Args:
        retrieve: Función `(pregunta, k) -> fragmentos`.
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
        chunks = retrieve(case.question, sorted_ks[-1])
        result = CaseResult(case, tuple(chunk.qualname for chunk in chunks))
        results.append(result)
        if on_result is not None:
            on_result(result)
    return EvalReport(tuple(results), sorted_ks)


def load_cases(path: Path) -> list[EvalCase]:
    """Lee casos de un TOML con tablas `[[case]]` (question, expected, category).

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
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(item, str) for item in expected)
        ):
            message = f"{path}: caso {index}: 'expected' debe ser una lista de textos"
            raise ValueError(message)
        if not isinstance(category, str):
            message = f"{path}: caso {index}: 'category' debe ser un texto"
            raise ValueError(message)
        cases.append(
            EvalCase(question, tuple(str(item) for item in expected), category)
        )
    return cases
