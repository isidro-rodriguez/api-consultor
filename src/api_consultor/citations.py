"""Citas `[DOC-01]`: identificadores de fragmentos y verificación de la respuesta."""

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\bDOC-(\d+)\b")

INSTRUCTION = (
    "Cita cada afirmación con el identificador del bloque que la respalda, "
    "por ejemplo [DOC-01]. No uses identificadores que no existan."
)
"""Texto que se añade a la pregunta para que el modelo cite los fragmentos."""


def doc_id(position: int) -> str:
    """Devuelve el identificador de un fragmento del contexto.

    Args:
        position: Posición del fragmento en el contexto, empezando en 1.

    Returns:
        El identificador con dos cifras como mínimo, p. ej. `DOC-01`.
    """
    return f"DOC-{position:02d}"


@dataclass(frozen=True, slots=True)
class Citations:
    """Citas de una respuesta frente a los fragmentos que se enviaron al modelo.

    Attributes:
        valid: Posiciones (base 1) citadas que existen, ordenadas y sin repetir.
        invalid: Identificadores citados que no existen, por orden de aparición.
    """

    valid: tuple[int, ...]
    invalid: tuple[str, ...]

    @property
    def any(self) -> bool:
        """Indica si la respuesta contiene alguna cita, válida o no."""
        return bool(self.valid or self.invalid)


def check_citations(answer: str, available: int) -> Citations:
    """Separa las citas de una respuesta en válidas e inventadas.

    Args:
        answer: Texto completo de la respuesta del modelo.
        available: Número de fragmentos enviados; solo `DOC-01`..`DOC-NN` existen.

    Returns:
        Las citas válidas (ordenadas) y los identificadores inventados.
    """
    valid: set[int] = set()
    invalid: list[str] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= available:
            valid.add(number)
        elif (name := f"DOC-{match.group(1)}") not in invalid:
            invalid.append(name)
    return Citations(tuple(sorted(valid)), tuple(invalid))
