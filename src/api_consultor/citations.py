"""Citas `[DOC-01]`: identificadores de fragmentos y verificación de la respuesta."""

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\bDOC-(\d+)\b")

INSTRUCTION = (
    "Cita cada afirmación con el identificador del bloque que la respalda, "
    "por ejemplo [DOC-01]. No uses identificadores que no existan."
)


def doc_id(position: int) -> str:
    """Identificador del fragmento en la posición `position` (base 1): `DOC-01`."""
    return f"DOC-{position:02d}"


@dataclass(frozen=True, slots=True)
class Citations:
    """Citas de una respuesta frente a los fragmentos que se enviaron al modelo."""

    valid: tuple[int, ...]  # posiciones (base 1) citadas que existen, ordenadas
    invalid: tuple[str, ...]  # identificadores citados que no existen, por aparición

    @property
    def any(self) -> bool:
        return bool(self.valid or self.invalid)


def check_citations(answer: str, available: int) -> Citations:
    """Separa las citas de `answer` en válidas e inventadas (fuera de 1..available)."""
    valid: set[int] = set()
    invalid: list[str] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= available:
            valid.add(number)
        elif (name := f"DOC-{match.group(1)}") not in invalid:
            invalid.append(name)
    return Citations(tuple(sorted(valid)), tuple(invalid))
