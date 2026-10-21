"""Versión instalada de las fuentes y comparación con la versión indexada."""

import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata

from .store import SourceInfo

UNKNOWN = "desconocida"
"""Versión que se registra cuando no se puede averiguar la del paquete."""


def installed_version(
    import_name: str, distributions: Mapping[str, list[str]] | None = None
) -> str | None:
    """Averigua la versión instalada de un paquete importable.

    El nombre importable puede diferir del de la distribución (`yaml` → PyYAML), por
    eso se resuelve con `packages_distributions()`. Los módulos de la biblioteca
    estándar tienen la versión de Python.

    Args:
        import_name: Nombre con el que se importa, p. ej. `yaml`.
        distributions: Correspondencia módulo → distribuciones, para no recalcularla
            en cada llamada; por defecto se obtiene del entorno.

    Returns:
        La versión, o `None` si el paquete no está instalado.
    """
    if import_name in sys.stdlib_module_names:
        return platform.python_version()
    mapping = (
        metadata.packages_distributions() if distributions is None else distributions
    )
    for distribution in (*mapping.get(import_name, []), import_name):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class VersionStatus:
    """Estado de una fuente indexada respecto a lo instalado.

    Attributes:
        source: La fuente indexada.
        installed: Versión instalada ahora, si procede.
        state: `"ok"`, `"stale"` (versión distinta), `"missing"` (ya no está
            instalada), `"unknown"` (se indexó sin versión) o `"docs"`
            (documentación: nada que comparar).
    """

    source: SourceInfo
    installed: str | None
    state: str

    @property
    def is_stale(self) -> bool:
        """Indica si la versión instalada difiere de la indexada."""
        return self.state == "stale"


def check_versions(sources: Sequence[SourceInfo]) -> list[VersionStatus]:
    """Compara cada fuente indexada con lo instalado en el entorno.

    Args:
        sources: Fuentes del índice.

    Returns:
        Un estado por fuente, en el mismo orden.
    """
    distributions = (
        metadata.packages_distributions()
        if any(source.kind == "package" for source in sources)
        else {}
    )
    statuses: list[VersionStatus] = []
    for source in sources:
        if source.kind != "package":
            statuses.append(VersionStatus(source, None, "docs"))
            continue
        installed = installed_version(source.package, distributions)
        if installed is None:
            state = "missing"
        elif source.version == UNKNOWN:
            state = "unknown"
        else:
            state = "ok" if installed == source.version else "stale"
        statuses.append(VersionStatus(source, installed, state))
    return statuses
