"""Versión instalada de las fuentes y comparación con la versión indexada."""

import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata

from .store import SourceInfo

UNKNOWN = "desconocida"


def installed_version(
    import_name: str, distributions: Mapping[str, list[str]] | None = None
) -> str | None:
    """Versión instalada del paquete importable `import_name`, o None si no está.

    El nombre importable puede diferir del de la distribución (`yaml` → PyYAML), por
    eso se resuelve con `packages_distributions()`. Los módulos de la biblioteca
    estándar tienen la versión de Python.
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

    `state`: "ok", "stale" (versión distinta), "missing" (ya no está instalada),
    "unknown" (se indexó sin versión) o "docs" (documentación: nada que comparar).
    """

    source: SourceInfo
    installed: str | None
    state: str

    @property
    def is_stale(self) -> bool:
        return self.state == "stale"


def check_versions(sources: Sequence[SourceInfo]) -> list[VersionStatus]:
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
