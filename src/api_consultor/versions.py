"""Versión instalada de las fuentes y comparación con la versión indexada."""

import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from .external import ProbeError, is_own, remote_versions
from .records import SourceInfo

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
            (documentación: nada que comparar), `"gone"` (ya no existe el entorno
            o la carpeta donde estaba).
    """

    source: SourceInfo
    installed: str | None
    state: str

    @property
    def is_stale(self) -> bool:
        """Indica si la versión instalada difiere de la indexada."""
        return self.state == "stale"


def _state(source: SourceInfo, installed: str | None) -> str:
    if installed is None:
        return "missing"
    if source.version == UNKNOWN:
        return "unknown"
    return "ok" if installed == source.version else "stale"


def _foreign_versions(
    sources: Sequence[SourceInfo],
) -> dict[str, dict[str, str | None]]:
    """Versiones instaladas en los entornos ajenos; vacío si el entorno ya no está."""
    wanted: dict[str, list[str]] = {}
    for source in sources:
        if source.kind != "docs" and not is_own(source.python):
            wanted.setdefault(source.python, []).append(source.package)
    found: dict[str, dict[str, str | None]] = {}
    for python, names in wanted.items():
        if not Path(python).is_file():
            continue
        try:
            found[python] = remote_versions(python, names)
        except ProbeError:
            continue
    return found


def check_versions(sources: Sequence[SourceInfo]) -> list[VersionStatus]:
    """Compara cada fuente indexada con lo instalado en su entorno.

    Los paquetes indexados con `--python` se comparan con ese entorno, no con el de
    la herramienta. Una fuente cuyo entorno o carpeta ya no existe queda `"gone"`.

    Args:
        sources: Fuentes del índice.

    Returns:
        Un estado por fuente, en el mismo orden.
    """
    distributions = (
        metadata.packages_distributions()
        if any(s.kind != "docs" and is_own(s.python) for s in sources)
        else {}
    )
    foreign = _foreign_versions(sources)
    statuses: list[VersionStatus] = []
    for source in sources:
        if source.kind == "docs":
            gone = bool(source.root) and not Path(source.root).is_dir()
            statuses.append(VersionStatus(source, None, "gone" if gone else "docs"))
        elif is_own(source.python):
            installed = installed_version(source.package, distributions)
            statuses.append(VersionStatus(source, installed, _state(source, installed)))
        elif source.python not in foreign:
            statuses.append(VersionStatus(source, None, "gone"))
        else:
            installed = foreign[source.python].get(source.package)
            statuses.append(VersionStatus(source, installed, _state(source, installed)))
    return statuses


_SEVERITY = ("gone", "stale", "missing", "unknown", "ok")
"""Estados de peor a mejor: el de un grupo es el de su peor miembro."""


def collapse_stdlib(statuses: Sequence[VersionStatus]) -> list[VersionStatus]:
    """Resume en una fila por intérprete las fuentes de la biblioteca estándar.

    Args:
        statuses: Estados de todas las fuentes.

    Returns:
        Los estados no estándar tal cual y, en el lugar del primer módulo estándar de
        cada intérprete, una fila de resumen con la suma de fragmentos.
    """
    groups: dict[str, list[VersionStatus]] = {}
    for status in statuses:
        if status.source.kind == "stdlib":
            groups.setdefault(status.source.python, []).append(status)
    result: list[VersionStatus] = []
    for status in statuses:
        if status.source.kind != "stdlib":
            result.append(status)
            continue
        members = groups.pop(status.source.python, None)
        if members is None:
            continue  # ya resumido
        source = status.source
        summary = SourceInfo(
            f"stdlib ({len(members)} módulos)",
            source.version,
            sum(member.source.chunks for member in members),
            "stdlib",
            max(member.source.indexed_at for member in members),
            source.root,
            source.python,
        )
        state = min((m.state for m in members), key=_SEVERITY.index)
        result.append(VersionStatus(summary, status.installed, state))
    return result
