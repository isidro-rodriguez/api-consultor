"""Indexado y consulta de versiones en el entorno de otro intérprete de Python."""

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .records import Chunk, Extraction

PROBE = Path(__file__).resolve().with_name("probe.py")
"""Guion que se ejecuta con el intérprete ajeno."""

NOT_FOUND, IMPORT_FAILED = 4, 5
"""Códigos de salida de la sonda (ver `probe`)."""


class ProbeError(RuntimeError):
    """No se pudo ejecutar la sonda con el intérprete indicado."""


def resolve_interpreter(location: Path) -> Path:
    """Localiza el intérprete de un entorno virtual o valida un ejecutable.

    Args:
        location: Carpeta del entorno virtual (`.venv`) o ruta al ejecutable de Python.

    Returns:
        Ruta absoluta al intérprete (sin resolver enlaces simbólicos: en un venv eso
        llevaría al Python base y se perdería el entorno).

    Raises:
        FileNotFoundError: Si no hay ningún intérprete en `location`.
    """
    location = location.expanduser().absolute()
    if location.is_dir():
        for candidate in (
            location / "Scripts" / "python.exe",
            location / "bin" / "python",
        ):
            if candidate.is_file():
                return candidate
        message = f"No hay intérprete de Python en {location}."
        raise FileNotFoundError(message)
    if location.is_file():
        return location
    message = f"No existe {location}."
    raise FileNotFoundError(message)


def _run(python: str, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [python, "-I", str(PROBE), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError as exc:
        message = f"No se pudo ejecutar {python}: {exc.strerror or exc}."
        raise ProbeError(message) from exc


def _payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        lines = result.stderr.strip().splitlines() or result.stdout.strip().splitlines()
        detail = lines[-1:] or ["sin salida"]
        message = f"La sonda no respondió correctamente ({detail[0]})."
        raise ProbeError(message) from exc
    return (
        {str(key): value for key, value in data.items()}
        if isinstance(data, dict)
        else {}
    )


def extract_remote(python: str, package: str) -> Extraction:
    """Extrae los chunks de un paquete instalado en otro entorno.

    Args:
        python: Intérprete del entorno (ver `resolve_interpreter`).
        package: Nombre importable del paquete.

    Returns:
        Los chunks (rutas relativas a la raíz) y el intérprete usado.

    Raises:
        ModuleNotFoundError: Si el paquete no está instalado en ese entorno.
        ImportError: Si el paquete falla al importarse allí.
        ProbeError: Si el intérprete no se puede usar.
    """
    result = _run(python, "extract", package)
    data = _payload(result)
    if result.returncode == NOT_FOUND:
        raise ModuleNotFoundError(str(data.get("error")), name=str(data.get("name")))
    if result.returncode != 0:
        raise ImportError(str(data.get("error", "fallo desconocido")))
    raw = data.get("chunks")
    chunks = [Chunk(**item) for item in raw] if isinstance(raw, list) else []
    return Extraction(chunks, str(data.get("root", "")), python)


def remote_versions(python: str, names: Sequence[str]) -> dict[str, str | None]:
    """Averigua las versiones instaladas en otro entorno.

    Args:
        python: Intérprete del entorno.
        names: Nombres importables.

    Returns:
        Versión de cada uno (`None` si no está instalado allí).

    Raises:
        ProbeError: Si el intérprete no se puede usar.
    """
    result = _run(python, "versions", *names)
    data = _payload(result)
    if result.returncode != 0:
        raise ProbeError(str(data.get("error", "fallo desconocido")))
    return {name: None if v is None else str(v) for name, v in data.items()}


def remote_stdlib(python: str) -> list[str]:
    """Lista los módulos públicos de la biblioteca estándar de otro intérprete.

    Args:
        python: Intérprete del entorno.

    Returns:
        Nombres de módulo ordenados.

    Raises:
        ProbeError: Si el intérprete no se puede usar.
    """
    result = _run(python, "stdlib")
    data = _payload(result)
    modules = data.get("modules")
    if result.returncode != 0 or not isinstance(modules, list):
        raise ProbeError(str(data.get("error", "fallo desconocido")))
    return [str(name) for name in modules]


def is_own(python: str) -> bool:
    """Indica si `python` es el intérprete que ejecuta la aplicación."""
    return not python or Path(python) == Path(sys.executable)
