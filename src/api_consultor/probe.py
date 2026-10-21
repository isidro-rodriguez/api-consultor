"""Sonda que se ejecuta con el Python de OTRO entorno (`api index --python`).

Es un guion, no un módulo de la aplicación: solo usa la biblioteca estándar y carga
únicamente `records`, `docs`, `versions` e `indexer`, para no sustituir ni importar
ningún paquete del entorno que se está inspeccionando. Salida: JSON por stdout;
cualquier otra cosa que se escriba al importar el paquete (p. ej. el saludo de
pygame) se desvía a stderr para no corromper el JSON.
"""

import contextlib
import importlib
import json
import os
import sys
import types
from pathlib import Path

MIN_PYTHON = (3, 12)
EXIT_UNSUPPORTED, EXIT_NOT_FOUND, EXIT_IMPORT = 3, 4, 5


def _load() -> types.ModuleType:
    """Registra `api_consultor` como un paquete vacío que apunta a esta carpeta."""
    if "api_consultor" in sys.modules:  # ejecutado desde la propia aplicación
        return sys.modules["api_consultor"]
    package = types.ModuleType("api_consultor")
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["api_consultor"] = package
    return package


def _emit(data: object, code: int = 0) -> int:
    sys.stdout.write(json.dumps(data))
    sys.stdout.flush()
    return code


def _guard_stdout() -> None:
    """Desvía el descriptor 1 a stderr y deja `sys.stdout` en el stdout real.

    Así ni las extensiones en C ni los subprocesos que escriban en el descriptor 1
    al importar el paquete mezclan su salida con el JSON.
    """
    sys.stdout.flush()
    real = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(real, "w", encoding="utf-8")


def _dispatch(command: str, names: list[str]) -> tuple[object, int]:
    if command == "versions":
        versions = importlib.import_module("api_consultor.versions")
        return {n: versions.installed_version(n) for n in names}, 0
    indexer = importlib.import_module("api_consultor.indexer")
    if command == "stdlib":
        return {"modules": indexer.stdlib_modules()}, 0
    try:
        extraction = indexer.extract_package(names[0])
    except ModuleNotFoundError as exc:
        return {"error": str(exc), "name": exc.name}, EXIT_NOT_FOUND
    except (ImportError, ValueError, TypeError) as exc:
        return {"error": str(exc)}, EXIT_IMPORT
    chunks = [
        {key: getattr(chunk, key) for key in chunk.__slots__}
        for chunk in extraction.chunks
    ]
    return {"root": extraction.root, "chunks": chunks}, 0


def main(argv: list[str]) -> int:
    """Ejecuta `extract PAQUETE`, `versions PAQUETE...` o `stdlib` y escribe el JSON."""
    if sys.version_info < MIN_PYTHON:
        found = ".".join(map(str, sys.version_info[:2]))
        return _emit({"error": f"Se necesita Python 3.12 o superior (es {found})."}, 3)
    _load()
    command, *names = argv
    with contextlib.redirect_stdout(sys.stderr):
        data, code = _dispatch(command, names)
    return _emit(data, code)


if __name__ == "__main__":
    _guard_stdout()
    sys.exit(main(sys.argv[1:]))
