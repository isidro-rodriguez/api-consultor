"""Extrae firmas y docstrings de paquetes instalados y los convierte en chunks."""

import contextlib
import importlib
import importlib.metadata
import inspect
import logging
import pkgutil
from collections.abc import Iterator
from types import ModuleType

from .store import Chunk
from .versions import UNKNOWN, installed_version

log = logging.getLogger(__name__)

MAX_CHARS = 2000  # ~500 tokens por chunk: 5 chunks caben de sobra en 8k de contexto


def _version(package: str) -> str:
    return installed_version(package) or UNKNOWN


def _header(qualname: str, obj: object) -> str:
    if inspect.ismodule(obj):
        return f"module {qualname}"
    kind = "class" if inspect.isclass(obj) else "def"
    signature = "(...)"
    if callable(obj):
        with contextlib.suppress(TypeError, ValueError):  # builtins sin firma
            signature = str(inspect.signature(obj))
    return f"{kind} {qualname}{signature}"


def _walk_modules(package: str) -> Iterator[ModuleType]:
    root = importlib.import_module(package)
    yield root
    path = getattr(root, "__path__", None)  # los módulos de un solo fichero no tienen
    if path is None:
        return
    for info in pkgutil.walk_packages(
        path,
        prefix=f"{package}.",
        onerror=lambda name: log.warning("No se pudo importar %s", name),
    ):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            yield importlib.import_module(info.name)
        except Exception:  # código de terceros: cualquier fallo vale
            log.warning("No se pudo importar %s", info.name)


def _public_members(module: ModuleType, package: str) -> Iterator[tuple[str, object]]:
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        origin = getattr(obj, "__module__", None) or ""
        if origin.split(".")[0] == package:  # descarta lo importado de otros paquetes
            yield name, obj


def _public_methods(cls: type) -> Iterator[tuple[str, object]]:
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(member, classmethod | staticmethod):
            member = member.__func__
        func = member
        if inspect.isfunction(func):
            yield name, func


def extract_chunks(package: str) -> list[Chunk]:
    """Devuelve un chunk por módulo, función, clase y método público con docstring.

    Lanza ModuleNotFoundError si el paquete no está instalado en el entorno.
    """
    version = _version(package)
    seen: set[int] = set()
    chunks: list[Chunk] = []

    def add(qualname: str, obj: object) -> None:
        doc = inspect.getdoc(obj)
        if doc is None or not doc.strip() or id(obj) in seen:
            return
        seen.add(id(obj))
        content = f"{_header(qualname, obj)}\n\n{doc}"[:MAX_CHARS]
        chunks.append(Chunk(qualname, package, version, content))

    for module in _walk_modules(package):
        add(module.__name__, module)
        for name, obj in _public_members(module, package):
            qualname = f"{module.__name__}.{name}"
            add(qualname, obj)  # raíz primero: gana el alias público
            if inspect.isclass(obj):
                for method_name, method in _public_methods(obj):
                    add(f"{qualname}.{method_name}", method)
    return chunks
