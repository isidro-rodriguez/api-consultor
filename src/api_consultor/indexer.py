"""Extrae firmas y docstrings de paquetes instalados y los convierte en chunks."""

import contextlib
import importlib
import importlib.metadata
import inspect
import logging
import pkgutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

from .docs import sections, split_text
from .store import Chunk
from .versions import UNKNOWN, installed_version

log = logging.getLogger(__name__)

MAX_CHARS = 2000  # ~500 tokens por chunk: 5 chunks caben de sobra en 8k de contexto
CONTINUATION = " (continuación 99)"  # marca máxima que se añade a las cabeceras
MIN_BODY = 500  # espacio mínimo para el docstring aunque la firma sea larguísima


def _version(package: str) -> str:
    return installed_version(package) or UNKNOWN


def doc_parts(doc: str, limit: int) -> list[str]:
    """Divide un docstring en partes de hasta `limit` caracteres sin perder nada.

    Respeta las secciones (`Parameters`, `Returns`, `Examples`…): agrupa secciones
    completas mientras quepan y solo parte por párrafos las que no caben solas.
    """
    parts: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            parts.append(buffer.strip("\n"))
        buffer = ""

    for title, body in sections(doc):
        block = f"{title}\n{body}".strip("\n") if title else body.strip("\n")
        if not block.strip():
            continue
        if len(block) <= limit:
            if buffer and len(buffer) + len(block) + 2 > limit:
                flush()
            buffer = f"{buffer}\n\n{block}" if buffer else block
            continue
        flush()
        pieces = list(split_text(body, limit - len(title) - 1))
        for index, piece in enumerate(pieces):
            parts.append(f"{title}\n{piece}".strip("\n") if index == 0 else piece)
    flush()
    return parts


def _location(obj: object) -> tuple[str, int]:
    """Fichero fuente y línea de definición de `obj`; ("", 0) si no se conocen."""
    with contextlib.suppress(TypeError, OSError):
        if inspect.ismodule(obj):
            file, line = inspect.getsourcefile(obj), 1
        elif inspect.isfunction(obj) or inspect.isclass(obj):
            file, line = inspect.getsourcefile(obj), inspect.getsourcelines(obj)[1]
        else:
            return "", 0
        if file is not None and Path(file).is_file():
            return str(Path(file).resolve()), max(line, 1)
    return "", 0


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
    """Devuelve chunks de cada módulo, función, clase y método público con docstring.

    Un docstring largo se reparte en varios chunks por secciones y párrafos; ninguno
    pierde texto (`MAX_CHARS` es un límite de tamaño, no un truncado).

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
        header = _header(qualname, obj)
        path, line = _location(obj)
        reserved = len(header) + len(CONTINUATION) + 8  # cabecera, marca y saltos
        limit = max(MAX_CHARS - reserved, MIN_BODY)
        for number, part in enumerate(doc_parts(doc, limit), start=1):
            if number == 1:
                label, text = qualname, f"{header}\n\n{part}"
            else:  # continuación: misma cabecera y etiqueta única
                label = f"{qualname} [{number}]"
                text = f"{header} (continuación {number})\n\n{part}"
            chunks.append(Chunk(label, package, version, text, path, line))

    for module in _walk_modules(package):
        add(module.__name__, module)
        for name, obj in _public_members(module, package):
            qualname = f"{module.__name__}.{name}"
            add(qualname, obj)  # raíz primero: gana el alias público
            if inspect.isclass(obj):
                for method_name, method in _public_methods(obj):
                    add(f"{qualname}.{method_name}", method)
    return chunks
