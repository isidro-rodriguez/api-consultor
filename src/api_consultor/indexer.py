"""Extrae firmas y docstrings de paquetes instalados y los convierte en chunks."""

import contextlib
import importlib
import importlib.metadata
import inspect
import logging
import pkgutil
import sys
from collections.abc import Collection, Iterator
from functools import cached_property
from pathlib import Path
from types import ModuleType

from .docs import sections, split_text
from .records import Chunk, Extraction, relativize
from .versions import UNKNOWN, installed_version

log = logging.getLogger(__name__)

MAX_CHARS = 2000
"""Tamaño máximo de un fragmento (~500 tokens: caben 5 en un contexto de 8k)."""

CONTINUATION = " (continuación 99)"
"""Marca más larga que se añade a la cabecera de un fragmento de continuación."""

MIN_BODY = 500
"""Espacio mínimo para el docstring aunque la firma sea larguísima."""


def _version(package: str) -> str:
    return installed_version(package) or UNKNOWN


def doc_parts(doc: str, limit: int) -> list[str]:
    """Divide un docstring en partes de hasta `limit` caracteres sin perder nada.

    Respeta las secciones (`Parameters`, `Returns`, `Examples`…): agrupa secciones
    completas mientras quepan y solo parte por párrafos las que no caben solas.

    Args:
        doc: Docstring ya limpio (`inspect.getdoc`).
        limit: Tamaño máximo de cada parte.

    Returns:
        Las partes, en orden; su unión contiene todo el texto original.
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
        elif isinstance(obj, property | cached_property):
            getter = obj.fget if isinstance(obj, property) else obj.func
            return _location(getter) if getter is not None else ("", 0)
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
    if isinstance(obj, property | cached_property):
        return f"property {qualname}"
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


def _public_methods(
    cls: type, package: str, public: Collection[int]
) -> Iterator[tuple[str, object]]:
    """Métodos y propiedades públicos de `cls`.

    Incluye los definidos en `cls` y los heredados de clases base *privadas* del mismo
    paquete (las que el usuario no ve, como `pathlib._abc.PurePathBase`); los de las
    bases públicas se indexan en ellas. Gana la definición más cercana en el MRO.

    Args:
        cls: Clase pública.
        package: Paquete que se indexa; las bases de otros paquetes se ignoran.
        public: `id` de todas las clases públicas del paquete.

    Yields:
        Pares (nombre, objeto).
    """
    names: set[str] = set()
    for klass in cls.__mro__:
        if klass is not cls and (
            id(klass) in public
            or (getattr(klass, "__module__", "") or "").split(".")[0] != package
        ):
            continue
        for name, member in vars(klass).items():
            if name.startswith("_") or name in names:
                continue
            names.add(name)
            if isinstance(member, classmethod | staticmethod):
                member = member.__func__
            if inspect.isfunction(member) or isinstance(
                member, property | cached_property
            ):
                yield name, member


def extract_chunks(package: str) -> list[Chunk]:
    """Devuelve chunks de cada módulo, función, clase, método y propiedad públicos.

    Solo cuentan los que tienen docstring. Los métodos heredados se indexan en la clase
    que los define; si esa clase es privada (p. ej. `pathlib._abc.PurePathBase`), en la
    primera clase pública que los hereda.

    Un docstring largo se reparte en varios chunks por secciones y párrafos; ninguno
    pierde texto (`MAX_CHARS` es un límite de tamaño, no un truncado).

    Cada chunk recuerda el fichero y la línea de su símbolo (vacíos si no hay fuente,
    p. ej. en extensiones en C).

    Args:
        package: Nombre importable del paquete, p. ej. `rich`.

    Returns:
        Los chunks de todos los módulos públicos del paquete.

    Raises:
        ModuleNotFoundError: Si el paquete no está instalado en el entorno.
        ImportError: Si el paquete (no sus submódulos) falla al importarse.
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

    loaded = [
        (module, list(_public_members(module, package)))
        for module in _walk_modules(package)
    ]
    public = {
        id(obj)
        for _module, found in loaded
        for _name, obj in found
        if inspect.isclass(obj)
    }
    for module, found in loaded:
        add(module.__name__, module)
        for name, obj in found:
            qualname = f"{module.__name__}.{name}"
            add(qualname, obj)  # raíz primero: gana el alias público
            if inspect.isclass(obj):
                for method_name, method in _public_methods(obj, package, public):
                    add(f"{qualname}.{method_name}", method)
    return chunks


def _package_root(package: str) -> Path | None:
    """Carpeta que contiene el paquete (o el módulo) importado, si es de ficheros."""
    module = sys.modules.get(package)
    file = getattr(module, "__file__", None)
    if file is None:  # espacios de nombres, builtins o módulos congelados
        return None
    path = Path(file).resolve()
    return path.parent.parent if path.stem == "__init__" else path.parent


def extract_package(package: str) -> Extraction:
    """Extrae los chunks de un paquete con rutas relativas a su carpeta contenedora.

    Args:
        package: Nombre importable del paquete.

    Returns:
        Los chunks y la raíz (`site-packages`, `Lib`...) de la que cuelgan sus ficheros.

    Raises:
        ModuleNotFoundError: Si el paquete no está instalado.
        ImportError: Si el paquete (no sus submódulos) falla al importarse.
    """
    chunks = extract_chunks(package)
    root = _package_root(package)
    if root is None:
        return Extraction(chunks)
    return Extraction(relativize(chunks, root), str(root))
