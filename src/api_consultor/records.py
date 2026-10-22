"""Registros de datos compartidos; solo biblioteca estándar (los usa `probe`)."""

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Chunk:
    """Fragmento indexado: un símbolo de una API o una sección de documentación.

    Attributes:
        qualname: Nombre único dentro de la fuente (`pkg.mod.func`, `ruta.md#Título`).
        package: Fuente a la que pertenece (paquete o carpeta/zip de documentación).
        version: Versión de la fuente cuando se indexó.
        content: Texto que se vectoriza y se envía al modelo.
        path: Fichero de origen enlazable; vacío si no existe (builtins). En el índice
            es relativo a la raíz de la fuente (`SourceInfo.root`).
        line: Línea del símbolo o sección en `path`; 0 si se desconoce.
    """

    qualname: str
    package: str
    version: str
    content: str
    path: str = ""
    line: int = 0


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Resumen de una fuente indexada.

    Attributes:
        package: Nombre de la fuente.
        version: Versión registrada al indexar.
        chunks: Número de fragmentos.
        kind: `"package"` (API de un paquete), `"stdlib"` (módulo de la biblioteca
            estándar) o `"docs"` (documentación).
        indexed_at: Fecha ISO de indexado (no cuenta en la comparación).
        root: Carpeta que contiene los ficheros de la fuente; vacía si se desconoce.
        python: Intérprete del entorno donde se indexó el paquete; vacío si fue el
            de la propia herramienta.
        module: Nombre importable del paquete; vacío si coincide con `package`. Se
            distinguen cuando el mismo paquete está indexado desde dos entornos
            (`zensical` y `zensical@reCocina`).
    """

    package: str
    version: str
    chunks: int
    kind: str = "package"
    indexed_at: str = field(default="", compare=False)
    root: str = field(default="", compare=False)
    python: str = field(default="", compare=False)
    module: str = field(default="", compare=False)

    @property
    def import_name(self) -> str:
        """Nombre con el que se importa el paquete (para versiones y reindexado)."""
        return self.module or self.package


@dataclass(frozen=True, slots=True)
class Extraction:
    """Resultado de extraer una fuente: sus chunks y dónde están sus ficheros.

    Attributes:
        chunks: Fragmentos, con `path` relativo a `root` cuando es posible.
        root: Carpeta raíz de los ficheros de la fuente.
        python: Intérprete usado (vacío si fue el de la herramienta).
    """

    chunks: list[Chunk]
    root: str = ""
    python: str = ""


def relativize(chunks: list[Chunk], root: Path) -> list[Chunk]:
    """Hace relativas a `root` las rutas de los chunks que cuelgan de ella.

    Args:
        chunks: Fragmentos con rutas absolutas.
        root: Carpeta raíz de la fuente.

    Returns:
        Fragmentos nuevos; los que no están bajo `root` conservan su ruta.
    """
    result: list[Chunk] = []
    for chunk in chunks:
        path = Path(chunk.path) if chunk.path else None
        if path is not None and path.is_absolute() and path.is_relative_to(root):
            chunk = replace(chunk, path=path.relative_to(root).as_posix())
        result.append(chunk)
    return result
