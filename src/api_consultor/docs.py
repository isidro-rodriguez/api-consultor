"""Indexado de documentación local: .txt, .md y .rst, sueltos, en carpetas o en .zip."""

import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path

from .records import Chunk

SUFFIXES = frozenset({".txt", ".md", ".rst"})
"""Extensiones de los ficheros de documentación que se indexan."""

CHUNK_CHARS = 1500
"""Tamaño máximo de un fragmento de documentación (~400 tokens)."""

# Encabezados estilo Setext (rst, Sphinx text, Markdown) y ATX (Markdown).
_SETEXT = re.compile(r'^([=\-*^~"+`#])\1{2,}\s*$')
_ATX = re.compile(r"^#{1,6}\s+(\S.*?)\s*#*\s*$")
_VERSION = re.compile(r"\d+(?:\.\d+)+")

MAX_FILE_BYTES = 20 * 2**20
"""Tamaño máximo de un fichero de documentación (descomprimido)."""

MAX_SOURCE_BYTES = 512 * 2**20
"""Tamaño máximo de una fuente de documentación (descomprimida)."""

MAX_FILES = 20_000
"""Número máximo de ficheros de texto en una fuente."""

type Files = list[tuple[str, str]]
"""Ficheros de una fuente: pares (ruta relativa, texto)."""


class DocsError(Exception):
    """La documentación no se puede indexar tal como está.

    Args:
        message: Qué ocurre.
        hint: Cómo solucionarlo, si se sabe.

    Attributes:
        hint: Sugerencia para el usuario, o `None`.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def _check_limits(source: Path, sizes: Sequence[tuple[str, int]]) -> None:
    """Rechaza fuentes desmesuradas (p. ej. una bomba zip) antes de leerlas."""
    hint = "Revisa que sea la documentación que querías o divídela en varias fuentes."
    if len(sizes) > MAX_FILES:
        message = (
            f"{source.name} tiene {len(sizes)} ficheros de texto (máximo {MAX_FILES})."
        )
        raise DocsError(message, hint)
    for name, size in sizes:
        if size > MAX_FILE_BYTES:
            message = (
                f"{source.name}: {name} ocupa {size / 2**20:.0f} MiB "
                f"(máximo {MAX_FILE_BYTES // 2**20} MiB por fichero)."
            )
            raise DocsError(message, hint)
    if (total := sum(size for _name, size in sizes)) > MAX_SOURCE_BYTES:
        message = (
            f"{source.name} ocupa {total / 2**20:.0f} MiB de texto "
            f"(máximo {MAX_SOURCE_BYTES // 2**20} MiB por fuente)."
        )
        raise DocsError(message, hint)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _zip_members(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(ruta relativa, nombre en el zip) de los textos, sin la carpeta raíz común."""
    names = sorted(n for n in zf.namelist() if not n.endswith("/"))
    # Quita la carpeta raíz común (p. ej. python-3.15-docs-text/).
    roots = {n.split("/", 1)[0] for n in names}
    strip = len(roots) == 1 and all("/" in n for n in names)
    return [
        (n.split("/", 1)[1] if strip else n, n)
        for n in names
        if Path(n).suffix.lower() in SUFFIXES
    ]


def _read_zip(path: Path) -> Files:
    try:
        with zipfile.ZipFile(path) as zf:
            members = _zip_members(zf)
            # `file_size` es fiable: `zipfile` no lee más allá de lo declarado.
            _check_limits(
                path, [(rel, zf.getinfo(name).file_size) for rel, name in members]
            )
            return [(rel, _decode(zf.read(name))) for rel, name in members]
    except zipfile.BadZipFile as exc:
        message = f"{path.name} no es un .zip válido ({exc})."
        raise DocsError(
            message, "Vuelve a descargarlo o quítalo de la carpeta."
        ) from exc


def _extract_zip(path: Path, target: Path, exclude: Sequence[str]) -> Path:
    """Extrae los textos del zip a `target` (sustituyendo lo anterior) y lo devuelve."""
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    root = target.resolve()
    with zipfile.ZipFile(path) as zf:
        for rel, name in _zip_members(zf):
            if any(fnmatch(rel, pattern) for pattern in exclude):
                continue
            dest = (root / rel).resolve()
            if not dest.is_relative_to(root):  # nombres con `..` o rutas absolutas
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
    return root


def _read_dir(path: Path) -> Files:
    files = [
        file
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.suffix.lower() in SUFFIXES
    ]
    _check_limits(path, [(file.name, file.stat().st_size) for file in files])
    return [
        (file.relative_to(path).as_posix(), _decode(file.read_bytes()))
        for file in files
    ]


def _add(sources: dict[str, Files], name: str, files: Files, entry: str) -> None:
    if name in sources:
        message = f"Dos entradas dan la fuente «{name}»: {entry} y otra carpeta o .zip."
        hint = "Renombra una: el nombre de la carpeta o del .zip es el de la fuente."
        raise DocsError(message, hint)
    sources[name] = files


def collect(docs_dir: Path) -> dict[str, Files]:
    """Agrupa la documentación por fuente: un .zip o carpeta = una fuente.

    Los ficheros sueltos de primer nivel forman la fuente "docs".

    Args:
        docs_dir: Carpeta con la documentación; se ignoran las entradas ocultas.

    Returns:
        Para cada fuente, sus ficheros de texto con ruta relativa.

    Raises:
        DocsError: Si dos entradas dan el mismo nombre de fuente (`foo/` y `foo.zip`),
            un `.zip` está dañado o una fuente supera los límites de tamaño.
    """
    sources: dict[str, Files] = {}
    loose: list[Path] = []
    for entry in sorted(docs_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            _add(sources, entry.name, _read_dir(entry), f"{entry.name}/")
        elif entry.suffix.lower() == ".zip":
            _add(sources, entry.stem, _read_zip(entry), entry.name)
        elif entry.suffix.lower() in SUFFIXES:
            loose.append(entry)
    if loose:
        _check_limits(docs_dir, [(file.name, file.stat().st_size) for file in loose])
        files = [(file.name, _decode(file.read_bytes())) for file in loose]
        _add(sources, "docs", files, "los ficheros sueltos")
    return sources


def materialize(
    docs_dir: Path, source: str, cache_dir: Path, exclude: Sequence[str] = ()
) -> Path:
    """Localiza en disco los ficheros de una fuente, para poder enlazarlos.

    Una carpeta o los ficheros sueltos ya existen en disco; un `.zip` se extrae a
    `cache_dir/source` (solo los textos no excluidos), sustituyendo lo anterior.

    Args:
        docs_dir: Carpeta con la documentación.
        source: Nombre de la fuente (carpeta, `.zip` sin extensión o `docs`).
        cache_dir: Carpeta donde extraer los `.zip`.
        exclude: Patrones `fnmatch` de ruta que no se extraen.

    Returns:
        La carpeta que contiene los ficheros de la fuente.
    """
    folder = docs_dir / source
    if folder.is_dir():
        return folder.resolve()
    archive = docs_dir / f"{source}.zip"
    if archive.is_file():
        return _extract_zip(archive, cache_dir / source, exclude)
    return docs_dir.resolve()  # ficheros sueltos


def _heading_line(lines: Sequence[str], title: str, start: int) -> int:
    """Índice (base 0) de la primera línea desde `start` que es el título; -1 si no."""
    for index in range(start, len(lines)):
        if lines[index].strip().strip("#").strip() == title:
            return index
    return -1


def sections(text: str) -> Iterator[tuple[str, str]]:
    """Divide un texto en secciones usando sus encabezados (Setext o ATX).

    Ignora los encabezados dentro de bloques de código.

    Args:
        text: Texto en `.md`, `.rst` o `.txt`.

    Yields:
        Pares (título, cuerpo); la primera sección puede tener título vacío.
    """
    lines = text.splitlines()
    title: str = ""
    body: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        heading: str | None = None
        step = 1
        if not in_fence:
            underline = lines[i + 1] if i + 1 < len(lines) else ""
            if (
                line.strip()
                and not _SETEXT.match(line)
                and _SETEXT.match(underline)
                and len(underline.strip()) >= len(line.strip())
            ):
                heading, step = line.strip(), 2
            elif match := _ATX.match(line):
                heading = str(match.group(1))
        if heading is None:
            body.append(line)
        else:
            yield title, "\n".join(body)
            title = heading
            body.clear()
        i += step
    yield title, "\n".join(body)


def _wrap(paragraph: str, limit: int) -> Iterator[str]:
    """Parte un párrafo gigante (tablas, listados) por saltos de línea."""
    while len(paragraph) > limit:
        cut = paragraph.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        yield paragraph[:cut]
        paragraph = paragraph[cut:].lstrip("\n")
    yield paragraph


def split_text(body: str, limit: int) -> Iterator[str]:
    """Agrupa párrafos hasta un tamaño máximo sin cortarlos.

    Un párrafo mayor que `limit` se parte por saltos de línea.

    Args:
        body: Texto de una sección.
        limit: Tamaño máximo de cada trozo.

    Yields:
        Trozos de texto en orden.
    """
    buffer: list[str] = []
    size = 0
    for paragraph in re.split(r"\n\s*\n", body):
        if not paragraph.strip():
            continue
        for piece in _wrap(paragraph.strip("\n"), limit):
            if buffer and size + len(piece) > limit:
                yield "\n\n".join(buffer)
                buffer.clear()
                size = 0
            buffer.append(piece)
            size += len(piece) + 2
    if buffer:
        yield "\n\n".join(buffer)


def extract_doc_chunks(
    source: str,
    files: Files,
    exclude: Sequence[str] = (),
    root: Path | None = None,
) -> list[Chunk]:
    """Convierte los ficheros de una fuente en chunks, uno por trozo de sección.

    Con `root` (carpeta de la fuente) los chunks recuerdan fichero y línea para poder
    abrirlos; sin ella no hay ubicación.

    Args:
        source: Nombre de la fuente; si contiene una versión (`py-3.15`) se registra.
        files: Ficheros de la fuente (ruta relativa y texto).
        exclude: Patrones `fnmatch` de ruta a omitir.
        root: Carpeta donde están los ficheros en disco.

    Returns:
        Los chunks, etiquetados `ruta#título` (con ` [n]` si hay varios).
    """
    found = _VERSION.search(source)
    version = found.group() if found else "desconocida"
    chunks: list[Chunk] = []
    for relpath, text in files:
        if any(fnmatch(relpath, pattern) for pattern in exclude):
            continue
        counts: Counter[str] = Counter()
        path = str(root / relpath) if root is not None else ""
        lines = text.splitlines()
        cursor = 0
        for title, body in sections(text):
            line = 1 if root is not None else 0
            if root is not None and title:
                found = _heading_line(lines, title, cursor)
                if found >= 0:
                    cursor, line = found + 1, found + 1
            base = f"{relpath}#{title}" if title else relpath
            for part in split_text(body, CHUNK_CHARS):
                counts[base] += (
                    1  # etiquetas únicas: títulos repetidos o secciones largas
                )
                n = counts[base]
                label = base if n == 1 else f"{base} [{n}]"
                chunks.append(
                    Chunk(label, source, version, f"{label}\n\n{part}", path, line)
                )
    return chunks
