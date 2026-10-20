"""Indexado de documentación local: .txt, .md y .rst, sueltos, en carpetas o en .zip."""

import re
import zipfile
from collections import Counter
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path

from .store import Chunk

SUFFIXES = frozenset({".txt", ".md", ".rst"})
CHUNK_CHARS = 1500  # ~400 tokens: 5 fragmentos caben de sobra en 8k de contexto

# Encabezados estilo Setext (rst, Sphinx text, Markdown) y ATX (Markdown).
_SETEXT = re.compile(r'^([=\-*^~"+`#])\1{2,}\s*$')
_ATX = re.compile(r"^#{1,6}\s+(\S.*?)\s*#*\s*$")
_VERSION = re.compile(r"\d+(?:\.\d+)+")

type Files = list[tuple[str, str]]  # (ruta relativa, texto)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _read_zip(path: Path) -> Files:
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if not n.endswith("/"))
        # Quita la carpeta raíz común (p. ej. python-3.15-docs-text/).
        roots = {n.split("/", 1)[0] for n in names}
        strip = len(roots) == 1 and all("/" in n for n in names)
        return [
            (n.split("/", 1)[1] if strip else n, _decode(zf.read(n)))
            for n in names
            if Path(n).suffix.lower() in SUFFIXES
        ]


def _read_dir(path: Path) -> Files:
    return [
        (file.relative_to(path).as_posix(), _decode(file.read_bytes()))
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.suffix.lower() in SUFFIXES
    ]


def collect(docs_dir: Path) -> dict[str, Files]:
    """Agrupa la documentación por fuente: un .zip o carpeta = una fuente.

    Los ficheros sueltos de primer nivel forman la fuente "docs".
    """
    sources: dict[str, Files] = {}
    loose: Files = []
    for entry in sorted(docs_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            sources[entry.name] = _read_dir(entry)
        elif entry.suffix.lower() == ".zip":
            sources[entry.stem] = _read_zip(entry)
        elif entry.suffix.lower() in SUFFIXES:
            loose.append((entry.name, _decode(entry.read_bytes())))
    if loose:
        sources["docs"] = loose
    return sources


def _sections(text: str) -> Iterator[tuple[str, str]]:
    """Divide un texto en (título, cuerpo) usando sus encabezados."""
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


def _split(body: str, limit: int) -> Iterator[str]:
    """Agrupa párrafos hasta `limit` caracteres sin cortarlos."""
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
    source: str, files: Files, exclude: Sequence[str] = ()
) -> list[Chunk]:
    """Un chunk por trozo de sección; `exclude` son patrones fnmatch de ruta."""
    found = _VERSION.search(source)
    version = found.group() if found else "desconocida"
    chunks: list[Chunk] = []
    for relpath, text in files:
        if any(fnmatch(relpath, pattern) for pattern in exclude):
            continue
        counts: Counter[str] = Counter()
        for title, body in _sections(text):
            base = f"{relpath}#{title}" if title else relpath
            for part in _split(body, CHUNK_CHARS):
                counts[base] += (
                    1  # etiquetas únicas: títulos repetidos o secciones largas
                )
                n = counts[base]
                label = base if n == 1 else f"{base} [{n}]"
                chunks.append(Chunk(label, source, version, f"{label}\n\n{part}"))
    return chunks
