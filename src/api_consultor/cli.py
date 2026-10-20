"""Interfaz de línea de comandos (Typer)."""

import logging
import platform
import sqlite3
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, NoReturn

import ollama
import sqlite_vec
import typer
from rich.logging import RichHandler
from rich.progress import Progress

from . import __version__
from .config import SETTINGS
from .consultor import Consultor
from .docs import collect, extract_doc_chunks
from .indexer import extract_chunks
from .llm import embed
from .render import (
    err_console,
    print_error,
    print_hint,
    print_warning,
    progress_bar,
    prompt,
    show_sources,
    show_sources_table,
    show_version,
    status,
    stream_markdown,
    style_typer_help,
)
from .store import Chunk, SourceInfo, Store

BATCH_SIZE = 32

style_typer_help()
app = typer.Typer(
    name="api",
    help="Consultor de APIs Python offline con Ollama y RAG.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _version_details() -> list[tuple[str, str]]:
    return [
        ("Python", platform.python_version()),
        ("SQLite", f"{sqlite3.sqlite_version} · sqlite-vec {sqlite_vec.__version__}"),
        ("Modelo", SETTINGS.chat_model),
        ("Embeddings", SETTINGS.embed_model),
    ]


def _version_callback(value: bool) -> None:
    if value:
        show_version(__version__, _version_details())
        raise typer.Exit


@app.callback()
def _setup(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Muestra la versión y termina.",
        ),
    ] = False,
) -> None:
    # Avisos de logging (p. ej. módulos que no se pueden importar) con estilo Rich.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_path=False, show_time=False)],
    )


def _fail(message: str, hint: str | None = None) -> NoReturn:
    print_error(message, hint)
    raise typer.Exit(code=1)


@contextmanager
def _ollama_errors() -> Generator[None]:
    """Convierte los fallos típicos de Ollama en mensajes legibles."""
    try:
        yield
    except ConnectionError:
        _fail(
            "No se pudo conectar con Ollama.",
            "¿Está en ejecución? Prueba `ollama serve`.",
        )
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            _fail(
                f"Falta un modelo en Ollama: {exc.error}",
                f"ollama pull {SETTINGS.embed_model}  ·  "
                f"ollama create {SETTINGS.chat_model} -f Modelfile",
            )
        _fail(f"Error de Ollama: {exc.error}")


def _index_source(
    store: Store,
    progress: Progress,
    name: str,
    build: Callable[[], list[Chunk]],
) -> SourceInfo | None:
    """Extrae los chunks de una fuente, calcula sus embeddings y los guarda."""
    task = progress.add_task(name, total=None)  # barra "pulsante" mientras se extrae
    try:
        chunks = build()
    except Exception:
        progress.remove_task(task)
        raise
    if not chunks:
        progress.remove_task(task)
        print_warning(f"{name}: nada indexable.")
        return None
    progress.update(task, total=len(chunks), completed=0)
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings.extend(embed([chunk.content for chunk in batch]))
        progress.update(task, completed=len(embeddings))
    store.replace(name, chunks, embeddings)
    return SourceInfo(name, chunks[0].version, len(chunks))


@app.command()
def index(
    packages: Annotated[
        list[str],
        typer.Argument(
            metavar="PAQUETE...",
            help="Paquetes instalados en el entorno.",
            show_default=False,
        ),
    ],
) -> None:
    """Indexa paquetes Python instalados (firmas y docstrings)."""
    results: list[SourceInfo] = []
    with (
        Store(SETTINGS.db_path, SETTINGS.embed_dim) as store,
        progress_bar() as progress,
        _ollama_errors(),
    ):
        for package in packages:
            try:
                info = _index_source(
                    store, progress, package, partial(extract_chunks, package)
                )
            except ModuleNotFoundError as exc:
                print_warning(
                    f"{package}: módulo no encontrado ({exc.name}).",
                    f"¿Está instalado en el entorno? Prueba `uv add --dev {package}`.",
                )
                continue
            if info:
                results.append(info)
    show_sources_table(results, "Indexado")


@app.command("index-docs")
def index_docs(
    directory: Annotated[
        Path,
        typer.Argument(
            metavar="DIRECTORIO",
            exists=True,
            file_okay=False,
            help="Carpeta con .txt, .md, .rst o .zip.",
        ),
    ] = SETTINGS.docs_dir,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            metavar="PATRÓN",
            help="Patrón fnmatch de ruta a omitir (repetible), p. ej. 'whatsnew/*'.",
        ),
    ] = None,
) -> None:
    """Indexa documentación local (.txt, .md, .rst; sueltos, en carpetas o en .zip)."""
    results: list[SourceInfo] = []
    with (
        Store(SETTINGS.db_path, SETTINGS.embed_dim) as store,
        progress_bar() as progress,
        _ollama_errors(),
    ):
        for source, files in collect(directory).items():
            build = partial(extract_doc_chunks, source, files, exclude or ())
            if info := _index_source(store, progress, source, build):
                results.append(info)
    show_sources_table(results, "Indexado")


def _read_questions() -> Iterator[str]:
    print_hint(
        'Modo interactivo: escribe tu pregunta ("salir" o Ctrl+D para terminar).'
    )
    while (line := prompt()) is not None:
        if line in {"salir", "exit"}:
            return
        if line:
            yield line


def _answer(consultor: Consultor, question: str) -> None:
    with status("Buscando en la documentación..."):
        chunks = consultor.retrieve(question)
    stream_markdown(consultor.stream(question, chunks))
    show_sources(chunks)


@app.command()
def ask(
    question: Annotated[
        str | None,
        typer.Argument(
            help="Pregunta; sin ella, modo interactivo.", show_default=False
        ),
    ] = None,
) -> None:
    """Consulta lo indexado y responde con el modelo local."""
    with Store(SETTINGS.db_path, SETTINGS.embed_dim) as store:
        if not store.sources():
            _fail(
                "No hay nada indexado todavía.",
                "Usa `api index <paquete>` o `api index-docs`.",
            )
        consultor = Consultor(store)
        interactive = question is None or not question.strip()
        questions: Sequence[str] | Iterator[str] = (
            _read_questions() if interactive else [question]
        )
        with _ollama_errors():
            for text in questions:
                _answer(consultor, text)


@app.command()
def sources() -> None:
    """Lista las fuentes indexadas."""
    with Store(SETTINGS.db_path, SETTINGS.embed_dim) as store:
        rows = store.sources()
    if rows:
        show_sources_table(rows, "Fuentes indexadas")
    else:
        print_hint("Nada indexado todavía.")
