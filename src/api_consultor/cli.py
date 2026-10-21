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
from .citations import check_citations
from .config import SETTINGS
from .consultor import Consultor, format_context
from .docs import collect, extract_doc_chunks
from .evaluation import DEFAULT_KS, evaluate, load_cases, missing_sources
from .indexer import extract_chunks
from .llm import embed
from .render import (
    err_console,
    print_error,
    print_hint,
    print_warning,
    progress_bar,
    prompt,
    show_debug,
    show_evaluation,
    show_sources,
    show_sources_table,
    show_status_table,
    show_version,
    status,
    stream_markdown,
    style_typer_help,
)
from .store import Chunk, IndexIntegrityError, SourceInfo, Store
from .versions import check_versions

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
def _store(*, check_model: bool = True) -> Generator[Store]:
    """Abre el índice; si no es compatible con la configuración, falla con claridad."""
    model = SETTINGS.embed_model if check_model else None
    try:
        store = Store(SETTINGS.db_path, SETTINGS.embed_dim, model)
    except IndexIntegrityError as exc:
        _fail(str(exc), exc.hint)
    with store:
        yield store


@contextmanager
def _handled_errors() -> Generator[None]:
    """Convierte los fallos típicos de Ollama y del índice en mensajes legibles."""
    try:
        yield
    except IndexIntegrityError as exc:
        _fail(str(exc), exc.hint)
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
    kind: str = "package",
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
        if start == 0:
            store.check_vector(embeddings[0])  # falla pronto si el modelo no encaja
        progress.update(task, completed=len(embeddings))
    store.replace(name, chunks, embeddings, kind)
    return SourceInfo(name, chunks[0].version, len(chunks), kind)


def _index_packages(store: Store, packages: Sequence[str]) -> list[SourceInfo]:
    results: list[SourceInfo] = []
    with progress_bar() as progress, _handled_errors():
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
    return results


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
    with _store() as store:
        results = _index_packages(store, packages)
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
    with _store() as store, progress_bar() as progress, _handled_errors():
        for source, files in collect(directory).items():
            build = partial(extract_doc_chunks, source, files, exclude or ())
            if info := _index_source(store, progress, source, build, "docs"):
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


def _warn_stale(store: Store) -> frozenset[str]:
    """Avisa de las fuentes cuya versión instalada difiere de la indexada."""
    stale = [s for s in check_versions(store.sources()) if s.is_stale]
    if stale:
        details = ", ".join(
            f"{s.source.package} (indexada {s.source.version}, instalada {s.installed})"
            for s in stale
        )
        print_warning(
            f"Documentación posiblemente obsoleta: {details}.",
            "Reindexa con `api update`.",
        )
    return frozenset(s.source.package for s in stale)


def _answer(
    consultor: Consultor, question: str, stale: frozenset[str], *, debug: bool = False
) -> None:
    with status("Buscando en la documentación..."):
        retrieval = consultor.search(question)
    if debug:
        context = len(format_context(retrieval.chunks))
        show_debug(retrieval, context)
    if retrieval.abstain_reason is not None:
        print_warning(
            "No encuentro documentación suficiente para responder con seguridad.",
            f"Motivo: {retrieval.abstain_reason}. Reformula o indexa más fuentes.",
        )
        return
    answer = stream_markdown(consultor.stream(question, retrieval.chunks))
    citations = check_citations(answer, len(retrieval.chunks))
    show_sources(retrieval.chunks, stale, citations)


@app.command()
def ask(
    question: Annotated[
        str | None,
        typer.Argument(
            help="Pregunta; sin ella, modo interactivo.", show_default=False
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug", "-d", help="Muestra candidatos, señales y tamaño del contexto."
        ),
    ] = False,
) -> None:
    """Consulta lo indexado y responde con el modelo local."""
    with _store() as store:
        if not store.sources():
            _fail(
                "No hay nada indexado todavía.",
                "Usa `api index <paquete>` o `api index-docs`.",
            )
        stale = _warn_stale(store)
        consultor = Consultor(store)
        interactive = question is None or not question.strip()
        questions: Sequence[str] | Iterator[str] = (
            _read_questions() if interactive else [question]
        )
        with _handled_errors():
            for text in questions:
                _answer(consultor, text, stale, debug=debug)


@app.command()
def sources() -> None:
    """Lista las fuentes indexadas y si siguen al día respecto a lo instalado."""
    with _store(check_model=False) as store:
        statuses = check_versions(store.sources())
        meta = store.meta()
    if not statuses:
        print_hint("Nada indexado todavía.")
        return
    caption = (
        f"Embeddings: {meta.get('embed_model', 'sin registrar')} · "
        f"{meta.get('embed_dim', '?')} dimensiones · "
        f"formato {meta.get('format_version', '?')}"
    )
    show_status_table(statuses, "Fuentes indexadas", caption)


@app.command()
def update(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Reindexar sin pedir confirmación.")
    ] = False,
) -> None:
    """Reindexa los paquetes cuya versión instalada difiere de la indexada."""
    with _store() as store:
        stale = [s for s in check_versions(store.sources()) if s.is_stale]
        if not stale:
            print_hint("Todas las fuentes están al día.")
            return
        show_status_table(stale, "Fuentes obsoletas")
        if not yes and not typer.confirm("¿Reindexarlas ahora?"):
            raise typer.Exit(code=1)
        results = _index_packages(store, [s.source.package for s in stale])
    show_sources_table(results, "Reindexado")


@app.command("eval")
def eval_command(
    cases: Annotated[
        Path,
        typer.Argument(
            metavar="CASOS",
            exists=True,
            dir_okay=False,
            help="Fichero TOML con los casos de referencia.",
        ),
    ],
    k: Annotated[
        int,
        typer.Option("--k", "-k", min=1, help="Fragmentos recuperados por pregunta."),
    ] = 10,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Muestra todos los casos, no solo los fallos."
        ),
    ] = False,
) -> None:
    """Mide Recall@k y MRR de la recuperación con casos de referencia."""
    try:
        loaded = load_cases(cases)
    except ValueError as exc:
        _fail(str(exc))
    if not loaded:
        _fail(
            f"{cases}: no contiene casos.",
            "Define tablas [[case]] con question y expected.",
        )
    ks = sorted({*(value for value in DEFAULT_KS if value < k), k})
    with _store() as store:
        if not store.sources():
            _fail(
                "No hay nada indexado todavía.",
                "Usa `api index <paquete>` o `api index-docs`.",
            )
        missing = missing_sources(loaded, store.qualnames())
        if missing:
            print_warning(
                f"{len(missing)} fuente(s) esperada(s) no están en el índice "
                "y contarán como fallo.",
                f"Por ejemplo: {missing[0][1]}. "
                "¿Falta `api index <paquete>` o `api index-docs`?",
            )
        consultor = Consultor(store)
        with progress_bar() as progress, _handled_errors():
            task = progress.add_task("Evaluando", total=len(loaded))
            report = evaluate(
                consultor.retrieve,
                loaded,
                ks,
                on_result=lambda _result: progress.advance(task),
            )
    show_evaluation(report, verbose=verbose)
