"""Interfaz de línea de comandos (Typer)."""

import logging
import platform
import shutil
import sqlite3
import subprocess
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from importlib.resources import as_file
from pathlib import Path
from typing import Annotated, NoReturn, cast

import ollama
import sqlite_vec
import typer
from rich.logging import RichHandler
from rich.progress import Progress

from . import __version__
from .citations import check_citations
from .config import SETTINGS
from .consultor import Consultor, format_context
from .docs import Files, collect, extract_doc_chunks, import_docs, materialize
from .evaluation import DEFAULT_KS, evaluate, load_cases, missing_sources
from .external import (
    ProbeError,
    extract_remote,
    is_own,
    remote_stdlib,
    resolve_interpreter,
)
from .indexer import extract_package, stdlib_modules
from .llm import embed
from .models import base_model, has_model, installed_models, modelfile
from .records import Chunk, Extraction, SourceInfo, relativize
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
    show_user_config,
    show_version,
    status,
    stream_markdown,
    style_typer_help,
)
from .store import IndexIntegrityError, Store
from .userconfig import (
    UserConfig,
    UserConfigError,
    chunk_file,
    chunk_link,
    config_path,
    create,
    effective,
    load,
)
from .versions import check_versions, collapse_stdlib

BATCH_SIZE = 32
"""Fragmentos que se envían a Ollama por petición de embeddings."""

style_typer_help()
app = typer.Typer(
    name="api",
    help="Consultor de APIs Python offline con Ollama y RAG.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
    options_metavar="[OPCIONES]",
    subcommand_metavar="COMANDO [ARGS]...",
)
"""Aplicación Typer: cada comando lleva `short_help` (una línea) y docstring."""


def _version_details() -> list[tuple[str, str]]:
    return [
        ("Python", platform.python_version()),
        ("SQLite", f"{sqlite3.sqlite_version} · sqlite-vec {sqlite_vec.__version__}"),
        ("Modelo", SETTINGS.chat_model),
        ("Embeddings", SETTINGS.embed_model),
    ]


def _load_config() -> UserConfig:
    """Lee `config.toml` y lo aplica a `SETTINGS`; falla con claridad si es inválido."""
    try:
        user = load()
    except UserConfigError as exc:
        _fail(str(exc), exc.hint)
    user.apply(SETTINGS)
    return user


def _version_callback(value: bool) -> None:
    if value:
        _load_config()
        show_version(__version__, _version_details())
        raise typer.Exit


@app.callback()
def _setup(
    ctx: typer.Context,
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
    ctx.obj = _load_config()
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
    except (IndexIntegrityError, UserConfigError) as exc:
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
                "Ejecuta `api setup` para descargar y crear los modelos.",
            )
        _fail(f"Error de Ollama: {exc.error}")


def _index_source(
    store: Store,
    progress: Progress,
    name: str,
    build: Callable[[], Extraction],
    kind: str = "package",
) -> SourceInfo | None:
    """Extrae los chunks de una fuente, calcula sus embeddings y los guarda."""
    task = progress.add_task(name, total=None)  # barra "pulsante" mientras se extrae
    try:
        extraction = build()
    except Exception:
        progress.remove_task(task)
        raise
    chunks = extraction.chunks
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
    store.replace(
        name, chunks, embeddings, kind, root=extraction.root, python=extraction.python
    )
    return SourceInfo(
        name,
        chunks[0].version,
        len(chunks),
        kind,
        root=extraction.root,
        python=extraction.python,
    )


def _index_packages(
    store: Store, packages: Sequence[str], python: str = "", kind: str = "package"
) -> list[SourceInfo]:
    """Indexa paquetes del entorno de la herramienta o, con `python`, de otro."""
    remote = not is_own(python)
    results: list[SourceInfo] = []
    with progress_bar() as progress, _handled_errors():
        for package in packages:
            build = (
                partial(extract_remote, python, package)
                if remote
                else partial(extract_package, package)
            )
            try:
                info = _index_source(store, progress, package, build, kind)
            except ModuleNotFoundError as exc:
                where = f"en {python}" if remote else "en el entorno de la herramienta"
                print_warning(
                    f"{package}: módulo no encontrado ({exc.name}).",
                    f"¿Está instalado {where}? "
                    "Indica el entorno del proyecto con `--python .venv`.",
                )
                continue
            except ProbeError as exc:
                print_warning(f"{package}: {exc}")
                continue
            except (ImportError, ValueError, TypeError) as exc:  # rota o inválido
                print_warning(f"{package}: no se pudo importar ({exc}).")
                continue
            if info:
                results.append(info)
    return results


@app.command(
    name="index",
    short_help="Indexa paquetes Python instalados. [muted](firmas y docstrings)[/]",
)
def index_command(
    packages: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[PAQUETE]...",
            help="Paquetes instalados en el entorno.",
            show_default=False,
        ),
    ] = None,
    stdlib: Annotated[
        bool,
        typer.Option(
            "--stdlib",
            help="Indexa toda la biblioteca estándar de Python (≈200 módulos).",
        ),
    ] = False,
    python: Annotated[
        Path | None,
        typer.Option(
            "--python",
            "-p",
            metavar="ENTORNO",
            help="Entorno virtual (o su Python) donde están instalados; "
            "por defecto, el de la herramienta.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Indexa paquetes Python instalados (firmas y docstrings).

    Importa cada paquete y guarda un fragmento por módulo, clase, función y propiedad
    pública con su firma, docstring, fichero y línea. Por defecto usa el entorno de la
    herramienta; con [info]--python[/] indexa los paquetes del venv de un proyecto.
    Reindexar un paquete sustituye su versión anterior en el índice.
    """
    names = [] if packages is None else packages
    if not names and not stdlib:
        _fail("Indica algún paquete o `--stdlib`.", "Ejemplo: `api index requests`.")
    interpreter = ""
    if python is not None:
        try:
            interpreter = str(resolve_interpreter(python))
        except FileNotFoundError as exc:
            _fail(str(exc), "Indica la carpeta del entorno (`.venv`) o su Python.")
    with _store() as store:
        results: list[SourceInfo] = []
        if stdlib:
            modules = _stdlib_names(interpreter)
            names = [name for name in names if name not in modules]
            results += _index_packages(store, modules, interpreter, "stdlib")
        results += _index_packages(store, names, interpreter)
    show_sources_table(results, "Indexado")


def _stdlib_names(python: str) -> list[str]:
    """Módulos de la biblioteca estándar del entorno de la herramienta o de `python`."""
    if is_own(python):
        return stdlib_modules()
    try:
        return remote_stdlib(python)
    except ProbeError as exc:
        _fail(str(exc))


def _docs_cache() -> Path:
    """Donde se extraen los .zip de documentación para poder enlazarlos."""
    return SETTINGS.db_path.parent / "docs-cache"


def _doc_chunks(
    directory: Path, source: str, files: Files, exclude: Sequence[str]
) -> Extraction:
    root = materialize(directory, source, _docs_cache(), exclude)
    chunks = relativize(extract_doc_chunks(source, files, exclude, root), root)
    return Extraction(chunks, str(root))


@app.command(
    name="index-docs",
    short_help="Indexa documentación local. "
    "[muted](txt, md o rst en ficheros, carpetas o zip)[/]",
)
def index_docs_command(
    directory: Annotated[
        Path | None,
        typer.Argument(
            metavar="DIRECTORIO",
            exists=True,
            file_okay=False,
            help="Carpeta con .txt, .md, .rst o .zip (por defecto, la de documentación "
            "del usuario).",
            show_default=False,
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-x",
            metavar="PATRÓN",
            help="Patrón fnmatch de ruta a omitir (repetible), p. ej. 'whatsnew/*'.",
        ),
    ] = None,
    copy: Annotated[
        bool | None,
        typer.Option(
            "--copy/--no-copy",
            help="Copiar (o no) la documentación a la carpeta del usuario sin "
            "preguntar.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Indexa documentación local (.txt, .md, .rst; sueltos, en carpetas o en .zip).

    Cada carpeta o .zip del directorio es una fuente y los ficheros sueltos forman la
    fuente «docs»; el texto se trocea por secciones y cada fragmento guarda su enlace.
    Los .zip se extraen a [muted]docs-cache/[/]. Si el directorio está fuera de la
    carpeta de documentación del usuario, ofrece copiarlo allí para no perderlo al
    mover o borrar el original.
    """
    directory = directory or SETTINGS.docs_dir
    if not directory.is_dir():
        _fail(
            f"No existe la carpeta de documentación: {directory}.",
            "Copia ahí tu documentación o indica otra: `api index-docs <carpeta>`.",
        )
    directory = _offer_copy(directory, copy=copy)
    results: list[SourceInfo] = []
    with _store() as store, progress_bar() as progress, _handled_errors():
        for source, files in collect(directory).items():
            build = partial(_doc_chunks, directory, source, files, exclude or ())
            if info := _index_source(store, progress, source, build, "docs"):
                results.append(info)
    show_sources_table(results, "Indexado")


def _offer_copy(directory: Path, *, copy: bool | None) -> Path:
    """Ofrece copiar la documentación a la carpeta del usuario y devuelve la efectiva.

    Allí no se pierde si se mueve o borra el original. No pregunta si ya está dentro
    de esa carpeta (o la contiene).
    """
    target = SETTINGS.docs_dir
    source, home = directory.resolve(), target.resolve()
    if source == home or source.is_relative_to(home) or home.is_relative_to(source):
        return directory
    if copy is None:
        copy = typer.confirm(
            f"¿Copiar la documentación a {target} para conservarla aunque muevas "
            "o borres el original?",
            default=True,
        )
    if not copy:
        return directory
    copied = import_docs(directory, target)
    print_hint(f"Copiados {copied} elementos a {target}.")
    return target


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


def _is_unavailable(chunk: Chunk, roots: Mapping[str, str]) -> bool:
    """Indica si el fichero de un fragmento ya no está donde se indexó."""
    file = chunk_file(roots.get(chunk.package, ""), chunk)
    return file is not None and not file.exists()


def _answer(
    consultor: Consultor,
    question: str,
    stale: frozenset[str],
    config: UserConfig,
    roots: Mapping[str, str],
    *,
    debug: bool = False,
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
    show_sources(
        retrieval.chunks,
        stale,
        citations,
        link=lambda c: chunk_link(config, c, roots.get(c.package, "")),
        unavailable=lambda c: _is_unavailable(c, roots),
    )


@app.command(
    name="ask",
    short_help="Consulta lo indexado y responde con el modelo local.",
)
def ask_command(
    ctx: typer.Context,
    question: Annotated[
        str | None,
        typer.Argument(
            metavar="[PREGUNTA]",
            help="Pregunta; sin ella, modo interactivo.",
            show_default=False,
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug", "-d", help="Muestra candidatos, señales y tamaño del contexto."
        ),
    ] = False,
) -> None:
    r"""Consulta lo indexado y responde con el modelo local.

    Combina búsqueda semántica y por palabras clave, y responde solo con los fragmentos
    recuperados, citándolos como [info]\[DOC-01][/]. Debajo muestra las fuentes y marca
    las citadas; si no encuentra base suficiente, se abstiene en lugar de inventar. Sin
    pregunta entra en modo interactivo ([muted]salir[/] o Ctrl+D para terminar).
    """
    with _store() as store:
        if not store.sources():
            _fail(
                "No hay nada indexado todavía.",
                "Usa `api index <paquete>`, `api index --stdlib` o `api index-docs`.",
            )
        stale = _warn_stale(store)
        roots = {source.package: source.root for source in store.sources()}
        consultor = Consultor(store)
        interactive = question is None or not question.strip()
        questions: Sequence[str] | Iterator[str] = (
            _read_questions() if interactive else [question]
        )
        with _handled_errors():
            for text in questions:
                _answer(consultor, text, stale, ctx.obj, roots, debug=debug)


@app.command(
    name="sources",
    short_help="Administra las fuentes indexadas.",
)
def sources_command(
    remove: Annotated[
        list[str] | None,
        typer.Option(
            "--remove",
            "-r",
            metavar="FUENTE",
            help="Elimina una fuente del índice (repetible).",
            show_default=False,
        ),
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Eliminar sin pedir confirmación.")
    ] = False,
    show_all: Annotated[
        bool,
        typer.Option(
            "--all", "-a", help="Muestra cada módulo de la biblioteca estándar."
        ),
    ] = False,
    relocate: Annotated[
        tuple[str, Path] | None,
        typer.Option(
            "--relocate",
            metavar="FUENTE CARPETA",
            help="Indica la nueva carpeta de una fuente movida, sin reindexar.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Administra las fuentes indexadas.

    Sin opciones, lista cada fuente con su versión indexada e instalada, fragmentos y
    estado (al día, obsoleta o no encontrada); los módulos de la biblioteca estándar se
    agrupan en una fila. También elimina fuentes ([info]--remove stdlib[/] las borra
    todas) o recoloca una cuya carpeta se ha movido.
    """
    with _store(check_model=False) as store:
        if remove is not None:
            _remove_sources(store, remove, yes=yes)
        if relocate is not None:
            _relocate_source(store, *relocate)
        statuses = check_versions(store.sources())
        meta = store.meta()
    if not statuses:
        print_hint("Nada indexado todavía.")
        return
    shown = statuses if show_all else collapse_stdlib(statuses)
    caption = (
        f"Embeddings: {meta.get('embed_model', 'sin registrar')} · "
        f"{meta.get('embed_dim', '?')} dimensiones · "
        f"formato {meta.get('format_version', '?')}"
    )
    if len(shown) < len(statuses):
        caption += " · `--all` muestra cada módulo estándar"
    show_status_table(shown, "Fuentes indexadas", caption)


def _relocate_source(store: Store, name: str, folder: Path) -> None:
    """Apunta una fuente a otra carpeta si allí están sus ficheros."""
    if name not in {source.package for source in store.sources()}:
        _fail(f"Fuente no indexada: {name}.")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        _fail(f"No existe la carpeta {root}.")
    paths = store.paths(name)
    found = sum((root / path).exists() for path in paths)
    if paths and not found:
        _fail(
            f"No encuentro en {root} ningún fichero de {name} (p. ej. {paths[0]}).",
            "Indica la carpeta que contiene esa estructura.",
        )
    store.relocate(name, str(root))
    print_hint(f"{name} ahora apunta a {root} ({found} de {len(paths)} ficheros).")


def _remove_sources(store: Store, names: list[str], *, yes: bool) -> None:
    """Elimina `names` del índice; no toca nada si alguna no existe."""
    known = {source.package: source for source in store.sources()}
    if "stdlib" in names and "stdlib" not in known:  # alias de todos los módulos
        every = [name for name, source in known.items() if source.kind == "stdlib"]
        names = [
            name for item in names for name in (every if item == "stdlib" else [item])
        ]
    names = list(dict.fromkeys(names))
    unknown = [name for name in names if name not in known]
    if unknown:
        print_error(
            f"Fuente no indexada: {', '.join(unknown)}.",
            f"Disponibles: {', '.join(known) or 'ninguna'}.",
        )
        raise typer.Exit(code=1)
    if not yes and not typer.confirm(
        f"¿Eliminar {', '.join(names)} del índice? Se borrarán "
        f"{sum(known[name].chunks for name in names)} fragmentos."
    ):
        raise typer.Exit(code=1)
    for name in names:
        removed = store.remove(name)
        shutil.rmtree(_docs_cache() / name, ignore_errors=True)  # extracción de un .zip
        print_hint(f"Eliminada {name}: {removed} fragmentos.")


def _ollama(*args: str) -> None:
    """Ejecuta la CLI de Ollama mostrando su progreso; falla si termina con error."""
    result = subprocess.run(["ollama", *args], check=False)
    if result.returncode != 0:
        _fail(f"`ollama {args[0]}` terminó con el código {result.returncode}.")


@app.command(
    name="setup",
    short_help="Descarga los modelos de Ollama y prepara el chat.",
)
def setup_command(
    stdlib: Annotated[
        bool | None,
        typer.Option(
            "--stdlib/--no-stdlib",
            help="Indexar (o no) la biblioteca estándar sin preguntar.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Descarga los modelos de Ollama, crea el de chat y ofrece indexar la stdlib.

    Requiere Ollama instalado y en ejecución. Descarga el modelo base y el de
    embeddings si faltan, y crea el modelo de chat con el Modelfile incluido en el
    paquete. Es idempotente: puedes repetirlo tras actualizar la herramienta.
    """
    if shutil.which("ollama") is None:
        _fail(
            "No se encuentra el comando `ollama`.",
            "Instálalo desde https://ollama.com y vuelve a ejecutar `api setup`.",
        )
    with _handled_errors():
        installed = installed_models()
        for name in (base_model(), SETTINGS.embed_model):
            if has_model(installed, name):
                print_hint(f"{name}: ya descargado.")
            else:
                print_hint(f"Descargando {name}...")
                _ollama("pull", name)
        print_hint(f"Creando el modelo {SETTINGS.chat_model}...")
        with as_file(modelfile()) as path:
            _ollama("create", SETTINGS.chat_model, "-f", str(path))
    _offer_stdlib(stdlib)
    print_hint("Listo. Indexa algo (`api index <paquete>`) y pregunta con `api ask`.")


def _offer_stdlib(answer: bool | None) -> None:
    """Ofrece indexar la biblioteca estándar si todavía no está en el índice."""
    with _store() as store:
        if any(source.kind == "stdlib" for source in store.sources()):
            return
        if answer is None:
            answer = typer.confirm(
                "¿Indexar ahora la biblioteca estándar de Python? Son unos 200 módulos "
                "y tarda unos minutos (después: `api index --stdlib -p <python>` para "
                "otra versión).",
                default=True,
            )
        if answer:
            show_sources_table(
                _index_packages(store, _stdlib_names(""), "", "stdlib"), "Indexado"
            )


@app.command(
    name="config",
    short_help="Muestra la configuración efectiva.",
)
def config_command(ctx: typer.Context) -> None:
    """Muestra las preferencias efectivas; ofrece crear `config.toml` si no existe.

    Lista cada ajuste con su valor y marca los que cambia tu [info]config.toml[/] frente
    a los valores por defecto. Si el fichero no existe, ofrece crearlo a partir de una
    plantilla comentada.
    """
    current = cast(UserConfig, ctx.obj)
    path = config_path()
    show_user_config(
        effective(current, SETTINGS), path, exists=current.source is not None
    )
    if current.source is None and typer.confirm(f"¿Crear {path} con la plantilla?"):
        print_hint(f"Creado {create(path)}. Edítalo y vuelve a ejecutar el comando.")


@app.command(
    name="update",
    short_help="Reindexa los paquetes cuya versión instalada difiere de la indexada.",
)
def update_command(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Reindexar sin pedir confirmación.")
    ] = False,
) -> None:
    """Reindexa los paquetes cuya versión instalada difiere de la indexada.

    Compara cada fuente con la versión instalada en el entorno donde se indexó (el de
    la herramienta o el de [info]--python[/]) y, tras confirmar, reindexa las
    obsoletas. La documentación local no se comprueba: reindéxala con
    [info]api index-docs[/].
    """
    with _store() as store:
        stale = [s for s in check_versions(store.sources()) if s.is_stale]
        if not stale:
            print_hint("Todas las fuentes están al día.")
            return
        show_status_table(collapse_stdlib(stale), "Fuentes obsoletas")
        if not yes and not typer.confirm("¿Reindexarlas ahora?"):
            raise typer.Exit(code=1)
        results: list[SourceInfo] = []
        for python, kind in dict.fromkeys(
            (s.source.python, s.source.kind) for s in stale
        ):
            names = [
                s.source.package
                for s in stale
                if (s.source.python, s.source.kind) == (python, kind)
            ]
            results += _index_packages(store, names, python, kind)
    show_sources_table(results, "Reindexado")


@app.command(
    name="eval",
    short_help="Mide Recall@k y MRR de la recuperación con casos de referencia.",
)
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
    r"""Mide Recall@k y MRR de la recuperación con casos de referencia.

    Lee preguntas con sus fuentes esperadas de un TOML ([muted]\[\[case]][/] con
    question y expected) y comprueba si la recuperación las encuentra entre los
    primeros resultados, sin llamar al modelo de chat. Por defecto solo muestra los
    casos que fallan.
    """
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
                "Usa `api index <paquete>`, `api index --stdlib` o `api index-docs`.",
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
