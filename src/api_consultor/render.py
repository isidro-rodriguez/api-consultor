"""Presentación en consola: tema Monokai, Markdown en streaming, tablas y avisos."""

import shutil
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path

import pyfiglet
from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.constrain import Constrain
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.status import Status
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from rich.theme import Theme
from typer import rich_utils
from typer._click import decorators as click_decorators
from typer._click.core import Command
from typer._click.formatting import HelpFormatter
from typer.core import TyperOption

from .citations import Citations, doc_id
from .config import SETTINGS
from .consultor import Retrieval
from .evaluation import EvalReport
from .store import Chunk, SourceInfo
from .versions import VersionStatus

FG = "#F8F8F2"
"""Monokai: texto principal."""

BG = "#272822"
"""Monokai: fondo (bloques de código)."""

LINE = "#3E3D32"
"""Monokai: líneas y fondos tenues."""

COMMENT = "#75715E"
"""Monokai: texto secundario."""

RED = "#F92672"
"""Monokai: errores y énfasis."""

ORANGE = "#FD971F"
"""Monokai: avisos."""

YELLOW = "#E6DB74"
"""Monokai: código en línea."""

GREEN = "#A6E22E"
"""Monokai: éxito y títulos."""

BLUE = "#66D9EF"
"""Monokai: información."""

PURPLE = "#AE81FF"
"""Monokai: acento."""

THEME = Theme(
    {
        # Semánticos
        "info": BLUE,
        "success": GREEN,
        "warning": ORANGE,
        "error": f"bold {RED}",
        "muted": COMMENT,
        "accent": PURPLE,
        "title": f"bold {GREEN}",
        "prompt": f"bold {RED}",
        "status.spinner": PURPLE,
        "logging.level.warning": ORANGE,
        # Markdown
        "markdown.h1": f"bold {RED}",
        "markdown.h1.border": RED,
        "markdown.h2": f"bold {ORANGE}",
        "markdown.h3": f"bold {YELLOW}",
        "markdown.h4": f"bold {BLUE}",
        "markdown.h5": f"italic {BLUE}",
        "markdown.h6": f"italic {COMMENT}",
        "markdown.code": f"{YELLOW} on {LINE}",
        "markdown.code_block": f"{FG} on {BG}",
        "markdown.link": BLUE,
        "markdown.link_url": f"underline {COMMENT}",
        "markdown.item.bullet": f"bold {RED}",
        "markdown.item.number": f"bold {RED}",
        "markdown.block_quote": f"italic {COMMENT}",
        "markdown.hr": COMMENT,
        "markdown.strong": f"bold {FG}",
        # Barras de progreso
        "bar.back": LINE,
        "bar.complete": GREEN,
        "bar.finished": GREEN,
        "bar.pulse": PURPLE,
        "progress.download": BLUE,
        "progress.elapsed": COMMENT,
    }
)

"""Estilos semánticos (info, success, warning, error, muted, accent…) y de Markdown."""

APP_NAME = "ApiConsultor"
"""Nombre de la aplicación que se dibuja en el banner."""

TAGLINE = "Consultor de APIs Python offline con Ollama y RAG"
"""Lema bajo el banner."""

BANNER_FONT = "slant"
"""Fuente figlet del banner."""

_RAINBOW = (RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)

console = Console(theme=THEME, highlight=False)
"""Consola de la salida normal."""

err_console = Console(theme=THEME, highlight=False, stderr=True)
"""Consola de errores y avisos (salida de error)."""


_typer_console = rich_utils._get_rich_console  # API privada de Typer


def _themed_console(stderr: bool = False) -> Console:
    """Crea la consola de la ayuda de Typer con los estilos semánticos de `THEME`.

    Args:
        stderr: Si la consola escribe en la salida de error.

    Returns:
        La consola de Typer con `THEME` apilado sobre sus estilos.
    """
    rich_console = _typer_console(stderr=stderr)
    rich_console.push_theme(THEME)
    return rich_console


def _join_lines(paragraphs: Sequence[str], markup_mode: str) -> str:
    """Une los saltos de línea simples de cada párrafo de una ayuda de Typer.

    Typer los conserva en modo `rich`, así que un docstring ajustado a 88 columnas se
    vería cortado a destiempo; aquí cada párrafo pasa a ser una sola línea y Rich lo
    ajusta al ancho de la consola.

    Args:
        paragraphs: Párrafos de la ayuda, ya separados por líneas en blanco.
        markup_mode: Modo de markup de Typer (no cambia el resultado).

    Returns:
        Los párrafos unidos por una línea en blanco.
    """
    del markup_mode
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in paragraphs)


ERROR_TITLE = "Error"
"""Título del panel de errores (el mismo para los de Typer y los propios)."""

WARNING_TITLE = "Aviso"
"""Título del panel de avisos."""

TYPER_TEXTS = {
    "ARGUMENTS_PANEL_TITLE": "Argumentos",
    "OPTIONS_PANEL_TITLE": "Opciones",
    "COMMANDS_PANEL_TITLE": "Comandos",
    "ERRORS_PANEL_TITLE": ERROR_TITLE,
    "DEFAULT_STRING": "[por defecto: {}]",
    "ENVVAR_STRING": "[variable de entorno: {}]",
    "REQUIRED_LONG_STRING": "[obligatorio]",
    "DEPRECATED_STRING": "(obsoleto) ",
    "ABORTED_TEXT": "Cancelado.",
    "RICH_HELP": "Prueba [info]'{command_path} {help_option}'[/] para ver la ayuda.",
}
"""Textos de la ayuda de Typer traducidos (constantes de `typer.rich_utils`)."""

HELP_OPTION_TEXT = "Muestra esta ayuda y termina."
"""Ayuda de la opción `--help` de cada comando."""

USAGE_PREFIX = "Uso: "
"""Prefijo de la línea de uso de la ayuda."""

_typer_help_option = click_decorators.help_option  # API privada de Typer
_typer_write_usage = HelpFormatter.write_usage


def _help_option(param_decls: list[str]) -> Callable[[Command], Command]:
    """Crea la opción `--help` de Typer con su texto en español.

    Args:
        param_decls: Nombres de la opción (p. ej. `["--help"]`).

    Returns:
        El decorador que añade la opción al comando.
    """
    add_option = _typer_help_option(param_decls)

    def decorator(command: Command) -> Command:
        command = add_option(command)
        if isinstance(option := command.params[-1], TyperOption):
            option.help = HELP_OPTION_TEXT
        return command

    return decorator


def _write_usage(
    self: HelpFormatter, prog: str, args: str = "", prefix: str | None = None
) -> None:
    """Escribe la línea de uso con el prefijo en español."""
    _typer_write_usage(self, prog, args, USAGE_PREFIX if prefix is None else prefix)


def _translate_typer() -> None:
    """Traduce al español los textos fijos de la ayuda y los errores de Typer."""
    for name, text in TYPER_TEXTS.items():
        setattr(rich_utils, name, text)
    setattr(click_decorators, "help_option", _help_option)  # noqa: B010 (ídem)
    setattr(HelpFormatter, "write_usage", _write_usage)  # noqa: B010 (ídem)
    highlighter = rich_utils.OptionHighlighter
    usage = rf"(?P<usage>{USAGE_PREFIX})"
    if usage not in highlighter.highlights:
        highlighter.highlights = [*highlighter.highlights, usage]


def style_typer_help() -> None:
    """Aplica la paleta, el ancho máximo y los estilos del tema a la ayuda de Typer.

    Tras llamarla, los textos de ayuda admiten el markup semántico de `THEME`
    (`[info]`, `[muted]`, `[warning]`…) y los textos fijos de Typer salen en
    español. Es idempotente.
    """
    width = min(SETTINGS.console_width, shutil.get_terminal_size().columns)
    rich_utils.MAX_WIDTH = width
    styles = {
        "STYLE_OPTION": f"bold {RED}",
        "STYLE_SWITCH": f"bold {GREEN}",
        "STYLE_USAGE": ORANGE,
        "STYLE_USAGE_COMMAND": f"bold {FG}",
        "STYLE_COMMANDS_TABLE_FIRST_COLUMN": f"bold {GREEN}",
        "STYLE_OPTIONS_PANEL_BORDER": COMMENT,
        "STYLE_COMMANDS_PANEL_BORDER": COMMENT,
        "STYLE_ERRORS_PANEL_BORDER": "error",
        "STYLE_ERRORS_SUGGESTION": f"italic {COMMENT}",
        "STYLE_REQUIRED_SHORT": RED,
        "STYLE_REQUIRED_LONG": f"dim {RED}",
        "STYLE_OPTION_DEFAULT": f"dim {YELLOW}",
    }
    for name, style in styles.items():
        setattr(rich_utils, name, style)  # constantes tipadas como Literal en Typer
    setattr(rich_utils, "_fix_linebreaks", _join_lines)  # noqa: B010 (ídem)
    setattr(rich_utils, "_get_rich_console", _themed_console)  # noqa: B010 (ty: funciones nominales)
    _translate_typer()


def _fit(renderable: RenderableType) -> RenderableType:
    """Limita el ancho al máximo configurado (o al de la terminal, si es menor)."""
    return Constrain(renderable, SETTINGS.console_width)


def _markdown(text: str) -> RenderableType:
    return _fit(Markdown(text, code_theme="monokai"))


# --- Mensajes -----------------------------------------------------------------


def _panel(title: str, style: str, message: str, hint: str | None) -> None:
    """Imprime un mensaje en un panel con borde del tema en la salida de error.

    Args:
        title: Título del panel, a la izquierda.
        style: Estilo del tema para el borde.
        message: Qué ocurre.
        hint: Qué hacer al respecto, tras una línea en blanco y atenuado.
    """
    body = Text(message)
    if hint is not None:
        body.append("\n\n").append(hint, style="muted")
    panel = Panel(
        body, title=title, title_align="left", border_style=style, box=box.ROUNDED
    )
    err_console.print(_fit(panel))


def print_error(message: str, hint: str | None = None) -> None:
    """Imprime un error en un panel «Error» en la salida de error.

    Usa el mismo aspecto que los errores de uso de Typer: borde con el estilo
    `error` y título a la izquierda.

    Args:
        message: Qué ha fallado.
        hint: Cómo solucionarlo, tras una línea en blanco y atenuado.
    """
    _panel(ERROR_TITLE, "error", message, hint)


def print_warning(message: str, hint: str | None = None) -> None:
    """Imprime un aviso en un panel «Aviso» en la salida de error.

    Args:
        message: Qué ocurre.
        hint: Qué hacer al respecto, tras una línea en blanco y atenuado.
    """
    _panel(WARNING_TITLE, "warning", message, hint)


def print_hint(message: str) -> None:
    """Imprime una nota informativa atenuada.

    Args:
        message: Texto a mostrar.
    """
    console.print(_fit(Text(message, style="muted")))


# --- Entrada ------------------------------------------------------------------


def prompt() -> str | None:
    """Lee una línea del usuario.

    Returns:
        El texto sin espacios sobrantes, o `None` si el usuario cierra la entrada
        (Ctrl+D o Ctrl+C).
    """
    try:
        return console.input(Text("> ", style="prompt")).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None


def status(message: str) -> Status:
    """Crea un spinner para esperas cortas.

    Args:
        message: Texto que acompaña al spinner.

    Returns:
        El indicador, para usarlo como gestor de contexto.
    """
    return console.status(message, spinner="dots")


# --- Respuesta ----------------------------------------------------------------


def stream_markdown(tokens: Iterable[str]) -> str:
    """Muestra la respuesta mientras llega y la deja renderizada como Markdown.

    Durante el streaming se redibuja en vivo (con un spinner hasta el primer token);
    al terminar se imprime una sola vez completa, aunque supere el alto de la terminal.

    Args:
        tokens: Trozos de la respuesta según llegan.

    Returns:
        El texto completo de la respuesta.
    """
    buffer = ""
    with Live(
        Spinner("dots", text="Generando respuesta...", style="accent"),
        console=console,
        transient=True,
        refresh_per_second=8,
        vertical_overflow="ellipsis",
    ) as live:
        for token in tokens:
            buffer += token
            live.update(_markdown(buffer))
    console.print()
    console.print(_markdown(buffer))
    return buffer


def show_sources(
    chunks: Sequence[Chunk],
    stale: Collection[str] = (),
    citations: Citations | None = None,
    link: Callable[[Chunk], str | None] | None = None,
    unavailable: Callable[[Chunk], bool] | None = None,
) -> None:
    """Panel con los fragmentos usados, su identificador y si la respuesta los citó.

    Marca con ✔ los citados y con ⚠ las fuentes con versión obsoleta. Si el modelo
    citó identificadores inexistentes o no citó nada, lo avisa bajo el panel. Con
    `link`, el nombre es un hiperenlace de terminal a la ubicación del fragmento.

    Args:
        chunks: Fragmentos enviados al modelo, en el orden de sus `DOC-NN`.
        stale: Fuentes cuya versión instalada difiere de la indexada.
        citations: Citas de la respuesta, si se han comprobado.
        link: Función que da la URL de un fragmento (o `None` si no tiene).
        unavailable: Función que dice si el fichero de un fragmento ya no existe.
    """
    if not chunks:
        return
    cited = set(citations.valid) if citations else set()
    linked = False
    missing = False
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(style="info", overflow="fold")
    grid.add_column(style="muted", no_wrap=True)
    for position, chunk in enumerate(chunks, start=1):
        numbered = chunk.version[:1].isdigit() and chunk.version not in chunk.package
        version = f" v{chunk.version}" if numbered else ""
        origin = Text(f"{chunk.package}{version}", style="muted")
        if chunk.package in stale:
            origin.append(" ⚠ obsoleta", style="warning")
        mark = Text(doc_id(position), style="success" if position in cited else "muted")
        if position in cited:
            mark.append(" ✔", style="success")
        name = Text(chunk.qualname)
        url = link(chunk) if link is not None else None
        if url is not None:
            name.stylize(Style(link=url, underline=True), 0)
            name.stylize("accent", 0)
            linked = True
        elif unavailable is not None and unavailable(chunk):
            name.append(" (fichero no disponible)", style="warning")
            missing = True
        grid.add_row(mark, name, origin)
    panel = Panel(
        grid,
        title="Fuentes",
        title_align="left",
        subtitle=" · ".join(
            hint
            for hint, shown in (
                ("Ctrl+clic en una fuente para abrirla", linked),
                ("Revisa `api sources`: hay ficheros que ya no existen", missing),
            )
            if shown
        )
        or None,
        subtitle_align="right",
        border_style="muted",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print()
    console.print(_fit(panel))
    if citations is not None:
        _show_citation_issues(citations)
    console.print()


def _show_citation_issues(citations: Citations) -> None:
    if citations.invalid:
        names = ", ".join(citations.invalid)
        print_warning(
            f"La respuesta cita fuentes inexistentes ({names}).",
            "Ignora esas citas y verifica la información en la documentación.",
        )
    elif not citations.valid:
        print_warning(
            "La respuesta no cita ninguna fuente.",
            "Contrástala con los fragmentos listados.",
        )


def show_debug(retrieval: Retrieval, context_chars: int, limit: int = 15) -> None:
    """Muestra el diagnóstico de la recuperación (`ask --debug`).

    Args:
        retrieval: Resultado de la recuperación, con candidatos y señales.
        context_chars: Tamaño en caracteres del contexto enviado al modelo.
        limit: Cuántos candidatos listar.
    """
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="muted", justify="right")
    summary.add_column(style=FG, overflow="fold")
    summary.add_row("Nombres", ", ".join(retrieval.names) or "—")
    summary.add_row("Términos", ", ".join(retrieval.terms) or "—")
    best = retrieval.best_distance
    summary.add_row("Mejor distancia", "—" if best is None else f"{best:.3f}")
    threshold = SETTINGS.max_distance
    summary.add_row("Umbral", "desactivado" if threshold is None else f"{threshold}")
    summary.add_row("Contexto", f"{context_chars:,} caracteres")
    if retrieval.abstain_reason is not None:
        summary.add_row("Abstención", Text(retrieval.abstain_reason, style="warning"))
    table = Table(
        title="Candidatos",
        title_style="title",
        title_justify="left",
        caption="N nombre cualificado · S nombre corto · L palabras · V vectorial",
        caption_style="muted",
        caption_justify="left",
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
        expand=True,
    )
    table.add_column("", width=1, no_wrap=True)
    table.add_column("Fragmento", ratio=1, overflow="ellipsis", no_wrap=True)
    for signal in "NSLV":
        table.add_column(signal, justify="right", no_wrap=True)
    table.add_column("Dist.", justify="right", no_wrap=True)
    table.add_column("Punt.", justify="right", no_wrap=True)
    for candidate in retrieval.candidates[:limit]:
        mark = Text("✔", style="success") if candidate.selected else Text(" ")
        distance = candidate.distance
        table.add_row(
            mark,
            Text(candidate.chunk.qualname),
            *(str(candidate.ranks.get(signal, "·")) for signal in "NSLV"),
            "—" if distance is None else f"{distance:.3f}",
            f"{candidate.score * 100:.2f}",
        )
    console.print()
    console.print(_fit(summary))
    console.print(_fit(table))


# --- Indexado -----------------------------------------------------------------


def progress_bar() -> Progress:
    """Crea la barra de progreso del indexado y la evaluación.

    Returns:
        Una barra transitoria, para usarla como gestor de contexto.
    """
    return Progress(
        SpinnerColumn(style="accent", finished_text=Text("✔", style="success")),
        TextColumn(
            "{task.description}",
            style="title",
            table_column=Column(width=24, no_wrap=True, overflow="ellipsis"),
        ),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def show_sources_table(sources: Sequence[SourceInfo], title: str) -> None:
    """Muestra una tabla de fuentes con su versión y número de fragmentos.

    Args:
        sources: Fuentes a listar (si está vacío, no imprime nada).
        title: Título de la tabla.
    """
    if not sources:
        return
    table = Table(
        title=title,
        title_style="title",
        title_justify="left",
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
    )
    table.add_column("Fuente", style="info")
    table.add_column("Versión", justify="right", style="muted", no_wrap=True)
    table.add_column("Fragmentos", justify="right", style="success")
    for source in sources:
        table.add_row(Text(source.package), Text(source.version), f"{source.chunks:,}")
    console.print(_fit(table))


# --- Evaluación ---------------------------------------------------------------


def _score(value: float, *, percent: bool) -> Text:
    """Valor coloreado según su calidad: verde >= 0,8; naranja >= 0,5; rosa el resto."""
    style = "success" if value >= 0.8 else "warning" if value >= 0.5 else "error"
    return Text(f"{value:.0%}" if percent else f"{value:.3f}", style=style)


def show_evaluation(report: EvalReport, *, verbose: bool = False) -> None:
    """Muestra las métricas (globales y por categoría) y los casos fallidos.

    Args:
        report: Informe de la evaluación.
        verbose: Si es `True`, lista todos los casos y no solo los fallos.
    """
    table = Table(
        title="Evaluación de la recuperación",
        title_style="title",
        title_justify="left",
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
    )
    table.add_column("Categoría", style="info")
    table.add_column("Casos", justify="right", style="muted")
    for k in report.ks:
        table.add_column(f"R@{k}", justify="right")
    table.add_column("MRR", justify="right")
    rows = [("global", report.metrics())]
    if len(report.categories) > 1:
        rows += [(name, report.metrics(name)) for name in report.categories]
    for name, metrics in rows:
        table.add_row(
            Text(name),
            str(metrics.cases),
            *(_score(metrics.recall[k], percent=True) for k in report.ks),
            _score(metrics.mrr, percent=False),
        )
    console.print(_fit(table))

    within = 5 if 5 in report.ks else report.ks[-1]
    shown = list(report.results) if verbose else report.failures(within)
    if not shown:
        console.print(
            Text(f"\n✔ Todos los casos aciertan en el top-{within}", style="success")
        )
        return
    detail = Table(
        title="Casos" if verbose else f"Fallos (fuera del top-{within})",
        title_style="title",
        title_justify="left",
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
        expand=True,
    )
    detail.add_column("Rango", justify="right", no_wrap=True, min_width=5)
    for name, ratio in (("Pregunta", 5), ("Esperado", 4), ("Primer resultado", 5)):
        detail.add_column(name, ratio=ratio, overflow="ellipsis", no_wrap=True)
    for result in shown:
        rank = result.rank
        cell = (
            Text("—", style="error")
            if rank is None
            else Text(str(rank), style="success" if rank <= within else "warning")
        )
        first = result.ranked[0] if result.ranked else "—"
        detail.add_row(
            cell,
            Text(result.case.question),
            Text(", ".join(result.case.expected)),
            Text(first, style="muted"),
        )
    console.print()
    console.print(_fit(detail))


# --- Versión ------------------------------------------------------------------


def _banner(max_width: int) -> RenderableType:
    """Nombre de la app en figlet, una línea de la paleta por fila de texto."""
    lines = (
        pyfiglet.figlet_format(APP_NAME, font=BANNER_FONT, width=200)
        .rstrip()
        .splitlines()
    )
    width = max(len(line) for line in lines)
    if width > max_width:  # terminal muy estrecha: sin arte
        return Align.center(Text(APP_NAME, style=f"bold {GREEN}"))
    art = Text(no_wrap=True)
    for index, line in enumerate(lines):
        art.append(line.ljust(width), style=_RAINBOW[index % len(_RAINBOW)])
        if index < len(lines) - 1:
            art.append("\n")
    return Align.center(art)


def show_version(version: str, details: Sequence[tuple[str, str]]) -> None:
    """Muestra el panel con el banner, la versión y datos del entorno.

    Args:
        version: Versión de la aplicación.
        details: Pares (etiqueta, valor) del entorno.
    """
    available = min(console.width, SETTINGS.console_width) - 6  # borde + relleno
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", justify="right")
    grid.add_column(style=FG)
    for label, value in details:
        grid.add_row(label, value)
    body = Group(
        _banner(available),
        Align.center(Text(TAGLINE, style="muted")),
        Text(),
        Align.center(grid),
    )
    panel = Panel(
        body,
        title=Text(f" v{version} ", style="title"),
        title_align="right",
        border_style="muted",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(_fit(panel))


_STATE_LABELS = {
    "ok": ("✔", "al día", "success"),
    "stale": ("⚠", "obsoleta", "warning"),
    "missing": ("○", "no instalada", "muted"),
    "unknown": ("?", "sin versión", "muted"),
    "docs": ("·", "docs", "muted"),
    "gone": ("✘", "no encontrada", "error"),
}
"""Símbolo, texto y estilo de cada estado (todos con símbolo para alinear el texto)."""


def _environment(source: SourceInfo) -> str | None:
    """Entorno donde se indexó un paquete, abreviado para la tabla.

    Muestra la carpeta del venv (no su `bin/python`) con `~` en lugar de la carpeta
    personal, o «herramienta» si se indexó en el propio entorno.
    """
    if source.kind == "docs":
        return None
    if not source.python:
        return "herramienta"
    python = Path(source.python)
    folder = (
        python.parent.parent if python.parent.name in {"bin", "Scripts"} else python
    )
    home = Path.home()
    return (
        f"~/{folder.relative_to(home).as_posix()}"
        if folder.is_relative_to(home)
        else str(folder)
    )


def show_status_table(
    statuses: Sequence[VersionStatus],
    title: str,
    caption: str | None = None,
    *,
    environment: bool = False,
) -> None:
    """Muestra las fuentes indexadas con su versión instalada y si están al día.

    Args:
        statuses: Estado de cada fuente.
        title: Título de la tabla.
        caption: Pie de tabla opcional.
        environment: Añade bajo cada paquete el entorno donde se indexó.
    """
    table = Table(
        title=title,
        title_style="title",
        title_justify="left",
        caption=caption,
        caption_style="muted",
        caption_justify="left",
        show_lines=environment,  # el entorno ocupa una segunda línea
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
    )
    table.add_column("Fuente", style="info", no_wrap=not environment, overflow="fold")
    table.add_column("Indexada", justify="right", style="muted", no_wrap=True)
    table.add_column("Instalada", justify="right", style=FG, no_wrap=True)
    table.add_column("Estado", no_wrap=True)
    table.add_column("Fragmentos", justify="right", style="success")
    table.add_column("Fecha", style="muted")
    for entry in statuses:
        source = entry.source
        symbol, label, style = _STATE_LABELS[entry.state]
        name = Text(source.package)
        if environment and (where := _environment(source)) is not None:
            name.append(f"\n{where}", style="muted")
        table.add_row(
            name,
            Text(source.version or "—"),
            Text(entry.installed or "—"),
            Text(f"{symbol} {label}", style=style),
            f"{source.chunks:,}",
            source.indexed_at[:10] or "—",
        )
    console.print(_fit(table))


def show_user_config(
    rows: Sequence[tuple[str, str, bool]], path: Path, *, exists: bool
) -> None:
    """Muestra las preferencias efectivas y cuáles vienen de `config.toml`.

    Args:
        rows: Tuplas (opción, valor, definido por el usuario).
        path: Ruta de `config.toml`.
        exists: Si el fichero existe.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", no_wrap=True)
    grid.add_column(style=FG, overflow="fold")
    grid.add_column(style="muted", no_wrap=True)
    for name, value, custom in rows:
        grid.add_row(
            Text(name), Text(value), "config.toml" if custom else "por defecto"
        )
    note = "" if exists else " (no existe)"
    console.print(
        _fit(
            Panel(
                grid,
                title="Preferencias",
                title_align="left",
                border_style="muted",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
    )
    console.print(Text(f"Fichero: {path}{note}", style="muted"), soft_wrap=True)
