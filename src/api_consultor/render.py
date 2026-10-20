"""Presentación en consola: tema Monokai, Markdown en streaming, tablas y avisos."""

import shutil
from collections.abc import Iterable, Sequence

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
from rich.table import Column, Table
from rich.text import Text
from rich.theme import Theme
from typer import rich_utils

from .config import SETTINGS
from .evaluation import EvalReport
from .store import Chunk, SourceInfo

# Paleta Monokai
FG = "#F8F8F2"
BG = "#272822"
LINE = "#3E3D32"
COMMENT = "#75715E"
RED = "#F92672"
ORANGE = "#FD971F"
YELLOW = "#E6DB74"
GREEN = "#A6E22E"
BLUE = "#66D9EF"
PURPLE = "#AE81FF"

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

APP_NAME = "ApiConsultor"
TAGLINE = "Consultor de APIs Python offline con Ollama y RAG"
BANNER_FONT = "slant"
_RAINBOW = (RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)

console = Console(theme=THEME, highlight=False)
err_console = Console(theme=THEME, highlight=False, stderr=True)


def style_typer_help() -> None:
    """Aplica la paleta y el ancho máximo a la ayuda que genera Typer."""
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
        "STYLE_ERRORS_PANEL_BORDER": RED,
        "STYLE_ERRORS_SUGGESTION": f"italic {COMMENT}",
        "STYLE_REQUIRED_SHORT": RED,
        "STYLE_REQUIRED_LONG": f"dim {RED}",
        "STYLE_OPTION_DEFAULT": f"dim {YELLOW}",
    }
    for name, style in styles.items():
        setattr(rich_utils, name, style)  # constantes tipadas como Literal en Typer


def _fit(renderable: RenderableType) -> RenderableType:
    """Limita el ancho al máximo configurado (o al de la terminal, si es menor)."""
    return Constrain(renderable, SETTINGS.console_width)


def _markdown(text: str) -> RenderableType:
    return _fit(Markdown(text, code_theme="monokai"))


# --- Mensajes -----------------------------------------------------------------


def _message(
    target: Console, symbol: str, style: str, text: str, hint: str | None
) -> None:
    target.print(_fit(Text.assemble((f"{symbol} ", style), text)))
    if hint is not None:
        target.print(_fit(Text(f"  {hint}", style="muted")))


def print_error(message: str, hint: str | None = None) -> None:
    _message(err_console, "✖", "error", message, hint)


def print_warning(message: str, hint: str | None = None) -> None:
    _message(err_console, "!", "warning", message, hint)


def print_hint(message: str) -> None:
    console.print(_fit(Text(message, style="muted")))


# --- Entrada ------------------------------------------------------------------


def prompt() -> str | None:
    """Lee una línea; devuelve None si el usuario cierra (Ctrl+D o Ctrl+C)."""
    try:
        return console.input(Text("> ", style="prompt")).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None


def status(message: str) -> Status:
    """Spinner para esperas cortas (usar como gestor de contexto)."""
    return console.status(message, spinner="dots")


# --- Respuesta ----------------------------------------------------------------


def stream_markdown(tokens: Iterable[str]) -> None:
    """Muestra la respuesta mientras llega y la deja renderizada como Markdown.

    Durante el streaming se redibuja en vivo (con un spinner hasta el primer token);
    al terminar se imprime una sola vez completa, aunque supere el alto de la terminal.
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


def show_sources(chunks: Sequence[Chunk]) -> None:
    """Panel con los fragmentos usados para responder."""
    if not chunks:
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="info", overflow="fold")
    grid.add_column(style="muted", no_wrap=True)
    for chunk in chunks:
        numbered = chunk.version[:1].isdigit() and chunk.version not in chunk.package
        version = f" v{chunk.version}" if numbered else ""
        grid.add_row(chunk.qualname, f"{chunk.package}{version}")
    panel = Panel(
        grid,
        title="Fuentes",
        title_align="left",
        border_style="muted",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print()
    console.print(_fit(panel))
    console.print()


# --- Indexado -----------------------------------------------------------------


def progress_bar() -> Progress:
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
    table.add_column("Versión", style="muted")
    table.add_column("Fragmentos", justify="right", style="success")
    for source in sources:
        table.add_row(source.package, source.version, f"{source.chunks:,}")
    console.print(_fit(table))


# --- Evaluación ---------------------------------------------------------------


def _score(value: float, *, percent: bool) -> Text:
    """Valor coloreado según su calidad: verde >= 0,8; naranja >= 0,5; rosa el resto."""
    style = "success" if value >= 0.8 else "warning" if value >= 0.5 else "error"
    return Text(f"{value:.0%}" if percent else f"{value:.3f}", style=style)


def show_evaluation(report: EvalReport, *, verbose: bool = False) -> None:
    """Tabla de métricas (global y por categoría) y lista de casos fallidos."""
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
            name,
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
            result.case.question,
            ", ".join(result.case.expected),
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
    """Panel con el banner, la versión y datos del entorno."""
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
