"""Presentación en consola: tema Monokai, Markdown en streaming, tablas y avisos."""

import shutil
from collections.abc import Iterable, Sequence

from rich import box
from rich.console import Console, RenderableType
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

console = Console(theme=THEME, highlight=False)
err_console = Console(theme=THEME, highlight=False, stderr=True)


def style_typer_help() -> None:
    """Aplica la paleta y el ancho máximo a la ayuda que genera Typer."""
    width = min(SETTINGS.console_width, shutil.get_terminal_size().columns)
    rich_utils.MAX_WIDTH = width
    rich_utils.STYLE_OPTION = f"bold {RED}"
    rich_utils.STYLE_SWITCH = f"bold {GREEN}"
    rich_utils.STYLE_USAGE = ORANGE
    rich_utils.STYLE_USAGE_COMMAND = f"bold {FG}"
    rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {GREEN}"
    rich_utils.STYLE_OPTIONS_PANEL_BORDER = COMMENT
    rich_utils.STYLE_COMMANDS_PANEL_BORDER = COMMENT
    rich_utils.STYLE_ERRORS_PANEL_BORDER = RED
    rich_utils.STYLE_ERRORS_SUGGESTION = f"italic {COMMENT}"
    rich_utils.STYLE_REQUIRED_SHORT = RED
    rich_utils.STYLE_REQUIRED_LONG = f"dim {RED}"
    rich_utils.STYLE_OPTION_DEFAULT = f"dim {YELLOW}"


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
    if hint:
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
