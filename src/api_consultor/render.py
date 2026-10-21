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

from .citations import Citations, doc_id
from .config import SETTINGS
from .consultor import Retrieval
from .evaluation import EvalReport
from .store import Chunk, SourceInfo
from .userconfig import UserConfig
from .versions import VersionStatus

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


def stream_markdown(tokens: Iterable[str]) -> str:
    """Muestra la respuesta mientras llega y la deja renderizada como Markdown.

    Durante el streaming se redibuja en vivo (con un spinner hasta el primer token);
    al terminar se imprime una sola vez completa, aunque supere el alto de la terminal.
    Devuelve el texto completo de la respuesta.
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
) -> None:
    """Panel con los fragmentos usados, su identificador y si la respuesta los citó.

    Marca con ✔ los citados y con ⚠ las fuentes con versión obsoleta. Si el modelo
    citó identificadores inexistentes o no citó nada, lo avisa bajo el panel. Con
    `link`, el nombre es un hiperenlace de terminal a la ubicación del fragmento.
    """
    if not chunks:
        return
    cited = set(citations.valid) if citations else set()
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
            name.stylize(Style(link=url, underline=True))
        grid.add_row(mark, name, origin)
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
    """Diagnóstico de la recuperación: señales, candidatos y tamaño del contexto."""
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
            candidate.chunk.qualname,
            *(str(candidate.ranks.get(signal, "·")) for signal in "NSLV"),
            "—" if distance is None else f"{distance:.3f}",
            f"{candidate.score * 100:.2f}",
        )
    console.print()
    console.print(_fit(summary))
    console.print(_fit(table))


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


_STATE_LABELS = {
    "ok": ("✔ al día", "success"),
    "stale": ("⚠ obsoleta", "warning"),
    "missing": ("no instalada", "muted"),
    "unknown": ("? sin versión", "muted"),
    "docs": ("docs", "muted"),
}


def show_status_table(
    statuses: Sequence[VersionStatus], title: str, caption: str | None = None
) -> None:
    """Fuentes indexadas con su versión instalada y si están al día."""
    table = Table(
        title=title,
        title_style="title",
        title_justify="left",
        caption=caption,
        caption_style="muted",
        caption_justify="left",
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
    )
    table.add_column("Fuente", style="info")
    table.add_column("Indexada", style="muted")
    table.add_column("Instalada", style=FG)
    table.add_column("Estado")
    table.add_column("Fragmentos", justify="right", style="success")
    table.add_column("Fecha", style="muted")
    for status in statuses:
        source = status.source
        label, style = _STATE_LABELS[status.state]
        table.add_row(
            source.package,
            source.version,
            status.installed or "—",
            Text(label, style=style),
            f"{source.chunks:,}",
            source.indexed_at[:10] or "—",
        )
    console.print(_fit(table))


def show_user_config(config: UserConfig, path: Path) -> None:
    """Muestra el fichero de preferencias y los valores efectivos."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", justify="right")
    grid.add_column(style=FG, overflow="fold")
    grid.add_row("Fichero", f"{path}" + ("" if config.source else " (no existe)"))
    grid.add_row("Editor", " ".join(config.command))
    grid.add_row("Enlace", config.link)
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
