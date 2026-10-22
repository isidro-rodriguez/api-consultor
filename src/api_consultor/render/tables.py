"""Tablas: progreso del indexado, fuentes, estados, evaluación y preferencias."""

from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

from ..evaluation import Abstention, EvalReport
from ..records import SourceInfo
from ..versions import VersionStatus
from . import theme
from .theme import FG


def progress_bar() -> Progress:
    """Crea la barra de progreso del indexado y la evaluación.

    Va por la salida de error, como los avisos y el logging: así Rich pinta cada aviso
    por encima de la barra en vez de pegado a ella.

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
        console=theme.err_console,
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
    theme.console.print(theme.fit(table))


def _score(value: float, *, percent: bool) -> Text:
    """Valor coloreado según su calidad: verde >= 0,8; naranja >= 0,5; rosa el resto."""
    style = "success" if value >= 0.8 else "warning" if value >= 0.5 else "error"
    return Text(f"{value:.0%}" if percent else f"{value:.3f}", style=style)


def show_evaluation(
    report: EvalReport, *, verbose: bool = False, threshold: float | None = None
) -> None:
    """Muestra las métricas (globales y por categoría), los fallos y la abstención.

    Args:
        report: Informe de la evaluación.
        verbose: Si es `True`, lista todos los casos y no solo los fallos.
        threshold: `max_distance` configurado, para medir su abstención.
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
    theme.console.print(theme.fit(table))

    _show_cases(report, verbose=verbose)
    _show_abstention(report, threshold, verbose=verbose)


def _show_cases(report: EvalReport, *, verbose: bool) -> None:
    within = 5 if 5 in report.ks else report.ks[-1]
    answerable = [r for r in report.results if r.case.answerable]
    shown = answerable if verbose else report.failures(within)
    if not shown:
        if answerable:
            theme.console.print(
                Text(
                    f"\n✔ Todos los casos aciertan en el top-{within}", style="success"
                )
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
    theme.console.print()
    theme.console.print(theme.fit(detail))


def _abstention_line(label: str, result: Abstention) -> Text:
    line = Text(f"{label}: ", style="info")
    line.append(
        f"responde {result.false_answers}/{result.unanswerable} sin respuesta",
        style="success" if result.false_answers == 0 else "error",
    )
    if result.by_name:
        line.append(f" ({result.by_name} por nombre)", style="muted")
    line.append(" · ")
    line.append(
        f"calla {result.lost}/{result.answerable} con respuesta",
        style="success" if result.lost == 0 else "warning",
    )
    return line


def _show_abstention(
    report: EvalReport, threshold: float | None, *, verbose: bool
) -> None:
    """Respuestas falsas y perdidas con el umbral actual y con el sugerido."""
    current = report.abstention(threshold)
    if not current.unanswerable:
        return
    theme.console.print()
    theme.console.print(Text("Abstención", style="title"))
    label = "desactivada" if threshold is None else f"max_distance = {threshold:g}"
    theme.console.print(_abstention_line(label, current))
    suggested = report.suggest_threshold()
    if suggested is not None and suggested != threshold:
        theme.console.print(
            _abstention_line(f"sugerido {suggested:g}", report.abstention(suggested))
        )
        theme.console.print(
            Text(
                f"  [retrieval] max_distance = {suggested:g} en config.toml "
                "(calíbralo con más casos antes de fiarte).",
                style="muted",
            )
        )
    if current.by_name:
        theme.console.print(
            Text(
                "  Las que coinciden por nombre con algún símbolo se responden con "
                "cualquier umbral.",
                style="muted",
            )
        )
    effective = threshold if suggested is None else suggested
    negatives = [r for r in report.results if not r.case.answerable]
    shown = (
        negatives if verbose else [r for r in negatives if not r.abstains(effective)]
    )
    if not shown:
        return
    detail = Table(
        title="Sin respuesta" if verbose else "Sin respuesta que se responden",
        title_style="title",
        title_justify="left",
        box=box.ROUNDED,
        border_style="muted",
        header_style=f"bold {FG}",
        expand=True,
    )
    detail.add_column("Distancia", justify="right", no_wrap=True, min_width=9)
    for name, ratio in (("Pregunta", 6), ("Primer resultado", 5)):
        detail.add_column(name, ratio=ratio, overflow="ellipsis", no_wrap=True)
    for result in shown:
        distance = (
            Text("nombre", style="error")
            if result.distance is None
            else Text(
                f"{result.distance:.3f}",
                style="success" if result.abstains(effective) else "warning",
            )
        )
        first = result.ranked[0] if result.ranked else "—"
        detail.add_row(distance, Text(result.case.question), Text(first, style="muted"))
    theme.console.print()
    theme.console.print(theme.fit(detail))


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
    theme.console.print(theme.fit(table))


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
    theme.console.print(
        theme.fit(
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
    theme.console.print(Text(f"Fichero: {path}{note}", style="muted"), soft_wrap=True)
