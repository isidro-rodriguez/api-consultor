"""Respuesta del modelo: Markdown en streaming, panel de fuentes y diagnóstico."""

from collections.abc import Callable, Collection, Iterable, Sequence

from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text

from ..citations import Citations, doc_id
from ..config import SETTINGS
from ..consultor import Retrieval
from ..records import Chunk
from . import theme
from .messages import print_warning
from .theme import FG


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
        console=theme.console,
        transient=True,
        refresh_per_second=8,
        vertical_overflow="ellipsis",
    ) as live:
        for token in tokens:
            buffer += token
            live.update(theme.markdown(buffer))
    theme.console.print()
    theme.console.print(theme.markdown(buffer))
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
    theme.console.print()
    theme.console.print(theme.fit(panel))
    if citations is not None:
        _show_citation_issues(citations)
    theme.console.print()


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
    theme.console.print()
    theme.console.print(theme.fit(summary))
    theme.console.print(theme.fit(table))
