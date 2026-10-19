"""Renderizado de respuestas en consola con Rich."""

from collections.abc import Iterable

from rich.console import Console, RenderableType
from rich.constrain import Constrain
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner

from .config import SETTINGS

console = Console()


def _markdown(text: str) -> RenderableType:
    """Markdown limitado al ancho máximo (o al de la terminal, si es menor)."""
    return Constrain(Markdown(text), SETTINGS.console_width)


def stream_markdown(tokens: Iterable[str]) -> None:
    """Muestra la respuesta mientras llega y la deja renderizada como Markdown.

    Durante el streaming se redibuja en vivo (con un spinner hasta el primer token);
    al terminar se imprime una sola vez completa, aunque supere el alto de la terminal.
    """
    buffer = ""
    with Live(
        Spinner("dots", text="Consultando..."),
        console=console,
        transient=True,
        refresh_per_second=8,
        vertical_overflow="ellipsis",
    ) as live:
        for token in tokens:
            buffer += token
            live.update(_markdown(buffer))
    console.print(_markdown(buffer))
