"""Mensajes: errores y avisos en panel, notas, entrada, esperas y versión."""

from collections.abc import Sequence

import pyfiglet
from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from ..config import SETTINGS
from . import theme
from .theme import (
    BLUE,
    ERROR_TITLE,
    FG,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    WARNING_TITLE,
    YELLOW,
)

APP_NAME = "ApiConsultor"
"""Nombre de la aplicación que se dibuja en el banner."""

TAGLINE = "Consultor de APIs Python offline con Ollama y RAG"
"""Lema bajo el banner."""

BANNER_FONT = "slant"
"""Fuente figlet del banner."""

_RAINBOW = (RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)


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
    theme.err_console.print(theme.fit(panel))


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
    theme.console.print(theme.fit(Text(message, style="muted")))


# --- Entrada ------------------------------------------------------------------


def prompt() -> str | None:
    """Lee una línea del usuario.

    Returns:
        El texto sin espacios sobrantes, o `None` si el usuario cierra la entrada
        (Ctrl+D o Ctrl+C).
    """
    try:
        return theme.console.input(Text("> ", style="prompt")).strip()
    except (EOFError, KeyboardInterrupt):
        theme.console.print()
        return None


def status(message: str) -> Status:
    """Crea un spinner para esperas cortas.

    Args:
        message: Texto que acompaña al spinner.

    Returns:
        El indicador, para usarlo como gestor de contexto.
    """
    return theme.console.status(message, spinner="dots")


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
    available = min(theme.console.width, SETTINGS.console_width) - 6  # borde + relleno
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
    theme.console.print(theme.fit(panel))
