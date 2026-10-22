"""Tema Monokai, consolas compartidas y utilidades de ancho."""

from rich.console import Console, RenderableType
from rich.constrain import Constrain
from rich.markdown import Markdown
from rich.theme import Theme

from ..config import SETTINGS

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

ERROR_TITLE = "Error"
"""Título del panel de errores (el mismo para los de Typer y los propios)."""

WARNING_TITLE = "Aviso"
"""Título del panel de avisos."""

console = Console(theme=THEME, highlight=False)
"""Consola de la salida normal."""

err_console = Console(theme=THEME, highlight=False, stderr=True)
"""Consola de errores y avisos (salida de error)."""


def fit(renderable: RenderableType) -> RenderableType:
    """Limita el ancho al máximo configurado (o al de la terminal, si es menor)."""
    return Constrain(renderable, SETTINGS.console_width)


def markdown(text: str) -> RenderableType:
    """Markdown con el tema de código Monokai, limitado al ancho máximo."""
    return fit(Markdown(text, code_theme="monokai"))
