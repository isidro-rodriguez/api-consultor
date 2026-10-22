"""Ayuda de Typer en español y con el tema de la aplicación.

Parchea API privada de Typer (`rich_utils._get_rich_console`, `_fix_linebreaks`,
`typer._click.decorators.help_option`, `HelpFormatter.write_usage`); por eso la
dependencia está acotada a `typer<0.28` y todo queda aislado en este módulo.
"""

import shutil
from collections.abc import Callable, Sequence
from functools import partial

from rich.console import Console
from typer import rich_utils
from typer._click import decorators as click_decorators
from typer._click.core import Command
from typer._click.formatting import HelpFormatter
from typer.core import TyperOption

from ..config import SETTINGS
from .theme import (
    COMMENT,
    ERROR_TITLE,
    FG,
    GREEN,
    ORANGE,
    RED,
    THEME,
    YELLOW,
)

_typer_console = rich_utils._get_rich_console  # API privada de Typer


def _configured_width() -> int:
    return SETTINGS.console_width


def _themed_console(
    stderr: bool = False, *, width: Callable[[], int] = _configured_width
) -> Console:
    """Crea la consola de la ayuda de Typer con los estilos semánticos de `THEME`.

    El ancho se calcula al crearla (no al importar), para que respete el de
    `config.toml` aunque la configuración se lea después de registrar los parches.

    Args:
        stderr: Si la consola escribe en la salida de error.
        width: Devuelve el ancho máximo configurado.

    Returns:
        La consola de Typer con `THEME` apilado sobre sus estilos.
    """
    rich_utils.MAX_WIDTH = min(width(), shutil.get_terminal_size().columns)
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


def style_typer_help(width: Callable[[], int] = _configured_width) -> None:
    """Aplica la paleta, el ancho máximo y los estilos del tema a la ayuda de Typer.

    Tras llamarla, los textos de ayuda admiten el markup semántico de `THEME`
    (`[info]`, `[muted]`, `[warning]`…) y los textos fijos de Typer salen en
    español. Es idempotente.

    Args:
        width: Devuelve el ancho máximo; se consulta cada vez que Typer pinta una
            ayuda o un error. Por defecto, `SETTINGS.console_width`.
    """
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
    setattr(rich_utils, "_get_rich_console", partial(_themed_console, width=width))  # noqa: B010 (ty: funciones nominales)
    _translate_typer()
