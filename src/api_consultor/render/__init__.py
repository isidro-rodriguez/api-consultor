"""Presentación en consola: toda la salida de la aplicación pasa por este paquete.

- `theme`: paleta Monokai, consolas compartidas y límite de ancho.
- `typer_help`: ayuda y errores de Typer en español y con el tema (API privada).
- `messages`: errores y avisos en panel, notas, entrada, esperas y versión.
- `answer`: respuesta en streaming, panel de fuentes y diagnóstico de `ask --debug`.
- `tables`: barra de progreso y tablas de fuentes, estados, evaluación y preferencias.

Las funciones leen `theme.console` y `theme.err_console` al ejecutarse: para capturar
la salida (tests) basta con sustituirlas allí.
"""

from .answer import show_debug, show_sources, stream_markdown
from .messages import (
    print_error,
    print_hint,
    print_warning,
    prompt,
    show_version,
    status,
)
from .tables import (
    progress_bar,
    show_evaluation,
    show_sources_table,
    show_status_table,
    show_user_config,
)
from .theme import THEME
from .typer_help import style_typer_help

__all__ = [
    "THEME",
    "print_error",
    "print_hint",
    "print_warning",
    "progress_bar",
    "prompt",
    "show_debug",
    "show_evaluation",
    "show_sources",
    "show_sources_table",
    "show_status_table",
    "show_user_config",
    "show_version",
    "status",
    "stream_markdown",
    "style_typer_help",
]
