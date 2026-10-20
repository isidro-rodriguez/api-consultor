from collections.abc import Callable
from io import StringIO

import pytest
from rich.console import Console

from api_consultor import render
from api_consultor.store import Chunk, SourceInfo

type Capture = Callable[[int], StringIO]


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Capture:
    """Sustituye las consolas de render por otras que escriben en memoria."""

    def factory(width: int = 200) -> StringIO:
        buffer = StringIO()
        out = Console(file=buffer, width=width, theme=render.THEME, color_system=None)
        monkeypatch.setattr(render, "console", out)
        monkeypatch.setattr(render, "err_console", out)
        return buffer

    return factory


LONG = "palabra " * 80


def test_markdown_never_exceeds_max_width(capture: Capture) -> None:
    buffer = capture(200)
    render.stream_markdown([f"## Título\n\n{LONG}\n\n```python\n{'x = 1  ' * 30}\n```"])
    lines = buffer.getvalue().splitlines()
    assert max(len(line) for line in lines) <= render.SETTINGS.console_width


def test_markdown_adapts_to_narrow_terminals(capture: Capture) -> None:
    buffer = capture(50)
    render.stream_markdown([LONG])
    assert max(len(line) for line in buffer.getvalue().splitlines()) <= 50


def test_stream_markdown_prints_the_whole_answer(capture: Capture) -> None:
    buffer = capture(100)
    render.stream_markdown(["Hola ", "**mundo**"])
    assert "Hola mundo" in buffer.getvalue()


def test_show_sources_hides_unknown_or_duplicated_versions(capture: Capture) -> None:
    buffer = capture(100)
    render.show_sources(
        [
            Chunk("pathlib.Path", "pathlib", "3.12.3", ""),
            Chunk("a.md#T", "mis-notas", "desconocida", ""),
            Chunk("os.txt#T", "python-3.15-docs-text", "3.15", ""),
        ]
    )
    text = buffer.getvalue()
    assert "pathlib v3.12.3" in text
    assert "vdesconocida" not in text
    assert "python-3.15-docs-text v3.15" not in text


def test_show_sources_without_chunks_prints_nothing(capture: Capture) -> None:
    buffer = capture(100)
    render.show_sources([])
    assert buffer.getvalue() == ""


def test_sources_table_formats_thousands(capture: Capture) -> None:
    buffer = capture(100)
    render.show_sources_table([SourceInfo("python-3.15-docs-text", "3.15", 9712)], "T")
    assert "9,712" in buffer.getvalue()


def test_errors_and_warnings_include_hint(capture: Capture) -> None:
    buffer = capture(100)
    render.print_error("Algo falló", "Prueba esto")
    render.print_warning("Cuidado")
    text = buffer.getvalue()
    assert "✖ Algo falló" in text
    assert "Prueba esto" in text
    assert "! Cuidado" in text


def test_show_version_renders_banner_version_and_details(capture: Capture) -> None:
    buffer = capture(200)
    render.show_version("1.2.3", [("Python", "3.12.0")])
    text = buffer.getvalue()
    assert "v1.2.3" in text
    assert "Python" in text
    assert "/_/" in text  # arte figlet
    assert max(len(line) for line in text.splitlines()) <= render.SETTINGS.console_width


def test_show_version_falls_back_to_plain_name_in_narrow_terminals(
    capture: Capture,
) -> None:
    buffer = capture(40)
    render.show_version("1.2.3", [])
    text = buffer.getvalue()
    assert "ApiConsultor" in text
    assert "/_/" not in text
