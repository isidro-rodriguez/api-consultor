from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.style import Style
from typer import rich_utils

from api_consultor import render
from api_consultor.citations import Citations
from api_consultor.consultor import Candidate, Retrieval
from api_consultor.evaluation import EvalCase, evaluate
from api_consultor.store import Chunk, SourceInfo
from api_consultor.versions import VersionStatus

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
    assert "─ Error ─" in text
    assert "─ Aviso ─" in text
    assert "│ Algo falló" in text
    assert "│ Cuidado" in text
    lines = text.splitlines()
    error = next(i for i, line in enumerate(lines) if "Algo falló" in line)
    assert lines[error + 1].strip("│ ") == ""  # línea en blanco antes de la pista
    assert "Prueba esto" in lines[error + 2]


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


def test_stream_markdown_returns_the_full_answer(capture: Capture) -> None:
    capture(100)
    assert render.stream_markdown(["Hola ", "mundo"]) == "Hola mundo"


def test_show_sources_marks_cited_chunks(capture: Capture) -> None:
    buffer = capture(100)
    chunks = [Chunk("a.f", "a", "1.0", "x"), Chunk("a.g", "a", "1.0", "y")]
    render.show_sources(chunks, citations=Citations((2,), ()))
    text = buffer.getvalue()
    assert "DOC-01" in text
    assert "DOC-02 ✔" in text
    assert "DOC-01 ✔" not in text
    assert "no cita" not in text


def test_show_sources_warns_about_invented_citations(capture: Capture) -> None:
    buffer = capture(100)
    render.show_sources(
        [Chunk("a.f", "a", "1.0", "x")], citations=Citations((), ("DOC-09",))
    )
    assert "inexistentes (DOC-09)" in buffer.getvalue()


def test_show_sources_warns_when_nothing_is_cited(capture: Capture) -> None:
    buffer = capture(100)
    render.show_sources([Chunk("a.f", "a", "1.0", "x")], citations=Citations((), ()))
    assert "no cita ninguna fuente" in buffer.getvalue()


def test_show_sources_links_names_when_a_link_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = StringIO()
    out = Console(
        file=buffer,
        width=100,
        theme=render.THEME,
        force_terminal=True,
        color_system="standard",
        legacy_windows=False,  # el modo heredado de Windows no emite hipervínculos
    )
    monkeypatch.setattr(render, "console", out)
    chunks = [Chunk("a.f", "a", "1", "x", "/p/a.py", 3), Chunk("a.g", "a", "1", "y")]
    render.show_sources(chunks, link=lambda c: f"file://{c.path}" if c.path else None)
    text = buffer.getvalue()
    assert text.count("\x1b]8;id=") == 1  # solo el fragmento con ubicación
    assert "file:///p/a.py" in text
    assert "Ctrl+clic" in text


def test_show_sources_without_links_has_no_hint(capture: Capture) -> None:
    buffer = capture(100)
    render.show_sources([Chunk("a.f", "a", "1", "x")])
    assert "Ctrl+clic" not in buffer.getvalue()


def test_show_user_config_marks_custom_values(capture: Capture) -> None:
    buffer = capture(100)
    rows = [("retrieval.k", "3", True), ("display.width", "90", False)]
    path = Path("/x/config.toml")
    render.show_user_config(rows, path, exists=False)
    text = buffer.getvalue()
    assert "retrieval.k" in text
    assert "config.toml" in text
    assert "por defecto" in text
    assert f"Fichero: {path} (no existe)" in text


TRICKY = "a.md#[/bold] [red]x[/red] [tool.ruff]"


def test_data_that_looks_like_markup_is_printed_literally(capture: Capture) -> None:
    """Títulos como `[tool.ruff]` no deben interpretarse como estilos de Rich."""
    buffer = capture(160)
    chunk = Chunk(TRICKY, "[red]pkg", "[1]", "x")
    candidate = Candidate(chunk, 0.5, {"N": 1}, 0.1, selected=True)
    render.show_debug(Retrieval([chunk], [candidate]), 10)
    render.show_sources_table([SourceInfo("[red]pkg", "[1]", 1)], "T")
    state = VersionStatus(SourceInfo("[red]pkg", "[1]", 1, "docs", ""), "[x]", "docs")
    render.show_status_table([state], "T")
    report = evaluate(lambda _q, _k: [chunk], [EvalCase(TRICKY, (TRICKY,))], (1,))
    render.show_evaluation(report, verbose=True)
    render.show_user_config([("[red]x", "[/y]", True)], Path("/p"), exists=True)
    text = buffer.getvalue()
    assert TRICKY in text
    assert "[/bold]" in text
    assert "[red]pkg" in text
    assert "[/y]" in text


def test_typer_help_console_knows_theme_styles() -> None:
    render.style_typer_help()
    help_console = rich_utils._get_rich_console()
    assert help_console.get_style("info") == Style.parse(render.BLUE)
    assert help_console.get_style("option") == Style.parse(rich_utils.STYLE_OPTION)


def test_style_typer_help_is_idempotent() -> None:
    render.style_typer_help()
    render.style_typer_help()
    assert rich_utils._get_rich_console is render._themed_console


def test_typer_help_paragraphs_are_reflowed() -> None:
    render.style_typer_help()
    text = rich_utils._fix_linebreaks(
        ["Una línea\n  partida.", "Otro\npárrafo."], "rich"
    )
    assert text == "Una línea partida.\n\nOtro párrafo."


def test_status_table_shows_abbreviated_environments(
    capture: Capture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    venv = tmp_path / "web" / ".venv" / "bin" / "python"
    statuses = [
        VersionStatus(SourceInfo("json", "3.13", 1, "stdlib"), "3.13", "ok"),
        VersionStatus(SourceInfo("requests", "2", 1, python=str(venv)), "2", "ok"),
        VersionStatus(SourceInfo("ajeno", "1", 1, python="/opt/py"), "1", "ok"),
        VersionStatus(SourceInfo("notas", "", 1, "docs"), None, "docs"),
    ]
    buffer = capture(120)
    render.show_status_table(statuses, "Fuentes", environment=True)
    text = buffer.getvalue()
    assert "herramienta" in text
    assert "~/web/.venv" in text
    assert str(Path("/opt/py")) in text
