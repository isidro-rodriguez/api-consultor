import tomllib
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from api_consultor import render
from api_consultor.cli import app
from api_consultor.evaluation import (
    EvalCase,
    EvalReport,
    Hits,
    evaluate,
    first_relevant_rank,
    load_cases,
    missing_sources,
    recall_at_k,
    reciprocal_rank,
)
from api_consultor.store import Chunk

from .conftest import StubOllama

runner = CliRunner()


def chunk(qualname: str) -> Chunk:
    return Chunk(qualname, "pkg", "1.0", qualname)


def test_first_relevant_rank_uses_exact_names_and_patterns() -> None:
    ranked = ["a.x", "b.y", "guia.md#Sección"]
    assert first_relevant_rank(ranked, ["b.y"]) == 2
    assert first_relevant_rank(ranked, ["guia.md#*"]) == 3
    assert first_relevant_rank(ranked, ["inexistente"]) is None


def test_alternatives_with_pipe_accept_any_of_them() -> None:
    ranked = ["x", "library/json.txt#Uso", "json.dumps"]
    expected = ["json.dumps|library/json.txt#*"]
    assert first_relevant_rank(ranked, expected) == 2
    assert recall_at_k(ranked, expected, 3) == 1.0
    assert first_relevant_rank(["x"], expected) is None


def test_missing_sources_reports_patterns_without_any_match() -> None:
    cases = [
        EvalCase("q1", ("a.b|docs/a.txt#*",)),
        EvalCase("q2", ("no.existe", "a.b")),
    ]
    missing = missing_sources(cases, ["a.b", "z"])
    assert missing == [(cases[1], "no.existe")]
    assert missing_sources(cases, []) == [
        (cases[0], "a.b|docs/a.txt#*"),
        (cases[1], "no.existe"),
        (cases[1], "a.b"),
    ]


def test_exact_names_with_brackets_are_not_treated_as_character_classes() -> None:
    assert first_relevant_rank(["a.txt#Ejemplos [2]"], ["a.txt#Ejemplos [2]"]) == 1


def test_recall_at_k_counts_expected_sources_found() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["a", "d"], 1) == 0.5
    assert recall_at_k(ranked, ["a", "d"], 4) == 1.0
    assert recall_at_k(ranked, ["z"], 4) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_evaluate_aggregates_global_and_per_category_metrics() -> None:
    cases = [
        EvalCase("q1", ("a",), "uno"),
        EvalCase("q2", ("b",), "uno"),
        EvalCase("q3", ("c",), "dos"),
    ]
    results = {"q1": ["a", "x"], "q2": ["x", "b"], "q3": ["x", "y"]}
    seen: list[str] = []
    report = evaluate(
        lambda question, k: [chunk(n) for n in results[question][:k]],
        cases,
        ks=(2, 1, 2),
        on_result=lambda result: seen.append(result.case.question),
    )
    assert seen == ["q1", "q2", "q3"]
    assert report.ks == (1, 2)
    overall = report.metrics()
    assert overall.cases == 3
    assert overall.recall == pytest.approx({1: 1 / 3, 2: 2 / 3})
    assert overall.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.metrics("uno").mrr == pytest.approx(0.75)
    assert report.categories == ["dos", "uno"]
    assert [r.case.question for r in report.failures(within=1)] == ["q2", "q3"]


def test_evaluate_passes_the_largest_k_to_retrieve() -> None:
    requested: list[int] = []

    def retrieve(_question: str, k: int) -> list[Chunk]:
        requested.append(k)
        return []

    evaluate(retrieve, [EvalCase("q", ("a",))], ks=(1, 5, 10))
    assert requested == [10]


@pytest.mark.parametrize("ks", [(), (0,), (-1, 5)])
def test_evaluate_rejects_invalid_ks(ks: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positivos"):
        evaluate(lambda _q, _k: [], [EvalCase("q", ("a",))], ks=ks)


def test_metrics_of_an_empty_category_are_zero() -> None:
    report = evaluate(lambda _q, _k: [], [EvalCase("q", ("a",))])
    assert report.metrics("no-existe").cases == 0
    assert report.metrics("no-existe").mrr == 0.0


def test_load_cases_reads_toml(tmp_path: Path) -> None:
    path = tmp_path / "casos.toml"
    path.write_text(
        '[[case]]\nquestion = "¿Qué?"\nexpected = ["a.b", "c.*"]\ncategory = "x"\n\n'
        '[[case]]\nquestion = "Otra"\nexpected = ["z"]\n',
        encoding="utf-8",
    )
    assert load_cases(path) == [
        EvalCase("¿Qué?", ("a.b", "c.*"), "x"),
        EvalCase("Otra", ("z",), "general"),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('[[case]]\nexpected = ["a"]\n', "falta 'question'"),
        ('[[case]]\nquestion = "  "\nexpected = ["a"]\n', "falta 'question'"),
        ('[[case]]\nquestion = "q"\n', "'expected'"),
        ('[[case]]\nquestion = "q"\nexpected = [1]\n', "'expected'"),
        ('[[case]]\nquestion = "q"\nexpected = ["a"]\ncategory = 3\n', "'category'"),
        ('case = "no es una lista"\n', "lista de tablas"),
    ],
)
def test_load_cases_reports_invalid_files(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "malo.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_load_cases_reports_toml_syntax_errors(tmp_path: Path) -> None:
    path = tmp_path / "roto.toml"
    path.write_text("esto no es toml", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_cases(path)


def render_report(verbose: bool = False) -> str:
    cases = [
        EvalCase("pregunta acertada", ("a",), "uno"),
        EvalCase("pregunta fallida", ("zzz",), "dos"),
    ]
    report = evaluate(lambda _q, _k: [chunk("a"), chunk("b")], cases, ks=(1, 5))
    buffer = StringIO()
    out = Console(file=buffer, width=120, theme=render.THEME, color_system=None)
    original = render.theme.console
    render.theme.console = out
    try:
        render.show_evaluation(report, verbose=verbose)
    finally:
        render.theme.console = original
    return buffer.getvalue()


def test_show_evaluation_lists_metrics_and_only_failures_by_default() -> None:
    text = render_report()
    assert "R@1" in text
    assert "R@5" in text
    assert "MRR" in text
    assert "global" in text
    assert "pregunta fallida" in text
    assert "pregunta acertada" not in text


def test_show_evaluation_verbose_lists_every_case() -> None:
    text = render_report(verbose=True)
    assert "pregunta fallida" in text
    assert "pregunta acertada" in text


def test_show_evaluation_celebrates_when_nothing_fails() -> None:
    report = evaluate(lambda _q, _k: [chunk("a")], [EvalCase("q", ("a",))], ks=(1, 5))
    buffer = StringIO()
    original = render.theme.console
    render.theme.console = Console(file=buffer, width=100, theme=render.THEME)
    try:
        render.show_evaluation(report)
    finally:
        render.theme.console = original
    assert "Todos los casos aciertan en el top-5" in buffer.getvalue()


@pytest.fixture
def indexed(isolated_cwd: Path, stub_ollama: StubOllama) -> Path:
    folder = isolated_cwd / "docs" / "notas"
    folder.mkdir(parents=True)
    (folder / "os.md").write_text(
        "# os.listdir\n\nDevuelve los nombres de las entradas de un directorio.\n",
        encoding="utf-8",
    )
    assert runner.invoke(app, ["index-docs"]).exit_code == 0
    return isolated_cwd


def write_cases(folder: Path, expected: str) -> Path:
    path = folder / "casos.toml"
    case = f'category = "doc"\nquestion = "os.listdir"\nexpected = ["{expected}"]\n'
    path.write_text(f"[[case]]\n{case}", encoding="utf-8")
    return path


def test_eval_command_reports_metrics(indexed: Path) -> None:
    result = runner.invoke(app, ["eval", str(write_cases(indexed, "os.md#*"))])
    assert result.exit_code == 0, result.output
    assert "Evaluación de la recuperación" in result.output
    assert "MRR" in result.output
    assert "Todos los casos aciertan" in result.output


def test_eval_command_lists_failures_and_supports_k(indexed: Path) -> None:
    path = write_cases(indexed, "no.existe")
    result = runner.invoke(app, ["eval", str(path), "-k", "3"])
    assert result.exit_code == 0, result.output
    assert "R@3" in result.output
    assert "Fallos" in result.output


def test_eval_command_warns_about_sources_missing_from_the_index(indexed: Path) -> None:
    result = runner.invoke(app, ["eval", str(write_cases(indexed, "no.existe"))])
    assert result.exit_code == 0, result.output
    assert "no están en el índice" in result.output
    assert "no.existe" in result.output


def test_eval_command_does_not_warn_when_sources_exist(indexed: Path) -> None:
    result = runner.invoke(app, ["eval", str(write_cases(indexed, "os.md#*"))])
    assert "no están en el índice" not in result.output


def test_failures_table_keeps_every_column_readable() -> None:
    long_question = "pregunta larguísima " * 10
    cases = [EvalCase(long_question, ("fuente.esperada.larga|otra/ruta.txt#*",))]
    report = evaluate(lambda _q, _k: [chunk("resultado.distinto")], cases, ks=(1, 5))
    buffer = StringIO()
    original = render.theme.console
    render.theme.console = Console(
        file=buffer, width=90, theme=render.THEME, color_system=None
    )
    try:
        render.show_evaluation(report)
    finally:
        render.theme.console = original
    text = buffer.getvalue()
    lines = text.splitlines()
    header = next(line for line in lines if "Pregunta" in line)
    assert "Rango" in header
    assert "Esperado" in header
    assert "Primer resultado" in header
    assert max(len(line) for line in lines) <= render.theme.SETTINGS.console_width
    assert "…" in text  # el texto largo se recorta con elipsis, no se colapsa


def test_eval_command_rejects_invalid_cases_file(indexed: Path) -> None:
    path = indexed / "malo.toml"
    path.write_text('[[case]]\nquestion = "q"\n', encoding="utf-8")
    result = runner.invoke(app, ["eval", str(path)])
    assert result.exit_code == 1
    assert "'expected'" in result.output


def test_eval_command_rejects_empty_cases_file(indexed: Path) -> None:
    path = indexed / "vacio.toml"
    path.write_text("# sin casos\n", encoding="utf-8")
    result = runner.invoke(app, ["eval", str(path)])
    assert result.exit_code == 1
    assert "[[case]]" in result.output


def test_eval_command_requires_an_index(isolated_cwd: Path) -> None:
    result = runner.invoke(app, ["eval", str(write_cases(isolated_cwd, "x"))])
    assert result.exit_code == 1
    assert "nada indexado" in result.output.lower()


def test_eval_command_rejects_missing_file() -> None:
    assert runner.invoke(app, ["eval", "no_existe.toml"]).exit_code == 2


def test_bundled_stdlib_cases_file_is_well_formed() -> None:
    cases = load_cases(Path(__file__).parents[1] / "evals" / "stdlib.toml")
    assert len(cases) >= 10
    assert {case.category for case in cases} >= {
        "exacto",
        "nombre-simple",
        "descripcion",
    }


def test_an_empty_expected_list_marks_an_unanswerable_question(tmp_path: Path) -> None:
    path = tmp_path / "casos.toml"
    path.write_text('[[case]]\nquestion = "¿json.dumps_pretty?"\nexpected = []\n')
    [case] = load_cases(path)
    assert case.expected == ()
    assert not case.answerable


def abstention_report() -> EvalReport:
    distances = {"sí-1": 0.20, "sí-2": 0.35, "no-1": 0.60, "no-2": 0.80, "no-n": None}
    cases = [
        EvalCase("sí-1", ("a",), "api"),
        EvalCase("sí-2", ("a",), "api"),
        EvalCase("no-1", (), "inexistente"),
        EvalCase("no-2", (), "inexistente"),
        EvalCase("no-n", (), "inexistente"),  # coincide por nombre: nunca calla
    ]
    return evaluate(lambda q, _k: Hits([chunk("a")], distances[q]), cases, ks=(1,))


def test_unanswerable_cases_do_not_count_in_recall_nor_failures() -> None:
    report = abstention_report()
    assert report.metrics().cases == 2
    assert report.metrics().recall[1] == 1.0
    assert report.categories == ["api"]
    assert report.failures(1) == []


def test_abstention_counts_false_and_lost_answers() -> None:
    report = abstention_report()
    off = report.abstention(None)
    assert (off.false_answers, off.unanswerable, off.lost) == (3, 3, 0)
    assert off.by_name == 1
    strict = report.abstention(0.3)
    assert (strict.false_answers, strict.lost, strict.answerable) == (1, 1, 2)
    assert strict.false_answer_rate == pytest.approx(1 / 3)
    assert strict.lost_rate == 0.5


def test_suggested_threshold_separates_both_groups_with_margin() -> None:
    report = abstention_report()
    suggested = report.suggest_threshold()
    assert suggested == pytest.approx(0.475)  # entre 0,35 (sí) y 0,60 (no)
    best = report.abstention(suggested)
    assert (best.false_answers, best.lost) == (1, 0)  # solo la de nombre se escapa


def test_no_suggestion_without_both_kinds_of_cases() -> None:
    only_positive = evaluate(
        lambda _q, _k: Hits([chunk("a")], 0.2), [EvalCase("q", ("a",))], ks=(1,)
    )
    assert only_positive.suggest_threshold() is None
    assert only_positive.abstention(0.1).unanswerable == 0
    without_distances = evaluate(
        lambda _q, _k: [chunk("a")],
        [EvalCase("q", ("a",)), EvalCase("n", ())],
        ks=(1,),
    )
    assert without_distances.suggest_threshold() is None


def test_show_evaluation_reports_abstention_and_suggests_a_threshold() -> None:
    buffer = StringIO()
    original = render.theme.console
    render.theme.console = Console(file=buffer, width=140, theme=render.THEME)
    try:
        render.show_evaluation(abstention_report(), threshold=0.3)
    finally:
        render.theme.console = original
    text = " ".join(buffer.getvalue().split())
    assert "max_distance = 0.3: responde 1/3 sin respuesta (1 por nombre)" in text
    assert "calla 1/2 con respuesta" in text
    assert "sugerido 0.475: responde 1/3" in text
    assert "max_distance = 0.475 en config.toml" in text
    assert "coinciden por nombre" in text
    assert "no-n" in text  # la que se responde aun con el umbral sugerido
    assert "no-1" not in text


def test_eval_command_measures_abstention(indexed: Path) -> None:
    path = indexed / "casos.toml"
    path.write_text(
        '[[case]]\nquestion = "os.listdir"\nexpected = ["os.md#*"]\n'
        '[[case]]\nquestion = "receta de paella valenciana"\nexpected = []\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(path), "-v"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Abstención" in flat
    assert "desactivada: responde 1/1 sin respuesta" in flat
    assert "receta de paella" in flat


def test_forced_abstentions_hold_for_any_threshold_and_skip_the_suggestion() -> None:
    cases = [
        EvalCase("sí", ("a",)),
        EvalCase("falla", ("zzz",)),  # la recuperación no da con la fuente
        EvalCase("inventada", ()),
    ]
    found = {
        "sí": Hits([chunk("a")], 0.2),
        "falla": Hits([chunk("b")], 0.9),
        "inventada": Hits([chunk("a")], 0.1, forced=True),
    }
    report = evaluate(lambda q, _k: found[q], cases, ks=(1,))
    off = report.abstention(None)
    assert (off.false_answers, off.lost) == (0, 0)
    # Callar `falla` no es perder una respuesta: no se había recuperado su fuente.
    assert report.abstention(0.5).lost == 0
    assert report.abstention(0.1).lost == 1
    assert report.suggest_threshold() is not None


def test_show_evaluation_marks_nonexistent_symbols() -> None:
    report = evaluate(
        lambda q, _k: Hits([chunk("a")], 0.2, forced=q == "x.y"),
        [EvalCase("q", ("a",)), EvalCase("x.y", ())],
        ks=(1,),
    )
    buffer = StringIO()
    original = render.theme.console
    render.theme.console = Console(file=buffer, width=120, theme=render.THEME)
    try:
        render.show_evaluation(report, verbose=True)
    finally:
        render.theme.console = original
    text = " ".join(buffer.getvalue().split())
    assert "desactivada: responde 0/1 sin respuesta" in text
    assert "inexistente" in text
