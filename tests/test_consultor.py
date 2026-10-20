from pathlib import Path

import pytest

from api_consultor.config import SETTINGS
from api_consultor.consultor import (
    _IDENTIFIER,
    Consultor,
    _format_context,
    _fuse,
    _lexical_query,
    _terms,
)
from api_consultor.store import Chunk, Store

from .conftest import EMBED_DIM, StubOllama, fake_embed


@pytest.fixture
def consultor(tmp_path: Path, stub_ollama: StubOllama) -> Consultor:
    chunks = [
        Chunk(f"lib.f{i}", "lib", "1.0", f"def lib.f{i}()\n\nfunción número {i}")
        for i in range(8)
    ]
    chunks.append(Chunk("lib.Path.rglob", "lib", "1.0", "rglob busca recursivamente"))
    store = Store(tmp_path / "c.db", EMBED_DIM)
    store.replace("lib", chunks, fake_embed([c.content for c in chunks]))
    return Consultor(store)


def test_identifier_regex_finds_qualified_and_snake_case_names() -> None:
    found = _IDENTIFIER.findall("uso pathlib.Path.rglob y my_func, no Path")
    assert found == ["pathlib.Path.rglob", "my_func"]


def test_format_context_without_chunks_is_empty_tag() -> None:
    assert _format_context([]) == "<docs/>"


def test_format_context_includes_source_package_and_version() -> None:
    text = _format_context([Chunk("lib.f", "lib", "1.0", "def lib.f()")])
    assert '<docs source="lib.f" package="lib" version="1.0">' in text
    assert text.endswith("</docs>")


def test_retrieve_puts_exact_matches_first_without_duplicates(
    consultor: Consultor,
) -> None:
    chunks = consultor.retrieve("¿qué hace lib.Path.rglob?")
    names = [chunk.qualname for chunk in chunks]
    assert names[0] == "lib.Path.rglob"
    assert len(names) == len(set(names))


def test_retrieve_is_limited_to_retrieval_k(consultor: Consultor) -> None:
    assert len(consultor.retrieve("función número")) == SETTINGS.retrieval_k


def test_stream_sends_context_and_question(
    consultor: Consultor, stub_ollama: StubOllama
) -> None:
    chunks = consultor.retrieve("función número 3")
    answer = "".join(consultor.stream("función número 3", chunks))
    assert answer == "Respuesta de prueba."
    content = stub_ollama.messages[0][0]["content"]
    assert "<docs source=" in content
    assert content.endswith("Pregunta: función número 3")


def test_retrieve_accepts_a_custom_limit(consultor: Consultor) -> None:
    assert len(consultor.retrieve("función número", k=8)) == 8
    assert len(consultor.retrieve("función número", k=2)) == 2


def test_terms_drop_stopwords_short_words_and_repeats() -> None:
    assert _terms("¿Cómo uso copytree para copiar copytree? Es un ok") == [
        "copytree",
        "copiar",
    ]


def test_lexical_query_builds_phrases_for_dotted_names() -> None:
    assert (
        _lexical_query("¿Qué hace json.dumps?") == '"json dumps" OR "json" OR "dumps"'
    )
    assert _lexical_query("most_common") == '"most_common"'
    assert _lexical_query("¿Qué es?") is None


def test_fuse_rewards_agreement_and_respects_weights() -> None:
    a, b, c = (Chunk(n, "p", "1", n) for n in "abc")
    fused = _fuse([(3.0, [a]), (1.0, [b, c]), (1.0, [c])])
    assert [x.qualname for x in fused] == ["a", "c", "b"]


def test_fuse_keeps_same_qualname_from_different_packages_apart() -> None:
    one = Chunk("x.f", "uno", "1", "")
    two = Chunk("x.f", "dos", "1", "")
    assert len(_fuse([(1.0, [one]), (1.0, [two])])) == 2


@pytest.fixture
def mixed(tmp_path: Path, stub_ollama: StubOllama) -> Consultor:
    chunks = [
        Chunk("geo.distance.haversine", "geo", "1", "def haversine(a, b)\n\ndistancia"),
        Chunk("geo.distance.bearing", "geo", "1", "def bearing(a, b)\n\nrumbo inicial"),
        Chunk(
            "guia.md#Recorte",
            "guia",
            "desconocida",
            "guia.md#Recorte\n\nPara recortar capas usa clip_to_bbox(capa, caja).",
        ),
        *(
            Chunk(f"guia.md#Tema {i}", "guia", "desconocida", f"texto de relleno {i}")
            for i in range(40)
        ),
    ]
    store = Store(tmp_path / "m.db", EMBED_DIM)
    store.replace("mixed", chunks, fake_embed([c.content for c in chunks]))
    return Consultor(store)


def test_retrieve_finds_a_symbol_by_its_short_name(mixed: Consultor) -> None:
    first = mixed.retrieve("¿para qué sirve haversine?")[0]
    assert first.qualname == "geo.distance.haversine"


def test_retrieve_finds_identifiers_inside_documentation(mixed: Consultor) -> None:
    first = mixed.retrieve("¿cómo funciona clip_to_bbox?")[0]
    assert first.qualname == "guia.md#Recorte"


def test_retrieve_can_return_more_candidates_than_retrieval_k(mixed: Consultor) -> None:
    assert len(mixed.retrieve("relleno", k=40)) == 40
