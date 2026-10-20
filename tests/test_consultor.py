from pathlib import Path

import pytest

from api_consultor.config import SETTINGS
from api_consultor.consultor import _IDENTIFIER, Consultor, _format_context
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


def test_retrieve_is_limited_to_top_k(consultor: Consultor) -> None:
    assert len(consultor.retrieve("función número")) == SETTINGS.top_k


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
