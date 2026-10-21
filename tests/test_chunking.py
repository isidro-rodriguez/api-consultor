import pytest

from api_consultor.indexer import MAX_CHARS, doc_parts, extract_chunks
from api_consultor.store import Chunk

from .conftest import PackageFactory

NUMPY_DOC = (
    "Calcula algo importante.\n\n"
    "Parameters\n----------\n"
    + "\n".join(f"p{i} : int\n    Descripción del parámetro {i}." for i in range(80))
    + "\n\nReturns\n-------\nint\n    El resultado final.\n\n"
    "Raises\n------\nValueError\n    Si algo falla.\n\n"
    "Examples\n--------\n>>> f(1)\n42\n"
)


def test_short_docstring_is_a_single_part() -> None:
    assert doc_parts("Una línea.\n\nOtro párrafo.", 500) == [
        "Una línea.\n\nOtro párrafo."
    ]


def test_long_docstring_is_split_without_losing_text() -> None:
    parts = doc_parts(NUMPY_DOC, 800)
    assert len(parts) > 1
    assert all(len(part) <= 800 for part in parts)
    joined = "\n".join(parts)
    for fragment in (
        "p0 :",
        "p79 :",
        "Returns",
        "El resultado final.",
        "Raises",
        "Examples",
        "42",
    ):
        assert fragment in joined


def test_sections_are_kept_together_when_they_fit() -> None:
    parts = doc_parts(NUMPY_DOC, 800)
    tail = next(part for part in parts if "Returns" in part)
    assert "El resultado final." in tail
    assert "Raises" in tail or "Raises" in parts[parts.index(tail) + 1]


def test_oversized_section_keeps_its_title_in_the_first_piece() -> None:
    parts = doc_parts(NUMPY_DOC, 300)
    first = next(part for part in parts if "Parameters" in part)
    assert first.startswith("Parameters")


@pytest.fixture
def long_api(make_package: PackageFactory) -> str:
    code = f'''
        def big(a, b=1):
            """{NUMPY_DOC.replace(chr(10), chr(10) + "            ")}"""

        def small():
            """Corta."""
        '''
    return make_package("synth_long", {"__init__.py": code})


def test_long_docstrings_become_several_chunks_with_unique_labels(
    long_api: str,
) -> None:
    chunks = [
        c for c in extract_chunks(long_api) if c.qualname.startswith("synth_long.big")
    ]
    labels = [c.qualname for c in chunks]
    assert labels[0] == "synth_long.big"
    assert labels[1] == "synth_long.big [2]"
    assert len(set(labels)) == len(labels) > 1


def test_no_chunk_exceeds_the_size_limit_and_nothing_is_lost(long_api: str) -> None:
    chunks = [
        c for c in extract_chunks(long_api) if c.qualname.startswith("synth_long.big")
    ]
    assert all(len(c.content) <= MAX_CHARS for c in chunks)
    text = "\n".join(c.content for c in chunks)
    assert "Examples" in text
    assert "p79 : int" in text
    assert "El resultado final." in text


def test_continuations_repeat_the_signature(long_api: str) -> None:
    chunks = [
        c for c in extract_chunks(long_api) if c.qualname.startswith("synth_long.big")
    ]
    assert all("def synth_long.big(a, b=1)" in c.content for c in chunks)
    assert "(continuación 2)" in chunks[1].content


def big_chunks(package: str) -> list[Chunk]:
    return [
        c for c in extract_chunks(package) if c.qualname.startswith("synth_long.big")
    ]


def test_short_symbols_keep_a_single_chunk(long_api: str) -> None:
    small = [c for c in extract_chunks(long_api) if "small" in c.qualname]
    assert [c.qualname for c in small] == ["synth_long.small"]
