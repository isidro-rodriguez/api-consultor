import sqlite3
from pathlib import Path

import pytest

from api_consultor.store import Chunk, SourceInfo, Store

DIM = 4


def chunk(qualname: str, package: str = "pkg", version: str = "1.0") -> Chunk:
    return Chunk(qualname, package, version, f"contenido de {qualname}")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "t.db", DIM)


def test_replace_is_idempotent(store: Store) -> None:
    chunks = [chunk("pkg.a"), chunk("pkg.b")]
    vectors = [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]
    store.replace("pkg", chunks, vectors)
    store.replace("pkg", chunks, vectors)
    assert store.sources() == [SourceInfo("pkg", "1.0", 2)]
    assert len(store.search([1.0, 0, 0, 0], 10)) == 2


def test_replace_swaps_old_content_of_the_package(store: Store) -> None:
    store.replace("pkg", [chunk("pkg.old")], [[1.0, 0, 0, 0]])
    store.replace("pkg", [chunk("pkg.new", version="2.0")], [[1.0, 0, 0, 0]])
    assert [c.qualname for c in store.search([1.0, 0, 0, 0], 10)] == ["pkg.new"]
    assert store.sources() == [SourceInfo("pkg", "2.0", 1)]


def test_failed_replace_keeps_previous_data(store: Store) -> None:
    store.replace("pkg", [chunk("pkg.a")], [[1.0, 0, 0, 0]])
    with pytest.raises(ValueError, match="zip"):
        store.replace("pkg", [chunk("pkg.x"), chunk("pkg.y")], [[1.0, 0, 0, 0]])
    assert store.sources() == [SourceInfo("pkg", "1.0", 1)]


def test_by_name_matches_exact_and_suffix(store: Store) -> None:
    store.replace(
        "pkg",
        [chunk("pkg.Path.rglob"), chunk("pkg.Path"), chunk("pkg.other.rglob_all")],
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
    )
    assert [c.qualname for c in store.by_name(["pkg.Path"])] == ["pkg.Path"]
    assert [c.qualname for c in store.by_name(["Path.rglob"])] == ["pkg.Path.rglob"]
    assert [c.qualname for c in store.by_name(["rglob"])] == ["pkg.Path.rglob"]
    assert store.by_name(["inexistente"]) == []


def test_by_name_limits_results_per_name(store: Store) -> None:
    names = [f"m{i}.get" for i in range(5)]
    store.replace("pkg", [chunk(n) for n in names], [[1.0, 0, 0, 0]] * 5)
    assert len(store.by_name(["get"], limit_per_name=2)) == 2


def test_search_returns_nearest_first(store: Store) -> None:
    store.replace(
        "pkg",
        [chunk("pkg.a"), chunk("pkg.b"), chunk("pkg.c")],
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.9, 0.1, 0, 0]],
    )
    result = store.search([1.0, 0, 0, 0], 3)
    assert [c.qualname for c in result] == ["pkg.a", "pkg.c", "pkg.b"]


def test_sources_group_by_package_and_version(store: Store) -> None:
    store.replace("b", [chunk("b.x", "b", "2")], [[1.0, 0, 0, 0]])
    store.replace(
        "a", [chunk("a.x", "a", "1"), chunk("a.y", "a", "1")], [[0, 1.0, 0, 0]] * 2
    )
    assert store.sources() == [SourceInfo("a", "1", 2), SourceInfo("b", "2", 1)]


def test_qualnames_lists_every_indexed_chunk(store: Store) -> None:
    store.replace("pkg", [chunk("pkg.a"), chunk("pkg.b")], [[1.0, 0, 0, 0]] * 2)
    assert sorted(store.qualnames()) == ["pkg.a", "pkg.b"]


def test_lexical_finds_identifiers_and_weights_names_higher(store: Store) -> None:
    store.replace(
        "docs",
        [
            Chunk(
                "a.md#Intro", "docs", "1", "lista larga: shutil.copytree entre otras"
            ),
            Chunk("shutil.copytree", "docs", "1", "def shutil.copytree(src, dst)"),
            Chunk("otro", "docs", "1", "nada relevante aquí"),
        ],
        [[1.0, 0, 0, 0]] * 3,
    )
    names = [c.qualname for c in store.lexical('"copytree"', 10)]
    assert names == ["shutil.copytree", "a.md#Intro"]


def test_lexical_keeps_underscores_phrases_and_ignores_accents(store: Store) -> None:
    store.replace(
        "docs",
        [
            Chunk("a", "docs", "1", "Counter.most_common(n) devuelve los más comunes"),
            Chunk("b", "docs", "1", "json.dumps(obj) serializa; most common words"),
            Chunk("c", "docs", "1", "El área de un polígono"),
        ],
        [[1.0, 0, 0, 0]] * 3,
    )
    assert [c.qualname for c in store.lexical('"most_common"', 5)] == ["a"]
    assert [c.qualname for c in store.lexical('"json dumps"', 5)] == ["b"]
    assert [c.qualname for c in store.lexical('"area"', 5)] == ["c"]


def test_lexical_returns_nothing_for_invalid_queries(store: Store) -> None:
    store.replace("docs", [chunk("a")], [[1.0, 0, 0, 0]])
    assert store.lexical("AND OR (", 5) == []
    assert store.lexical('"', 5) == []


def test_replace_keeps_the_lexical_index_in_sync(store: Store) -> None:
    old = Chunk("a", "pkg", "1", "palabraantigua")
    store.replace("pkg", [old], [[1.0, 0, 0, 0]])
    new = Chunk("b", "pkg", "2", "palabranueva")
    store.replace("pkg", [new], [[1.0, 0, 0, 0]])
    assert store.lexical('"palabraantigua"', 5) == []
    assert [c.qualname for c in store.lexical('"palabranueva"', 5)] == ["b"]


def test_opening_an_index_without_fts_migrates_it(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with Store(path, DIM) as first:
        first.replace(
            "pkg", [Chunk("a", "pkg", "1", "contenidoantiguo")], [[1.0, 0, 0, 0]]
        )
    legacy = sqlite3.connect(path)
    legacy.execute(
        "DROP TABLE chunks_fts"
    )  # simula un índice anterior a la búsqueda léxica
    legacy.commit()
    legacy.close()
    with Store(path, DIM) as migrated:
        assert [c.qualname for c in migrated.lexical('"contenidoantiguo"', 5)] == ["a"]
        assert migrated.sources() == [SourceInfo("pkg", "1", 1)]


def test_by_short_name_matches_last_component_ignoring_case(store: Store) -> None:
    store.replace(
        "pkg",
        [
            chunk("pkg.geo.Haversine"),
            chunk("pkg.geo.bearing"),
            chunk("a.md#os.listdir"),
        ],
        [[1.0, 0, 0, 0]] * 3,
    )
    assert [c.qualname for c in store.by_short_name(["haversine"])] == [
        "pkg.geo.Haversine"
    ]
    assert [c.qualname for c in store.by_short_name(["listdir"])] == ["a.md#os.listdir"]
    assert store.by_short_name(["geo"]) == []  # solo el último componente


def test_by_short_name_ignores_generic_names(store: Store) -> None:
    names = [f"m{i}.get" for i in range(6)]
    store.replace("pkg", [chunk(n) for n in names], [[1.0, 0, 0, 0]] * 6)
    assert store.by_short_name(["get"], max_matches=5) == []
    assert len(store.by_short_name(["get"], max_matches=6)) == 6
