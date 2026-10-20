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
