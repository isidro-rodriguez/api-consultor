import sqlite3
from pathlib import Path

import pytest

from api_consultor.store import (
    FORMAT_VERSION,
    Chunk,
    IndexIntegrityError,
    SourceInfo,
    Store,
)

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


def test_open_records_index_metadata(tmp_path: Path) -> None:
    with Store(tmp_path / "m.db", DIM, "modelo-a") as store:
        meta = store.meta()
    assert meta["embed_model"] == "modelo-a"
    assert meta["embed_dim"] == str(DIM)
    assert meta["format_version"] == str(FORMAT_VERSION)
    assert meta["created_at"]


def test_reopening_with_the_same_model_keeps_the_creation_date(tmp_path: Path) -> None:
    with Store(tmp_path / "m.db", DIM, "modelo-a") as first:
        created = first.meta()["created_at"]
    with Store(tmp_path / "m.db", DIM, "modelo-a") as second:
        assert second.meta()["created_at"] == created


def test_a_different_embedding_model_is_rejected(tmp_path: Path) -> None:
    Store(tmp_path / "m.db", DIM, "modelo-a").close()
    with pytest.raises(IndexIntegrityError, match="modelo-a") as error:
        Store(tmp_path / "m.db", DIM, "modelo-b")
    assert "modelo-b" in str(error.value)
    assert error.value.hint is not None


def test_a_different_dimension_is_rejected(tmp_path: Path) -> None:
    Store(tmp_path / "m.db", DIM, "modelo-a").close()
    with pytest.raises(IndexIntegrityError, match="dimensiones"):
        Store(tmp_path / "m.db", DIM * 2, "modelo-a")


def test_opening_without_a_model_skips_the_model_check(tmp_path: Path) -> None:
    Store(tmp_path / "m.db", DIM, "modelo-a").close()
    with Store(tmp_path / "m.db", DIM) as store:
        assert store.meta()["embed_model"] == "modelo-a"


def test_an_index_without_model_adopts_the_configured_one(tmp_path: Path) -> None:
    with Store(tmp_path / "m.db", DIM) as first:
        assert "embed_model" not in first.meta()
    with Store(tmp_path / "m.db", DIM, "modelo-a") as second:
        assert second.meta()["embed_model"] == "modelo-a"


def test_a_newer_index_format_is_rejected(tmp_path: Path) -> None:
    Store(tmp_path / "m.db", DIM).close()
    raw = sqlite3.connect(tmp_path / "m.db")
    raw.execute("UPDATE meta SET value = '99' WHERE key = 'format_version'")
    raw.commit()
    raw.close()
    with pytest.raises(IndexIntegrityError, match="formato 99"):
        Store(tmp_path / "m.db", DIM)


def test_replace_validates_vector_sizes_before_changing_anything(store: Store) -> None:
    store.replace("pkg", [chunk("pkg.a")], [[1.0, 0, 0, 0]])
    with pytest.raises(IndexIntegrityError, match="3 dimensiones"):
        store.replace("pkg", [chunk("pkg.new")], [[1.0, 0, 0]])
    assert [c.qualname for c in store.search([1.0, 0, 0, 0], 5)] == ["pkg.a"]


def test_search_validates_the_query_vector_size(store: Store) -> None:
    with pytest.raises(IndexIntegrityError, match="espera 4"):
        store.search([1.0, 0], 3)


def test_sources_record_kind_and_indexing_date(store: Store) -> None:
    store.replace("pkg", [chunk("pkg.a")], [[1.0, 0, 0, 0]])
    store.replace("notas", [chunk("a.md#T", "notas")], [[0, 1.0, 0, 0]], kind="docs")
    packages, docs = store.sources()[1], store.sources()[0]
    assert (docs.package, docs.kind) == ("notas", "docs")
    assert (packages.package, packages.kind) == ("pkg", "package")
    assert packages.indexed_at.startswith("20")


def test_replacing_with_no_chunks_removes_the_source(store: Store) -> None:
    store.replace("pkg", [chunk("pkg.a")], [[1.0, 0, 0, 0]])
    store.replace("pkg", [], [])
    assert store.sources() == []


def test_a_legacy_index_gets_sources_inferred_from_its_chunks(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with Store(path, DIM, "modelo-a") as first:
        first.replace("pkg", [chunk("pkg.a", version="2.5")], [[1.0, 0, 0, 0]])
        first.replace(
            "notas", [chunk("a.md#T", "notas", "3.1")], [[0, 1.0, 0, 0]], kind="docs"
        )
    legacy = sqlite3.connect(path)
    legacy.execute("DROP TABLE sources")
    legacy.execute("DROP TABLE meta")
    legacy.commit()
    legacy.close()
    with Store(path, DIM, "modelo-b") as migrated:
        assert migrated.sources() == [
            SourceInfo("notas", "3.1", 1, kind="docs"),
            SourceInfo("pkg", "2.5", 1, kind="package"),
        ]
        assert migrated.meta()["embed_model"] == "modelo-b"


def test_search_scored_returns_distances_in_ascending_order(store: Store) -> None:
    store.replace(
        "pkg",
        [chunk("pkg.a"), chunk("pkg.b")],
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0]],
    )
    scored = store.search_scored([1.0, 0, 0, 0], 2)
    assert [c.qualname for c, _ in scored] == ["pkg.a", "pkg.b"]
    assert scored[0][1] == pytest.approx(0.0)
    assert scored[1][1] > scored[0][1]


def test_remove_deletes_the_source_from_every_index(store: Store) -> None:
    store.replace(
        "keep", [Chunk("k", "keep", "1", "textoconservado")], [[1.0, 0, 0, 0]]
    )
    store.replace("gone", [Chunk("g", "gone", "1", "textoborrado")], [[0, 1.0, 0, 0]])
    assert store.remove("gone") == 1
    assert [s.package for s in store.sources()] == ["keep"]
    assert store.lexical('"textoborrado"', 5) == []
    assert [c.qualname for c in store.search([0, 1.0, 0, 0], 5)] == ["k"]
    assert store.by_name(["g"]) == []


def test_remove_unknown_source_returns_zero(store: Store) -> None:
    assert store.remove("nada") == 0
