"""Banco de regresión de la recuperación sobre un corpus sintético.

Usa embeddings sintéticos (bolsa de palabras hasheada), así que mide la lógica de
recuperación —búsqueda exacta, mezcla y deduplicación—, no la calidad semántica de
`bge-m3`. Para eso está `api eval` sobre el índice real. Los umbrales son suelos de
no regresión: súbelos cuando una mejora los supere con holgura.
"""

from pathlib import Path

import pytest

from api_consultor.consultor import Consultor
from api_consultor.docs import extract_doc_chunks
from api_consultor.evaluation import EvalReport, evaluate, load_cases
from api_consultor.indexer import extract_chunks
from api_consultor.store import Store

from .conftest import EMBED_DIM, PackageFactory, StubOllama, fake_embed

GEOLIB = {
    "__init__.py": '"""Geolib: utilidades geométricas y de coordenadas."""\n',
    "shapes.py": '''
        class Polygon:
            """Polígono definido por una lista de vértices."""

            def area(self) -> float:
                """Calcula el área del polígono con la fórmula del cordón."""

            def perimeter(self) -> float:
                """Calcula el perímetro sumando la longitud de todos los lados."""

            def centroid(self) -> tuple[float, float]:
                """Devuelve el centroide, el punto medio geométrico del polígono."""

        class Circle:
            """Círculo definido por su centro y su radio."""

            def area(self) -> float:
                """Calcula el área del círculo a partir del radio."""

            def circumference(self) -> float:
                """Calcula la circunferencia del círculo."""
        ''',
    "geoio.py": '''
        def read_geojson(path: str) -> dict:
            """Lee un fichero GeoJSON y devuelve su contenido como diccionario."""

        def write_geojson(data: dict, path: str) -> None:
            """Escribe un diccionario como fichero GeoJSON en disco."""
        ''',
    "projection.py": '''
        class Mercator:
            """Proyección Mercator, habitual en mapas web."""

            def forward(self, lat: float, lon: float) -> tuple[float, float]:
                """Convierte latitud y longitud en coordenadas planas."""

            def inverse(self, x: float, y: float) -> tuple[float, float]:
                """Convierte coordenadas planas en latitud y longitud."""

        def project_point(lat: float, lon: float, crs: str = "EPSG:3857") -> tuple:
            """Proyecta un punto geográfico al sistema de referencia indicado."""
        ''',
    "distance.py": '''
        def haversine(a: tuple, b: tuple) -> float:
            """Calcula la distancia sobre la esfera terrestre entre dos coordenadas."""

        def bearing(a: tuple, b: tuple) -> float:
            """Calcula el rumbo inicial en grados entre dos coordenadas."""
        ''',
}

GUIDE = """\
# Guía de geolib

Introducción general a la biblioteca.

## Elegir proyección

Para un mapa web conviene la proyección Mercator. Para medir áreas, una proyección
equivalente.

## Formatos de entrada y salida

Geolib lee y escribe ficheros GeoJSON y también CSV con columnas de coordenadas.
"""

CASES = Path(__file__).parent / "data" / "geolib_cases.toml"


@pytest.fixture
def corpus(
    make_package: PackageFactory, tmp_path: Path, stub_ollama: StubOllama
) -> Consultor:
    chunks = extract_chunks(make_package("geolib", GEOLIB))
    docs = extract_doc_chunks("guia-1.0", [("guia.md", GUIDE)])
    store = Store(tmp_path / "bench.db", EMBED_DIM)
    store.replace("geolib", chunks, fake_embed([c.content for c in chunks]))
    store.replace("guia-1.0", docs, fake_embed([c.content for c in docs]))
    return Consultor(store)


@pytest.fixture
def report(corpus: Consultor) -> EvalReport:
    return evaluate(corpus.retrieve, load_cases(CASES))


def test_corpus_contains_expected_sources(corpus: Consultor) -> None:
    names = {c.qualname for c in corpus.retrieve("Polygon.area", k=50)}
    assert "geolib.shapes.Polygon.area" in names
    assert "guia.md#Elegir proyección" in names


def test_global_retrieval_quality_does_not_regress(report: EvalReport) -> None:
    metrics = report.metrics()
    assert metrics.cases == 9
    assert metrics.recall[5] >= 0.9
    assert metrics.recall[10] >= 0.9
    assert metrics.mrr >= 0.8


@pytest.mark.parametrize(
    ("category", "minimum_mrr"),
    [
        ("exacto", 1.0),
        ("cualificado-parcial", 1.0),
        ("snake_case", 1.0),
        ("nombre-simple", 0.5),
        ("descripcion", 0.5),
        ("documentacion", 0.5),
    ],
)
def test_per_category_floor(
    report: EvalReport, category: str, minimum_mrr: float
) -> None:
    assert report.metrics(category).mrr >= minimum_mrr
