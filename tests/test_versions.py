import platform
from pathlib import Path

import pytest

from api_consultor.store import SourceInfo
from api_consultor.versions import check_versions, installed_version

from .conftest import FakeDistribution


@pytest.fixture
def yamlish(fake_distribution: FakeDistribution) -> None:
    """Distribución `YamlLib` 2.0 que expone el módulo importable `yamlish`."""
    fake_distribution("YamlLib", "2.0", top_level="yamlish")


def test_stdlib_modules_report_the_python_version() -> None:
    assert installed_version("json") == platform.python_version()


def test_import_name_is_resolved_through_its_distribution(yamlish: None) -> None:
    assert installed_version("yamlish") == "2.0"


def test_distribution_without_top_level_falls_back_to_its_own_name(
    fake_distribution: FakeDistribution,
) -> None:
    fake_distribution("plainpkg", "0.3")
    assert installed_version("plainpkg") == "0.3"


def test_unknown_packages_return_none() -> None:
    assert installed_version("paquete_que_no_existe_xyz") is None


def test_explicit_mapping_is_used_when_given() -> None:
    assert installed_version("nada", {"nada": ["tampoco_existe_xyz"]}) is None


@pytest.mark.parametrize(
    ("indexed", "state"),
    [("2.0", "ok"), ("1.0", "stale"), ("desconocida", "unknown")],
)
def test_check_versions_compares_indexed_and_installed(
    yamlish: None, indexed: str, state: str
) -> None:
    [status] = check_versions([SourceInfo("yamlish", indexed, 3)])
    assert status.state == state
    assert status.installed == "2.0"
    assert status.is_stale is (state == "stale")


def test_check_versions_detects_missing_and_docs_sources(yamlish: None) -> None:
    sources = [
        SourceInfo("yamlish", "2.0", 1),
        SourceInfo("desinstalado_xyz", "1.0", 1),
        SourceInfo("python-3.15-docs-text", "3.15", 9, kind="docs"),
    ]
    states = {s.source.package: s.state for s in check_versions(sources)}
    assert states == {
        "yamlish": "ok",
        "desinstalado_xyz": "missing",
        "python-3.15-docs-text": "docs",
    }


def test_check_versions_without_sources_is_empty() -> None:
    assert check_versions([]) == []


def test_fake_distribution_fixture_creates_metadata(
    fake_distribution: FakeDistribution, tmp_path: Path
) -> None:
    fake_distribution("Otra", "1.2", top_level="otra")
    assert (tmp_path / "site" / "Otra-1.2.dist-info" / "METADATA").exists()
