import logging
import platform

import pytest

from api_consultor.indexer import extract_chunks

from .conftest import FakeDistribution, PackageFactory

FILES = {
    "__init__.py": '''
        """Paquete de prueba."""
        from ._core import Widget, helper

        def public_func(a: int, b: str = "x") -> str:
            """Hace algo público."""
            return b * a

        def undocumented(a):
            return a

        def _private():
            """No debe indexarse."""
        ''',
    "_core.py": '''
        def helper() -> None:
            """Ayudante reexportado por el paquete."""

        class Widget:
            """Un widget."""

            def render(self, width: int) -> str:
                """Dibuja el widget."""
                return ""

            def _hidden(self) -> None:
                """Privado."""
        ''',
    "broken.py": "raise ImportError('roto a propósito')\n",
}


def qualnames(package: str) -> set[str]:
    return {chunk.qualname for chunk in extract_chunks(package)}


def test_extracts_public_documented_symbols(make_package: PackageFactory) -> None:
    names = qualnames(make_package("synth_api", FILES))
    assert {
        "synth_api",
        "synth_api.public_func",
        "synth_api.helper",
        "synth_api.Widget",
        "synth_api.Widget.render",
    } <= names


def test_skips_private_and_undocumented(make_package: PackageFactory) -> None:
    names = qualnames(make_package("synth_skip", FILES))
    assert not any("_private" in name or "_hidden" in name for name in names)
    assert "synth_skip.undocumented" not in names


def test_reexports_are_indexed_under_the_public_path(
    make_package: PackageFactory,
) -> None:
    names = qualnames(make_package("synth_reexp", FILES))
    assert "synth_reexp.helper" in names
    assert "synth_reexp._core.helper" not in names


def test_chunk_content_has_signature_and_docstring(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_sig", FILES)
    chunk = next(
        c for c in extract_chunks(package) if c.qualname.endswith("public_func")
    )
    assert "def synth_sig.public_func(a: int, b: str = 'x') -> str" in chunk.content
    assert "Hace algo público." in chunk.content


def test_broken_submodule_does_not_abort_extraction(
    make_package: PackageFactory, caplog: pytest.LogCaptureFixture
) -> None:
    package = make_package("synth_broken", FILES)
    with caplog.at_level(logging.WARNING):
        names = qualnames(package)
    assert "synth_broken.public_func" in names
    assert "synth_broken.broken" in caplog.text


def test_missing_package_raises() -> None:
    with pytest.raises(ModuleNotFoundError):
        extract_chunks("paquete_que_no_existe_xyz")


def test_uninstalled_package_has_unknown_version(make_package: PackageFactory) -> None:
    chunks = extract_chunks(make_package("synth_ver", FILES))
    assert {chunk.version for chunk in chunks} == {"desconocida"}


def test_stdlib_version_is_the_python_version() -> None:
    chunks = extract_chunks("json")
    assert chunks
    assert {chunk.version for chunk in chunks} == {platform.python_version()}


def test_version_comes_from_the_distribution_not_the_import_name(
    make_package: PackageFactory, fake_distribution: FakeDistribution
) -> None:
    package = make_package("yamlish", FILES)
    fake_distribution("YamlLib", "2.0", top_level="yamlish")
    assert {chunk.version for chunk in extract_chunks(package)} == {"2.0"}
