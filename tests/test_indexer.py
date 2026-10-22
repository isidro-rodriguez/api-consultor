import logging
import platform
from pathlib import Path

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


def test_api_chunks_remember_file_and_line(make_package: PackageFactory) -> None:
    package = make_package(
        "synth_loc",
        {
            "__init__.py": '"""Módulo."""\n',
            "core.py": '"""Núcleo."""\nimport os\n\n\ndef run():\n    """Ejecuta."""\n',
        },
    )
    located = {c.qualname: c for c in extract_chunks(package)}
    func = located["synth_loc.core.run"]
    assert Path(func.path).name == "core.py"
    assert func.line == 5
    assert located["synth_loc"].line == 1
    assert located["synth_loc.core"].path == func.path


PROPS = {
    "__init__.py": '''
        """Módulo."""
        from functools import cached_property


        class Box:
            """Caja."""

            @property
            def size(self):
                """Tamaño de la caja."""
                return 1

            @cached_property
            def weight(self):
                """Peso calculado una vez."""
                return 2

            @property
            def undocumented(self):
                return 3

            @property
            def _private(self):
                """No debe indexarse."""
                return 4


        class Big(Box):
            """Caja grande."""
    ''',
}


def test_properties_are_indexed_with_their_own_header(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_props", PROPS)
    found = {c.qualname: c for c in extract_chunks(package)}
    assert set(found) == {
        "synth_props",
        "synth_props.Box",
        "synth_props.Box.size",
        "synth_props.Box.weight",
        "synth_props.Big",
    }
    assert found["synth_props.Box.size"].content.startswith(
        "property synth_props.Box.size\n\nTamaño de la caja."
    )
    assert "Peso calculado una vez." in found["synth_props.Box.weight"].content


def test_properties_remember_their_getter_location(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_props_loc", PROPS)
    found = {c.qualname: c for c in extract_chunks(package)}
    size = found["synth_props_loc.Box.size"]
    weight = found["synth_props_loc.Box.weight"]
    assert Path(size.path).name == "__init__.py"
    assert size.line > found["synth_props_loc.Box"].line
    assert weight.line > size.line


INHERIT = {
    "__init__.py": '"""Paquete."""\n',
    "_base.py": '''
        class Base:
            """Base privada."""

            def shared(self):
                """Definido en la base privada."""

            def overridden(self):
                """Versión de la base."""

            @property
            def hidden_prop(self):
                """Propiedad de la base privada."""
    ''',
    "core.py": '''
        from . import _base


        class A(_base.Base):
            """Primera pública."""

            def overridden(self):
                """Versión de A."""


        class B(_base.Base):
            """Segunda pública."""


        class Pub:
            """Pública con método propio."""

            def own(self):
                """Método de Pub."""


        class C(Pub):
            """Hereda de una base pública."""
    ''',
}


def test_members_of_private_bases_are_indexed_in_the_first_public_class(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_inherit", INHERIT)
    names = {c.qualname for c in extract_chunks(package)}
    assert "synth_inherit.core.A.shared" in names
    assert "synth_inherit.core.A.hidden_prop" in names
    assert "synth_inherit.core.B.shared" not in names  # ya está en A: sin duplicar


def test_nearest_definition_wins_over_private_bases(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_inherit2", INHERIT)
    found = {c.qualname: c for c in extract_chunks(package)}
    assert "Versión de A." in found["synth_inherit2.core.A.overridden"].content
    assert (
        "Versión de la base." not in found["synth_inherit2.core.A.overridden"].content
    )


def test_methods_of_public_bases_are_not_copied_to_subclasses(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_inherit3", INHERIT)
    names = {c.qualname for c in extract_chunks(package)}
    assert "synth_inherit3.core.Pub.own" in names
    assert "synth_inherit3.core.C.own" not in names


def test_modules_without_public_members_are_still_indexed(
    make_package: PackageFactory,
) -> None:
    package = make_package("synth_empty", {"__init__.py": '"""Solo docstring."""\n'})
    assert {c.qualname for c in extract_chunks(package)} == {"synth_empty"}


def test_a_subpackage_raising_system_exit_is_skipped(
    make_package: PackageFactory, caplog: pytest.LogCaptureFixture
) -> None:
    # `pkgutil.walk_packages` dejaba escapar este `SystemExit` y abortaba el indexado.
    name = make_package(
        "salida_sub",
        {
            "__init__.py": "def raiz() -> None:\n    '''Raíz.'''\n",
            "roto/__init__.py": "raise SystemExit\n",
            "roto/hijo.py": "def hijo() -> None:\n    '''No se alcanza.'''\n",
            "sano/__init__.py": "",
            "sano/hondo.py": "def hondo() -> None:\n    '''Anidado.'''\n",
        },
    )
    with caplog.at_level(logging.WARNING):
        qualnames = {chunk.qualname for chunk in extract_chunks(name)}
    assert {"salida_sub.raiz", "salida_sub.sano.hondo.hondo"} <= qualnames
    assert not any("hijo" in q for q in qualnames)
    assert "salida_sub.roto" in caplog.text


def test_private_subpackages_are_not_imported(make_package: PackageFactory) -> None:
    name = make_package(
        "con_privado",
        {
            "__init__.py": "",
            "_interno/__init__.py": "raise RuntimeError('no debería importarse')\n",
            "publico.py": "def f() -> None:\n    '''F.'''\n",
        },
    )
    assert "con_privado.publico.f" in {c.qualname for c in extract_chunks(name)}


def test_imported_objects_are_indexed_where_they_are_defined(
    make_package: PackageFactory,
) -> None:
    # Como `rich.bar` importando `Console`: «a» se recorre antes que «z».
    name = make_package(
        "reexporta",
        {
            "__init__.py": "from .z_def import Raiz\n",
            "a_usa.py": "from .z_def import Clase, Raiz, funcion\n",
            "z_def.py": (
                "class Clase:\n    '''Clase.'''\n"
                "    def metodo(self) -> None:\n        '''Método.'''\n"
                "class Raiz:\n    '''Reexportada en la raíz.'''\n"
                "def funcion() -> None:\n    '''Función.'''\n"
            ),
        },
    )
    qualnames = {chunk.qualname for chunk in extract_chunks(name)}
    assert {
        "reexporta.z_def.Clase",
        "reexporta.z_def.Clase.metodo",
        "reexporta.z_def.funcion",
        "reexporta.Raiz",  # el alias público de la raíz sigue ganando
    } <= qualnames
    assert not any(q.startswith("reexporta.a_usa.") for q in qualnames)
    assert "reexporta.z_def.Raiz" not in qualnames
