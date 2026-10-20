from importlib import metadata

import api_consultor


def test_version_matches_installed_metadata() -> None:
    assert api_consultor.__version__ == metadata.version("api-consultor")
