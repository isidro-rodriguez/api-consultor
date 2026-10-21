"""Fixtures comunes: aislamiento del sistema de ficheros y Ollama simulado."""

import dataclasses
import hashlib
import importlib
import math
import re
import sys
import textwrap
from collections.abc import Callable, Generator, Iterator, Sequence
from pathlib import Path

import pytest

from api_consultor.config import SETTINGS

EMBED_DIM = 1024  # coincide con Settings.embed_dim (bge-m3)

type PackageFactory = Callable[[str, dict[str, str]], str]
type FakeDistribution = Callable[..., None]


def hash_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Embedding sintético: bolsa de palabras hasheada y normalizada (sin Ollama)."""
    vector = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def fake_embed(texts: Sequence[str]) -> list[list[float]]:
    return [hash_embedding(text) for text in texts]


class StubOllama:
    """Registro de llamadas al modelo de chat simulado."""

    def __init__(self) -> None:
        self.tokens: list[str] = ["Respuesta ", "de prueba."]
        self.messages: list[list[dict[str, str]]] = []

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.messages.append(messages)
        yield from self.tokens


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Cada test corre en un directorio temporal: nunca toca data/ ni docs/ reales."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def restore_settings(tmp_path: Path) -> Generator[None]:
    """El índice de cada test va a un directorio temporal, nunca al del usuario.

    `config.toml` también modifica `SETTINGS` al arrancar la CLI: se restaura al acabar.
    """
    snapshot = dataclasses.replace(SETTINGS)
    SETTINGS.db_path = tmp_path / "data" / "consultor.db"
    SETTINGS.docs_dir = tmp_path / "docs"
    yield
    for item in dataclasses.fields(snapshot):
        setattr(SETTINGS, item.name, getattr(snapshot, item.name))


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Nunca se lee el `config.toml` real del usuario; por defecto no existe."""
    path = tmp_path / "user-config" / "config.toml"
    monkeypatch.setenv("API_CONSULTOR_CONFIG", str(path))
    return path


@pytest.fixture(autouse=True)
def block_real_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Red de seguridad: ningún test puede hablar con un Ollama real."""

    def blocked(*_args: object, **_kwargs: object) -> None:
        pytest.fail("El test intentó llamar a Ollama; usa la fixture stub_ollama.")

    monkeypatch.setattr("api_consultor.llm.ollama.embed", blocked)
    monkeypatch.setattr("api_consultor.llm.ollama.chat", blocked)


@pytest.fixture
def stub_ollama(monkeypatch: pytest.MonkeyPatch) -> StubOllama:
    """Sustituye embeddings y generación por versiones deterministas y locales."""
    stub = StubOllama()
    monkeypatch.setattr("api_consultor.cli.embed", fake_embed)
    monkeypatch.setattr("api_consultor.consultor.embed", fake_embed)
    monkeypatch.setattr("api_consultor.consultor.stream_chat", stub.stream_chat)
    return stub


@pytest.fixture
def make_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[PackageFactory]:
    """Crea paquetes importables temporales y limpia `sys.modules` al terminar."""
    root = tmp_path / "pkgs"
    created: list[str] = []

    def factory(name: str, files: dict[str, str]) -> str:
        for relative, code in files.items():
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(code), encoding="utf-8")
        monkeypatch.syspath_prepend(str(root))
        created.append(name)
        return name

    yield factory
    for module in [m for m in sys.modules if m.split(".")[0] in created]:
        del sys.modules[module]


@pytest.fixture
def fake_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FakeDistribution:
    """Instala distribuciones falsas (solo metadatos) en un directorio temporal."""
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))

    def factory(name: str, version: str, top_level: str | None = None) -> None:
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
        if top_level is not None:
            (info / "top_level.txt").write_text(f"{top_level}\n", encoding="utf-8")
        importlib.invalidate_caches()

    return factory
