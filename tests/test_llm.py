from types import SimpleNamespace

import pytest

from api_consultor import llm
from api_consultor.config import SETTINGS


def test_embed_returns_lists_of_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(embeddings=[(1.0, 2.0), (3.0, 4.0)])

    monkeypatch.setattr(llm.ollama, "embed", fake)
    assert llm.embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls[0]["model"] == SETTINGS.embed_model
    assert calls[0]["input"] == ["a", "b"]


def test_stream_chat_disables_thinking_and_tolerates_empty_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake(**kwargs: object) -> list[SimpleNamespace]:
        calls.append(kwargs)
        return [
            SimpleNamespace(message=SimpleNamespace(content="Hola")),
            SimpleNamespace(message=SimpleNamespace(content=None)),
            SimpleNamespace(message=SimpleNamespace(content=" mundo")),
        ]

    monkeypatch.setattr(llm.ollama, "chat", fake)
    assert "".join(llm.stream_chat([{"role": "user", "content": "x"}])) == "Hola mundo"
    assert calls[0]["think"] is False
    assert calls[0]["stream"] is True
    assert calls[0]["model"] == SETTINGS.chat_model


def test_complete_replaces_the_system_prompt_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="copy a tree"))

    monkeypatch.setattr(llm.ollama, "chat", fake)
    assert llm.complete("sistema", "copiar un árbol") == "copy a tree"
    call = calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": "copiar un árbol"},
    ]
    assert call["stream"] is False
    assert call["think"] is False
    assert call["options"] == {"temperature": 0, "num_predict": 96}

    monkeypatch.setattr(
        llm.ollama,
        "chat",
        lambda **_kw: SimpleNamespace(message=SimpleNamespace(content=None)),
    )
    assert llm.complete("s", "p") == ""
