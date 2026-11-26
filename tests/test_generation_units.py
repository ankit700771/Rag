import pytest

from app.generation.gemini import GeminiGenerator, GenerationUnavailable
from app.generation.prompt import build_prompt, extract_citations
from app.retrieval.types import Hit


def sample_hits():
    return [
        Hit(1, "a.md", "A", 0, "content a", 0.9),
        Hit(2, "b.md", "B", 1, "content b", 0.8),
    ]


def test_prompt_numbers_passages_and_includes_question():
    prompt = build_prompt("What is a?", sample_hits())
    assert "[1] (source: a.md)\ncontent a" in prompt
    assert "[2] (source: b.md)\ncontent b" in prompt
    assert "Question: What is a?" in prompt


def test_extract_citations_in_order_without_duplicates():
    assert extract_citations("Yes [2] and also [1][2].", 2) == [2, 1]


def test_extract_citations_ignores_out_of_range():
    assert extract_citations("Claim [7] and [0] and [1].", 2) == [1]


def test_extract_citations_none():
    assert extract_citations("no citations here", 3) == []


def test_gemini_requires_api_key(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gemini_api_key", None)
    with pytest.raises(GenerationUnavailable):
        GeminiGenerator()


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse(self.text)


class FakeClient:
    def __init__(self, text):
        self.models = FakeModels(text)


def test_gemini_generate_passes_prompt_and_system_instruction():
    generator = GeminiGenerator(api_key="test-key", model="gemini-test")
    generator.client = FakeClient("  answer [1]  ")
    result = generator.generate("be brief", "the prompt")
    kwargs = generator.client.models.kwargs
    assert result == "answer [1]"
    assert kwargs["model"] == "gemini-test"
    assert kwargs["contents"] == "the prompt"
    assert kwargs["config"].system_instruction == "be brief"


def test_gemini_empty_response_raises():
    generator = GeminiGenerator(api_key="test-key")
    generator.client = FakeClient("")
    with pytest.raises(GenerationUnavailable):
        generator.generate("s", "p")


def test_gemini_api_error_becomes_generation_unavailable():
    from google.genai import errors

    class FailingModels:
        def generate_content(self, **kwargs):
            raise errors.ServerError(503, {"error": {"status": "UNAVAILABLE", "message": "busy"}})

    class FailingClient:
        models = FailingModels()

    generator = GeminiGenerator(api_key="test-key")
    generator.client = FailingClient()
    with pytest.raises(GenerationUnavailable, match="503"):
        generator.generate("s", "p")
