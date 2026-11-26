from typing import Protocol

from app.config import settings


class GenerationUnavailable(RuntimeError):
    pass


class Generator(Protocol):
    def generate(self, system_instruction: str, prompt: str) -> str: ...


class GeminiGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise GenerationUnavailable("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=settings.gemini_timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=settings.gemini_retry_attempts),
            ),
        )

    def generate(self, system_instruction: str, prompt: str) -> str:
        from google.genai import errors, types

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction, temperature=0.2
                ),
            )
        except errors.APIError as exc:
            raise GenerationUnavailable(f"Gemini request failed: {exc.code} {exc.status}") from exc
        text = response.text
        if not text:
            raise GenerationUnavailable("Gemini returned an empty response")
        return text.strip()


def get_generator() -> Generator:
    return GeminiGenerator()
