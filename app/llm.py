"""Google Gemini LLM module.

Sole responsibility: load environment configuration, validate the API
key, and return a ready-to-use `ChatGoogleGenerativeAI` instance. All
Gemini-specific setup lives here so the rest of the app depends only
on this module's public interface.
"""

from __future__ import annotations

import functools

from langchain_google_genai import ChatGoogleGenerativeAI

from app.logger import get_logger
from config.settings import MissingAPIKeyError, settings

logger = get_logger(__name__)


class GeminiInitializationError(RuntimeError):
    """Raised when the Gemini chat model fails to initialize."""


class GeminiGenerationError(RuntimeError):
    """Raised when a Gemini API call fails at generation time."""


@functools.lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return a cached, configured Gemini chat model instance.

    Returns:
        A `ChatGoogleGenerativeAI` instance configured from
        `config/settings.py`.

    Raises:
        MissingAPIKeyError: If GOOGLE_API_KEY is not configured.
        GeminiInitializationError: If the client fails to initialize
            for any other reason (e.g. invalid model name).
    """
    api_key = settings.require_api_key()

    try:
        logger.info("Initializing Gemini model '%s'", settings.gemini_model)
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=api_key,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
        logger.info("Gemini model '%s' initialized successfully", settings.gemini_model)
        return llm
    except MissingAPIKeyError:
        raise
    except Exception as exc:  # noqa: BLE001 — SDK raises varied exception types
        raise GeminiInitializationError(
            f"Failed to initialize Gemini model '{settings.gemini_model}': {exc}"
        ) from exc


def generate_answer(messages: list[dict[str, str]]) -> str:
    """Send formatted messages to Gemini and return the generated answer.

    Args:
        messages: Role/content message dicts, typically produced by
            `app.prompts.build_rag_prompt`.

    Returns:
        The generated answer text.

    Raises:
        MissingAPIKeyError: If GOOGLE_API_KEY is not configured.
        GeminiGenerationError: If the Gemini API call fails (network
            issue, invalid key, rate limit, timeout, etc.).
    """
    llm = get_llm()
    try:
        response = llm.invoke(messages)
        return response.content
    except MissingAPIKeyError:
        raise
    except Exception as exc:  # noqa: BLE001 — SDK raises varied exception types
        logger.error("Gemini generation failed: %s", exc)
        raise GeminiGenerationError(
            f"Gemini failed to generate a response: {exc}. This may be due "
            f"to a network issue, an invalid API key, or a rate limit — "
            f"please try again."
        ) from exc
