"""
src/models.py
OpenRouter client and model call utilities.
"""

import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODELS = {
    "gpt52":  "openai/gpt-5.2",
    "claude": "anthropic/claude-sonnet-4-6",
    "gemini": "google/gemini-3.1-pro-preview",
}


def get_client() -> OpenAI:
    """Return OpenAI SDK client pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file: OPENROUTER_API_KEY=sk-or-..."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def call_model(
    client: OpenAI,
    model_key: str,
    messages: list[dict],
    temperature: float = 0.0,
) -> str:
    """
    Call a model via OpenRouter. Returns response text.
    Always uses temperature=0.0 for reproducibility unless explicitly overridden.
    Raises on API error after 3 retries with exponential backoff.
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model key '{model_key}'. Valid keys: {list(MODELS.keys())}"
        )

    model_string = MODELS[model_key]
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            logger.debug(
                f"Calling {model_key} ({model_string}), "
                f"attempt {attempt + 1}, temp={temperature}"
            )
            response = client.chat.completions.create(
                model=model_string,
                messages=messages,
                temperature=temperature,
                extra_headers={"X-Title": "SACD-Research"},
            )
            text = response.choices[0].message.content
            logger.debug(f"Response received ({len(text or '')} chars)")
            return text or ""
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            logger.warning(
                f"API error on attempt {attempt + 1}/3 for {model_key}: {e}. "
                f"Retrying in {wait}s..."
            )
            if attempt < 2:
                time.sleep(wait)

    raise RuntimeError(
        f"All 3 attempts failed for model '{model_key}'"
    ) from last_exc
