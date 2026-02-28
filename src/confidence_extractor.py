"""
src/confidence_extractor.py
Parse verbalized confidence percentages from model responses.

Models are instructed to always end with:
    Answer: [answer] | Confidence: [0-100]%

We try multiple patterns in priority order to be robust to minor formatting
deviations, then fall back to a best-effort scan.
"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_confidence(text: str) -> float | None:
    """
    Extract verbalized confidence from a model response.

    Returns float in [0.0, 1.0] or None if extraction fails.
    Tries multiple regex patterns in priority order.
    """
    if not text:
        return None

    patterns = [
        # Primary — the exact format we instruct the model to use
        r'\|\s*[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%',   # "| Confidence: 85%"
        r'[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%',         # "Confidence: 85%"
        # Fallbacks — natural language
        r'(\d+(?:\.\d+)?)\s*%\s+confident',               # "85% confident"
        r'confidence\s+(?:is|of|level[:\s]+)\s*(\d+(?:\.\d+)?)\s*%',  # "confidence is 85%"
        r'(\d+(?:\.\d+)?)\s*(?:percent|%)\s+(?:sure|certain|confidence)',  # "85 percent sure"
        r'[Cc]onfidence[:\s]+(\d+(?:\.\d+)?)\s*(?:out of 100|/100)',   # "Confidence: 85/100"
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 100:
                return val / 100.0
            else:
                logger.warning(f"Confidence value out of range [{val}] in: {text[:120]!r}")

    return None


def extract_answer(text: str) -> str | None:
    """
    Extract the stated answer from the model response.
    Looks for 'Answer: [...] | Confidence: ...' pattern.
    Returns the answer string or None.
    """
    m = re.search(
        r'[Aa]nswer:\s*(.+?)\s*\|\s*[Cc]onfidence:',
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # Fallback: 'Answer: ...' at end of line
    m = re.search(r'[Aa]nswer:\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()

    return None


def extraction_success_rate(responses: list[str]) -> float:
    """Return fraction of responses where confidence was extractable."""
    if not responses:
        return 0.0
    successes = sum(1 for r in responses if extract_confidence(r) is not None)
    return successes / len(responses)


def assert_extraction_success(
    responses: list[str],
    model_key: str,
    threshold: float = 0.90,
) -> None:
    """
    Assert that extraction_success_rate >= threshold.
    If below, log ALL failed responses and raise RuntimeError.
    This is a hard stop — something is wrong with the prompt or parser.
    """
    rate = extraction_success_rate(responses)
    if rate >= threshold:
        logger.info(
            f"[{model_key}] Confidence extraction success rate: "
            f"{rate:.1%} ({sum(extract_confidence(r) is not None for r in responses)}/{len(responses)})"
        )
        return

    # Below threshold — collect failures and halt loudly
    failures = [
        (i, r) for i, r in enumerate(responses)
        if extract_confidence(r) is None
    ]
    logger.error(
        f"\n{'=' * 70}\n"
        f"EXTRACTION FAILURE: {model_key}\n"
        f"Success rate {rate:.1%} is below threshold {threshold:.1%}.\n"
        f"{len(failures)} / {len(responses)} responses failed extraction.\n"
        f"{'=' * 70}"
    )
    for idx, resp in failures:
        logger.error(f"\n--- Failed response #{idx} ---\n{resp}\n")

    raise RuntimeError(
        f"Confidence extraction rate for '{model_key}' is {rate:.1%}, "
        f"below required threshold {threshold:.1%}. "
        f"Inspect the logs above for all {len(failures)} failed responses. "
        f"Fix the system prompt or extraction patterns before proceeding."
    )
