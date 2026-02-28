"""
src/utils.py
Caching, logging setup, and retry utilities.

The cache is SACRED — never delete data/cache/, never overwrite entries.
Every API response is persisted before any downstream processing.
Cache key = SHA-256 of (model_key + messages + temperature).
This makes the entire experiment resumable from any crash point.
"""

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve relative to the repo root (parent of src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
CACHE_DIR = os.path.join(_REPO_ROOT, "data", "cache")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def get_cache_key(model_key: str, messages: list, temperature: float) -> str:
    """Deterministic SHA-256 key from model + messages + temperature."""
    payload = json.dumps(
        {"model": model_key, "messages": messages, "temp": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_cache(key: str) -> str | None:
    """Return cached response string, or None if not cached."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)["response"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt cache file {path}: {e} — skipping cache entry")
    return None


def save_cache(key: str, response: str) -> None:
    """Persist a response to disk. NEVER overwrites an existing entry."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):  # sacred — never overwrite
        with open(path, "w") as f:
            json.dump({"response": response}, f)
        logger.debug(f"Cache saved: {key[:8]}...")
    else:
        logger.debug(f"Cache already exists, skipping write: {key[:8]}...")


# Counters for reporting in dry-run summaries
_cache_hits = 0
_cache_misses = 0


def get_cache_stats() -> dict:
    return {"hits": _cache_hits, "misses": _cache_misses}


def reset_cache_stats() -> None:
    global _cache_hits, _cache_misses
    _cache_hits = 0
    _cache_misses = 0


def cached_call(
    client,
    model_key: str,
    messages: list,
    temperature: float = 0.0,
) -> str:
    """
    Make an API call, using disk cache if available.
    Increments hit/miss counters for reporting.
    """
    from src.models import call_model

    global _cache_hits, _cache_misses

    key = get_cache_key(model_key, messages, temperature)
    cached = load_cache(key)
    if cached is not None:
        _cache_hits += 1
        logger.debug(f"Cache hit: {key[:8]}...")
        return cached

    _cache_misses += 1
    response = call_model(client, model_key, messages, temperature)
    save_cache(key, response)
    return response


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_git_commit() -> str | None:
    """Return current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def build_run_metadata(model_key: str, condition: str, question_count: int) -> dict:
    """Produce a metadata dict to embed in every result JSON."""
    from src.models import MODELS

    return {
        "model_key": model_key,
        "model_string": MODELS.get(model_key, "unknown"),
        "condition": condition,
        "question_count": question_count,
        "temperature": 0.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
    }


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(
    total_input_tokens: int,
    total_output_tokens: int,
    model_key: str,
) -> float:
    """
    Rough cost estimate in USD based on approximate per-token prices.
    These are approximate — check OpenRouter for current pricing.
    """
    # Price per million tokens (input, output)
    PRICES = {
        "gpt52":  (1.75, 14.0),
        "claude": (3.00, 15.0),
        "gemini": (2.00, 12.0),
    }
    if model_key not in PRICES:
        return 0.0
    input_price, output_price = PRICES[model_key]
    cost = (total_input_tokens / 1_000_000) * input_price
    cost += (total_output_tokens / 1_000_000) * output_price
    return cost
