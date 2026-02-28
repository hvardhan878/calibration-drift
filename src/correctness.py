"""
src/correctness.py
Check answer correctness per domain.

Factual / Technical:  exact match + alias matching (case-insensitive, punct-stripped)
Open-ended:           embedding cosine similarity between Turn 1 and Turn 4 ≥ 0.85
"""

import re
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factual / Technical correctness
# ---------------------------------------------------------------------------

# Common aliases for canonical ground-truth strings.
# Key = normalized canonical form.  Value = list of accepted normalized aliases.
_ALIASES: dict[str, list[str]] = {
    # Capitals
    "canberra": ["canberra"],
    "brasilia": ["brasilia", "brasília"],
    "cape town": ["cape town", "capetown"],
    "naypyidaw": ["naypyidaw", "naypyitaw"],
    "astana": ["astana", "nur-sultan", "nursultan"],
    "wellington": ["wellington"],
    "sucre": ["sucre"],
    "sri jayawardenepura kotte": [
        "sri jayawardenepura kotte",
        "jayawardenepura kotte",
        "kotte",
        "sri jayawardenapura",
    ],
    "amsterdam": ["amsterdam"],

    # Dates / numbers expressed multiple ways
    "1945": ["1945"],
    "299792458": ["299792458", "299,792,458", "3×10^8", "3 × 10^8", "3x10^8"],
    "6022": ["6.022", "avogadros number"],  # partial — containment handles rest

    # Science
    "h2o": ["h2o", "h₂o", "dihydrogen monoxide"],

    # Wars
    "world war ii": ["world war 2", "ww2", "wwii", "second world war", "world war two"],

    # Algorithms
    "o(log n)": ["o(log n)", "o log n", "log n", "olog n"],
    "o(n log n)": ["o(nlogn)", "o(n log n)", "nlogn", "n log n"],
    "o(1)": ["o(1)", "constant time", "constant"],
    "o(v^3)": ["o(v^3)", "o(v3)", "v cubed"],

    # ACID
    "atomicity consistency isolation durability": [
        "atomicity consistency isolation durability",
        "atomicity, consistency, isolation, durability",
    ],

    # SOLID
    "single responsibility openclosed liskov substitution interface segregation dependency inversion": [
        "single responsibility",  # containment — if any of these appear it's fine
    ],

    # Patterns
    "singleton": ["singleton"],
    "observer": ["observer"],
}


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_correct_factual(response: str, ground_truth: str) -> bool:
    """
    Exact match + alias matching for factual/technical domains.
    Returns True if ground truth (or any accepted alias) is found in the response.
    """
    if not ground_truth:
        # Open-ended question accidentally passed here — skip
        return False

    resp_norm = _normalize(response)
    gt_norm = _normalize(ground_truth)

    # Direct containment
    if gt_norm in resp_norm:
        return True

    # Alias lookup
    for canonical, alts in _ALIASES.items():
        if gt_norm == canonical or gt_norm in alts:
            if any(_normalize(alt) in resp_norm for alt in [canonical] + alts):
                return True

    # Numeric: strip commas/spaces and try again (e.g. "299,792,458")
    gt_digits = re.sub(r"[\s,]", "", gt_norm)
    resp_digits = re.sub(r"[\s,]", "", resp_norm)
    if gt_digits and gt_digits in resp_digits:
        return True

    return False


# ---------------------------------------------------------------------------
# Open-ended correctness (embedding similarity)
# ---------------------------------------------------------------------------

def _embed(client, text: str) -> np.ndarray:
    """Embed text using OpenRouter's text-embedding-3-small endpoint."""
    result = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text,
    )
    return np.array(result.data[0].embedding, dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def check_correct_openended(
    response_t1: str,
    response_t4: str,
    client,
    threshold: float = 0.85,
) -> bool:
    """
    For open-ended questions: check self-consistency between Turn 1 and Turn 4
    using embedding cosine similarity.
    Returns True if similarity >= threshold (default 0.85).
    Logs the similarity score for debugging.
    """
    try:
        e1 = _embed(client, response_t1)
        e4 = _embed(client, response_t4)
        sim = cosine_similarity(e1, e4)
        logger.debug(f"Open-ended cosine similarity T1↔T4: {sim:.4f} (threshold={threshold})")
        return sim >= threshold
    except Exception as e:
        logger.warning(f"Embedding call failed for open-ended correctness: {e}")
        return False


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def check_correct(
    domain: str,
    response: str,
    ground_truth: str,
    client=None,
    response_t1: str | None = None,
) -> bool:
    """
    Dispatch to the correct checker based on domain.

    Args:
        domain:      "factual", "technical", or "openended"
        response:    Current turn's raw model response
        ground_truth: Ground truth string (empty for open-ended)
        client:      OpenRouter client (required for open-ended)
        response_t1: Turn 1 response (required for open-ended)
    """
    if domain in ("factual", "technical"):
        return check_correct_factual(response, ground_truth)
    elif domain == "openended":
        if response_t1 is None or client is None:
            logger.debug("Open-ended check skipped — missing T1 response or client")
            return False
        return check_correct_openended(response_t1, response, client)
    else:
        raise ValueError(f"Unknown domain '{domain}'")
