"""
src/conversation.py
Run single- and multi-turn conversations for each experimental condition.

Three conditions:
  A — Single-turn baseline:          one call per question
  B — Self-anchoring multi-turn:     5 turns, full history passed each turn
  C — Independent repetition control: 5 independent single-turn calls, no history

System prompt enforces the required answer format on every turn.
Temperature = 0.0 throughout.
"""

import logging
from typing import Optional

from src.confidence_extractor import extract_confidence, extract_answer
from src.correctness import check_correct
from src.utils import cached_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful and precise assistant. "
    "Always end every response with this exact line:\n"
    "Answer: [your answer here] | Confidence: [0-100]%\n"
    "Do not skip this line under any circumstances."
)

# Turn prompts for Condition B (self-anchoring)
TURN_PROMPTS_B = [
    # Turn 1: the question itself (handled separately)
    None,
    # Turn 2
    (
        "Based on your answer above, can you elaborate on your reasoning and "
        "confirm your answer? State your confidence at the end in the required format."
    ),
    # Turn 3
    (
        "Building on what you've said, is there anything you'd add, refine, or change? "
        "State your confidence in the required format."
    ),
    # Turn 4
    (
        "Given everything you've said in this conversation, what is your final "
        "considered answer? State your confidence in the required format."
    ),
    # Turn 5
    (
        "You've now had four attempts at this question. How confident are you in "
        "your answer? Final format: Answer: [...] | Confidence: [X]%"
    ),
]

TEMPERATURE = 0.0
N_TURNS = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_turn_record(
    turn_num: int,
    raw_response: str,
    domain: str,
    ground_truth: str,
    client,
    response_t1: Optional[str] = None,
) -> dict:
    """Build a single turn result dict."""
    conf = extract_confidence(raw_response)
    correct = check_correct(
        domain=domain,
        response=raw_response,
        ground_truth=ground_truth,
        client=client,
        response_t1=response_t1,
    )
    return {
        "turn": turn_num,
        "raw_response": raw_response,
        "extracted_confidence": conf,
        "extracted_answer": extract_answer(raw_response),
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Condition A — Single-turn baseline
# ---------------------------------------------------------------------------

def run_condition_a(
    client,
    model_key: str,
    question: dict,
) -> dict:
    """
    Run Condition A: single-turn baseline.
    Returns a result dict with one-element 'turns' list.
    """
    q_text = question["question"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q_text},
    ]

    logger.debug(f"Condition A | {model_key} | {question['id']}")
    response = cached_call(client, model_key, messages, TEMPERATURE)

    turn_record = _build_turn_record(
        turn_num=1,
        raw_response=response,
        domain=question["domain"],
        ground_truth=question["ground_truth"],
        client=client,
    )

    return {
        "question_id": question["id"],
        "domain": question["domain"],
        "condition": "A",
        "model": model_key,
        "turns": [turn_record],
        "cds": None,  # Not applicable for single-turn
        "ground_truth": question["ground_truth"],
    }


# ---------------------------------------------------------------------------
# Condition B — Self-anchoring multi-turn
# ---------------------------------------------------------------------------

def run_condition_b(
    client,
    model_key: str,
    question: dict,
) -> dict:
    """
    Run Condition B: 5-turn self-anchoring conversation.
    Full message history is passed on every turn.
    Returns a result dict with 5-element 'turns' list.
    """
    q_text = question["question"]
    domain = question["domain"]
    ground_truth = question["ground_truth"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_records = []
    response_t1: Optional[str] = None

    for turn_num in range(1, N_TURNS + 1):
        # Build user message
        if turn_num == 1:
            user_content = q_text
        else:
            user_content = TURN_PROMPTS_B[turn_num - 1]

        messages.append({"role": "user", "content": user_content})
        logger.debug(f"Condition B | {model_key} | {question['id']} | Turn {turn_num}")
        response = cached_call(client, model_key, messages, TEMPERATURE)

        # Capture T1 response for open-ended correctness check
        if turn_num == 1:
            response_t1 = response

        turn_record = _build_turn_record(
            turn_num=turn_num,
            raw_response=response,
            domain=domain,
            ground_truth=ground_truth,
            client=client,
            response_t1=response_t1 if turn_num > 1 else None,
        )
        turn_records.append(turn_record)

        # Append assistant response to history for next turn
        messages.append({"role": "assistant", "content": response})

    # Compute CDS
    from src.metrics import compute_cds
    cds = compute_cds(turn_records)

    return {
        "question_id": question["id"],
        "domain": domain,
        "condition": "B",
        "model": model_key,
        "turns": turn_records,
        "cds": cds,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Condition C — Independent repetition control
# ---------------------------------------------------------------------------

def run_condition_c(
    client,
    model_key: str,
    question: dict,
) -> dict:
    """
    Run Condition C: 5 independent single-turn calls, NO conversation history.
    This isolates whether drift is due to self-anchoring vs. mere repetition.
    Returns a result dict with 5-element 'turns' list.
    """
    q_text = question["question"]
    domain = question["domain"]
    ground_truth = question["ground_truth"]

    turn_records = []
    response_t1: Optional[str] = None

    for turn_num in range(1, N_TURNS + 1):
        # Each call is completely independent — fresh message list each time.
        # We append a tiny disambiguator to each call so each has a unique
        # cache key even though the question text is identical.
        # (Without this, all 5 calls would return the same cached response.)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{q_text}\n\n"
                    f"[Independent attempt {turn_num} of {N_TURNS}]"
                ),
            },
        ]

        logger.debug(f"Condition C | {model_key} | {question['id']} | Attempt {turn_num}")
        response = cached_call(client, model_key, messages, TEMPERATURE)

        if turn_num == 1:
            response_t1 = response

        turn_record = _build_turn_record(
            turn_num=turn_num,
            raw_response=response,
            domain=domain,
            ground_truth=ground_truth,
            client=client,
            response_t1=response_t1 if turn_num > 1 else None,
        )
        turn_records.append(turn_record)

    # Compute CDS (T5 - T1 confidence)
    from src.metrics import compute_cds
    cds = compute_cds(turn_records)

    return {
        "question_id": question["id"],
        "domain": domain,
        "condition": "C",
        "model": model_key,
        "turns": turn_records,
        "cds": cds,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_question(
    client,
    model_key: str,
    question: dict,
    condition: str,
) -> dict:
    """
    Run a single question under the specified condition.

    Args:
        client:     OpenRouter client
        model_key:  Key from MODELS dict ("gpt52", "claude", "gemini")
        question:   Question dict from question bank
        condition:  "A", "B", or "C"

    Returns:
        Result dict for this question.
    """
    if condition == "A":
        return run_condition_a(client, model_key, question)
    elif condition == "B":
        return run_condition_b(client, model_key, question)
    elif condition == "C":
        return run_condition_c(client, model_key, question)
    else:
        raise ValueError(f"Unknown condition '{condition}'. Must be 'A', 'B', or 'C'.")
