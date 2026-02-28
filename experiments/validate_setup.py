"""
experiments/validate_setup.py
Pre-flight checks before running the full experiment.

Run this first:
    python experiments/validate_setup.py

All checks must pass before proceeding to run_experiment.py.
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models import get_client, MODELS, call_model
from src.utils import setup_logging, CACHE_DIR, estimate_cost
from src.question_bank import QUESTIONS_PATH, load_questions
from src.confidence_extractor import extract_confidence

setup_logging()

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def _result(ok: bool, msg: str) -> bool:
    print(f"  [{PASS if ok else FAIL}] {msg}")
    return ok


# ---------------------------------------------------------------------------
# Check 1: API key
# ---------------------------------------------------------------------------

def check_api_key() -> bool:
    print("\n[1] Checking OPENROUTER_API_KEY...")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return _result(False, "OPENROUTER_API_KEY is not set. Add it to .env: OPENROUTER_API_KEY=sk-or-...")
    if not key.startswith("sk-or-"):
        return _result(False, f"Key doesn't look like an OpenRouter key (should start with 'sk-or-'): {key[:12]}...")
    return _result(True, f"API key found: {key[:12]}...{key[-4:]}")


# ---------------------------------------------------------------------------
# Check 2: Test API call to all 3 models
# ---------------------------------------------------------------------------

def check_model_connectivity() -> bool:
    print("\n[2] Testing connectivity to all 3 models (simple 'Hello' prompt)...")
    all_ok = True
    try:
        client = get_client()
    except EnvironmentError as e:
        return _result(False, f"Could not create client: {e}")

    for model_key, model_string in MODELS.items():
        try:
            t0 = time.time()
            messages = [{"role": "user", "content": "Hello. Reply with exactly one word: OK"}]
            response = call_model(client, model_key, messages, temperature=0.0)
            elapsed = time.time() - t0
            ok = bool(response and len(response.strip()) > 0)
            _result(ok, f"{model_key:8s} ({model_string}): responded in {elapsed:.1f}s — {response.strip()[:60]!r}")
            if not ok:
                all_ok = False
        except Exception as e:
            _result(False, f"{model_key:8s} ({model_string}): ERROR — {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Check 3: Question bank
# ---------------------------------------------------------------------------

def check_question_bank() -> bool:
    print("\n[3] Checking data/questions.json...")
    if not os.path.exists(QUESTIONS_PATH):
        _result(False, f"File not found: {QUESTIONS_PATH}")
        print(f"       → Run: python src/question_bank.py")
        return False

    try:
        questions = load_questions()
    except Exception as e:
        return _result(False, f"Failed to load questions: {e}")

    all_ok = True

    n_total = len(questions)
    ok_total = n_total == 150
    _result(ok_total, f"Total questions: {n_total} (expected 150)")
    if not ok_total:
        all_ok = False

    # Per-domain counts
    domains: dict[str, int] = {}
    for q in questions:
        domains[q.get("domain", "?")] = domains.get(q.get("domain", "?"), 0) + 1

    for domain, expected in [("factual", 50), ("technical", 50), ("openended", 50)]:
        count = domains.get(domain, 0)
        ok = count == expected
        _result(ok, f"  {domain}: {count} (expected {expected})")
        if not ok:
            all_ok = False

    # Required fields check
    required_fields = ["id", "domain", "question", "ground_truth", "difficulty"]
    missing_field_qs = []
    for q in questions:
        missing = [f for f in required_fields if f not in q]
        if missing:
            missing_field_qs.append((q.get("id", "?"), missing))

    if missing_field_qs:
        _result(False, f"Questions missing required fields: {missing_field_qs[:5]}")
        all_ok = False
    else:
        _result(True, "All questions have required fields (id, domain, question, ground_truth, difficulty)")

    # ID uniqueness
    ids = [q["id"] for q in questions]
    dupes = [id_ for id_ in ids if ids.count(id_) > 1]
    if dupes:
        _result(False, f"Duplicate question IDs: {set(dupes)}")
        all_ok = False
    else:
        _result(True, "All question IDs are unique")

    return all_ok


# ---------------------------------------------------------------------------
# Check 4: Cache directory writable
# ---------------------------------------------------------------------------

def check_cache_dir() -> bool:
    print("\n[4] Checking data/cache/ directory...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    test_file = os.path.join(CACHE_DIR, "_validate_write_test.json")
    try:
        with open(test_file, "w") as f:
            json.dump({"test": True}, f)
        os.remove(test_file)
        return _result(True, f"Cache directory is writable: {CACHE_DIR}")
    except Exception as e:
        return _result(False, f"Cache directory not writable: {e}")


# ---------------------------------------------------------------------------
# Check 5: Cost estimate from 3-question sample
# ---------------------------------------------------------------------------

def check_cost_estimate() -> bool:
    print("\n[5] Estimating cost from 3-question sample...")
    try:
        client = get_client()
    except Exception as e:
        _result(False, f"Cannot create client: {e}")
        return False

    sample_questions = [
        "What is the capital of Australia?",
        "What is the time complexity of binary search?",
        "What are the main trade-offs between microservices and monolithic architectures?",
    ]

    from src.conversation import SYSTEM_PROMPT

    total_input_chars = 0
    total_output_chars = 0
    ok = True

    for i, q in enumerate(sample_questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        try:
            # Use the first model only for cost sampling
            response = call_model(client, "gpt52", messages, temperature=0.0)
            total_input_chars += sum(len(m["content"]) for m in messages)
            total_output_chars += len(response)
            print(f"    Sample {i + 1}: {len(response)} chars output")
        except Exception as e:
            _result(False, f"Sample call {i + 1} failed: {e}")
            ok = False

    if not ok:
        return False

    # Rough token counts (1 token ≈ 4 chars)
    avg_input_tokens = total_input_chars // 4 // len(sample_questions)
    avg_output_tokens = total_output_chars // 4 // len(sample_questions)

    # Full experiment: 150 questions × 3 conditions × avg 5 turns per B/C, 1 for A
    # Condition A: 1 turn, B: 5 turns with growing context, C: 5 turns
    # Average ≈ (1 + 5 + 5) / 3 ≈ 3.7 turns per condition
    total_turns = 150 * 3 * 3.7
    total_input_tokens = int(total_turns * avg_input_tokens)
    total_output_tokens = int(total_turns * avg_output_tokens)

    print(f"\n  Per question avg: ~{avg_input_tokens} input tokens, ~{avg_output_tokens} output tokens")
    print(f"  Full experiment estimate ({int(total_turns)} total turns across 3 models):\n")

    grand_total = 0.0
    for model_key in MODELS:
        cost = estimate_cost(total_input_tokens, total_output_tokens, model_key)
        grand_total += cost
        print(f"    {model_key:8s}  in≈{total_input_tokens:>9,} tok  out≈{total_output_tokens:>9,} tok  ≈ ${cost:.2f}")

    print(f"\n    {'TOTAL':8s}  {'':>25}  ≈ ${grand_total:.2f}")

    if grand_total > 100:
        print(f"  {WARN} Cost estimate > $100 — verify token counts before proceeding")
    elif grand_total > 50:
        print(f"  {WARN} Cost estimate > $50 — within expected range but verify")
    else:
        print(f"  {PASS} Cost estimate looks reasonable (${grand_total:.2f})")

    return True


# ---------------------------------------------------------------------------
# Check 6: Confidence extraction test
# ---------------------------------------------------------------------------

def check_extraction() -> bool:
    print("\n[6] Testing confidence extraction with known strings...")
    test_cases = [
        ("Answer: Canberra | Confidence: 85%", 0.85),
        ("The answer is Paris. Answer: Paris | Confidence: 72%", 0.72),
        ("Answer: O(log n) | Confidence: 95%", 0.95),
        ("I'm 60% confident the answer is X. Answer: X | Confidence: 60%", 0.60),
        ("No confidence given here.", None),
    ]

    all_ok = True
    for text, expected in test_cases:
        extracted = extract_confidence(text)
        ok = (extracted == expected) or (
            extracted is not None
            and expected is not None
            and abs(extracted - expected) < 0.001
        )
        status = PASS if ok else FAIL
        print(f"    [{status}] extract({text[:50]!r}) → {extracted} (expected {expected})")
        if not ok:
            all_ok = False

    return _result(all_ok, "Confidence extraction working correctly")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  SACD Experiment — Pre-flight Validation")
    print("=" * 70)

    from dotenv import load_dotenv
    load_dotenv()

    checks = [
        ("API key", check_api_key),
        ("Model connectivity", check_model_connectivity),
        ("Question bank", check_question_bank),
        ("Cache directory", check_cache_dir),
        ("Confidence extraction", check_extraction),
        ("Cost estimate", check_cost_estimate),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [{FAIL}] {name} raised unexpected exception: {e}")
            results[name] = False

    # Final summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("  ✓ ALL CHECKS PASSED. You may proceed to run_experiment.py.")
        print("  → python experiments/run_experiment.py --models gpt52 claude gemini --dry-run")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"  ✗ {len(failed)} check(s) FAILED: {failed}")
        print("  → Fix the issues above before running the experiment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
