"""
experiments/run_experiment.py
Main entry point for the SACD experiment.

Usage:
    # Always run dry-run first to validate everything:
    python experiments/run_experiment.py --models gpt52 claude gemini --dry-run

    # Full experiment (~$40-50 in API costs):
    python experiments/run_experiment.py --models gpt52 claude gemini
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path fix — allow running from repo root or experiments/ dir
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tqdm import tqdm

from src.models import get_client, MODELS
from src.utils import (
    setup_logging,
    build_run_metadata,
    get_cache_stats,
    reset_cache_stats,
    estimate_cost,
    CACHE_DIR,
)
from src.question_bank import load_questions, get_dry_run_questions
from src.conversation import run_question
from src.confidence_extractor import (
    extract_confidence,
    extraction_success_rate,
    assert_extraction_success,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_RAW_DIR = os.path.join(_REPO_ROOT, "results", "raw")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_model_condition(
    client,
    model_key: str,
    condition: str,
    questions: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """
    Run all questions for one model × condition combination.

    Returns list of result dicts.
    """
    label = f"{model_key}/{condition}"
    results = []
    all_responses: list[str] = []

    bar = tqdm(
        questions,
        desc=f"{label}",
        unit="q",
        leave=True,
        ncols=90,
    )

    for question in bar:
        try:
            result = run_question(client, model_key, question, condition)
            results.append(result)

            # Collect all raw responses for extraction rate check
            for turn in result.get("turns", []):
                raw = turn.get("raw_response", "")
                if raw:
                    all_responses.append(raw)

        except Exception as e:
            logger.error(
                f"Error processing {question['id']} ({label}): {e}"
            )
            # Record a failed result so we have a complete record
            results.append({
                "question_id": question["id"],
                "domain": question["domain"],
                "condition": condition,
                "model": model_key,
                "turns": [],
                "cds": None,
                "ground_truth": question.get("ground_truth", ""),
                "error": str(e),
            })

    # Extraction success check — halt loudly if below 90%
    if all_responses:
        assert_extraction_success(all_responses, model_key, threshold=0.90)

    return results


# ---------------------------------------------------------------------------
# Dry-run summary printer
# ---------------------------------------------------------------------------

def print_dryrun_summary(
    all_results: dict,  # {model_key: {condition: [results]}}
    questions: list[dict],
    token_usage_estimate: dict,
) -> None:
    """Print a detailed dry-run summary for manual inspection."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  DRY-RUN SUMMARY")
    print(sep)
    print(f"  Questions tested: {len(questions)}")
    print(f"  Models: {list(all_results.keys())}")
    print(f"  Conditions: A (baseline), B (self-anchoring), C (control)")

    stats = get_cache_stats()
    print(f"\n  Cache stats:  hits={stats['hits']}  misses={stats['misses']}")
    print(sep)

    for model_key, conditions in all_results.items():
        print(f"\n{'─' * 60}")
        print(f"  MODEL: {model_key}  ({MODELS.get(model_key, '?')})")
        print(f"{'─' * 60}")

        for condition, results in conditions.items():
            all_responses = [
                t.get("raw_response", "")
                for r in results
                for t in r.get("turns", [])
                if t.get("raw_response")
            ]
            success_rate = extraction_success_rate(all_responses)
            print(f"\n  Condition {condition} — Extraction success rate: {success_rate:.1%} ({len(all_responses)} responses)")

            # Show first 3 questions in detail
            for result in results[:3]:
                qid = result["question_id"]
                domain = result["domain"]
                gt = result.get("ground_truth", "")
                print(f"\n    Q [{qid}] ({domain})")

                for turn in result.get("turns", []):
                    raw = turn.get("raw_response", "")
                    conf = turn.get("extracted_confidence")
                    correct = turn.get("correct")
                    # Show truncated response
                    preview = raw[:120].replace("\n", " ")
                    if len(raw) > 120:
                        preview += "..."
                    conf_str = f"{conf:.2f}" if conf is not None else "FAILED"
                    print(f"      T{turn['turn']}: conf={conf_str}  correct={correct}")
                    print(f"           {preview!r}")

                if gt:
                    print(f"      ground_truth: {gt!r}")
                if result.get("cds") is not None:
                    print(f"      CDS (T5-T1 conf): {result['cds']:+.3f}")

            # Log any extraction failures
            failures = [
                r for r in results
                for t in r.get("turns", [])
                if t.get("raw_response") and extract_confidence(t["raw_response"]) is None
            ]
            if failures:
                print(f"\n  ⚠ Extraction failures ({len(failures)} turns):")
                for t in failures[:5]:
                    print(f"    {t.get('raw_response', '')[:200]!r}")

    # Cost estimate for full run
    print(f"\n{sep}")
    print("  COST ESTIMATE FOR FULL RUN (150 questions)")
    print(sep)
    total = 0.0
    for model_key in all_results:
        # Scale from dry-run token counts
        dry_n = len(questions)
        full_n = 150
        scale = full_n / dry_n if dry_n > 0 else 1.0
        est_in = token_usage_estimate.get(model_key, {}).get("input_tokens", 0) * scale
        est_out = token_usage_estimate.get(model_key, {}).get("output_tokens", 0) * scale
        cost = estimate_cost(int(est_in), int(est_out), model_key)
        total += cost
        print(f"  {model_key:8s}  in≈{int(est_in):>8,} tok  out≈{int(est_out):>8,} tok  ≈ ${cost:.2f}")
    print(f"  {'TOTAL':8s}  {'':>20}                  ≈ ${total:.2f}")
    print(sep)
    print("\n  → Inspect the output above carefully before running the full experiment.")
    print("  → Only proceed if extraction rates are ≥ 90% and answers look correct.\n")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    results: list[dict],
    model_key: str,
    condition: str,
    dry_run: bool,
    metadata: dict,
) -> str:
    """Save results to disk. Returns the output file path."""
    os.makedirs(RESULTS_RAW_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "dryrun_" if dry_run else ""
    filename = f"{prefix}{model_key}_{condition}_{timestamp}.json"
    path = os.path.join(RESULTS_RAW_DIR, filename)

    payload = {
        "metadata": metadata,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved {len(results)} results → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SACD experiment (Self-Anchoring Calibration Drift).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to run (default: all three)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Which conditions to run (default: all three)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run with only 5 questions per domain (15 total). "
            "ALWAYS run this before the full experiment."
        ),
    )
    parser.add_argument(
        "--n-per-domain",
        type=int,
        default=5,
        help="Number of questions per domain in dry-run mode (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    reset_cache_stats()

    logger.info(f"{'DRY RUN' if args.dry_run else 'FULL RUN'} starting")
    logger.info(f"Models: {args.models}")
    logger.info(f"Conditions: {args.conditions}")

    # Load questions
    questions_all = load_questions()
    if args.dry_run:
        questions = get_dry_run_questions(questions_all, n_per_domain=args.n_per_domain)
        logger.info(
            f"Dry-run mode: {len(questions)} questions "
            f"({args.n_per_domain} per domain)"
        )
    else:
        questions = questions_all
        logger.info(f"Full run: {len(questions)} questions")

    client = get_client()

    # Results accumulator for dry-run summary
    all_results: dict[str, dict[str, list[dict]]] = {}
    # Token usage estimator (approximate — we don't have real token counts
    # unless we parse API metadata; use response lengths as proxy)
    token_usage: dict[str, dict] = {}

    start_time = time.time()

    for model_key in args.models:
        all_results[model_key] = {}
        token_usage[model_key] = {"input_tokens": 0, "output_tokens": 0}

        for condition in args.conditions:
            logger.info(f"Running model={model_key} condition={condition}")

            results = run_model_condition(
                client=client,
                model_key=model_key,
                condition=condition,
                questions=questions,
                dry_run=args.dry_run,
            )
            all_results[model_key][condition] = results

            # Rough token estimate: avg 400 input tokens per turn, response length / 4 ≈ tokens
            total_response_chars = sum(
                len(t.get("raw_response", ""))
                for r in results
                for t in r.get("turns", [])
            )
            total_turns = sum(len(r.get("turns", [])) for r in results)
            token_usage[model_key]["input_tokens"] += total_turns * 400
            token_usage[model_key]["output_tokens"] += total_response_chars // 4

            # Save results to disk
            metadata = build_run_metadata(model_key, condition, len(results))
            save_results(results, model_key, condition, args.dry_run, metadata)

    elapsed = time.time() - start_time
    logger.info(f"Experiment completed in {elapsed:.1f}s")

    # Dry-run summary
    if args.dry_run:
        print_dryrun_summary(all_results, questions, token_usage)

    # Full-run summary
    else:
        print("\n" + "=" * 60)
        print("  FULL RUN COMPLETE")
        print("=" * 60)
        for model_key, conditions in all_results.items():
            for condition, results in conditions.items():
                n = len(results)
                n_with_cds = sum(1 for r in results if r.get("cds") is not None)
                cds_vals = [r["cds"] for r in results if r.get("cds") is not None]
                import numpy as np
                mean_cds = float(np.mean(cds_vals)) if cds_vals else float("nan")
                print(
                    f"  {model_key}/{condition}:  n={n}  "
                    f"mean_CDS={mean_cds:+.3f}  "
                    f"(CDS available: {n_with_cds}/{n})"
                )
        cache = get_cache_stats()
        print(f"\n  Cache: {cache['hits']} hits, {cache['misses']} misses")
        print(f"  Results saved to: {RESULTS_RAW_DIR}")
        print(
            "\n  Next steps:\n"
            "    python analysis/compute_metrics.py\n"
            "    python analysis/make_figures.py\n"
        )


if __name__ == "__main__":
    main()
