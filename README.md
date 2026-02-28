# Self-Anchoring Calibration Drift (SACD)

**Research question:** When an LLM builds on its own prior outputs across conversation turns — with no new external information — does its expressed confidence inflate even as accuracy stays flat?

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key (via OpenRouter)
cp env.example .env
# Edit .env and replace sk-or-YOUR_KEY_HERE with your real key

# 3. Generate the question bank
python src/question_bank.py
```

## Running Order

```bash
# 4. Validate everything works (makes test API calls, ~$0.10)
python experiments/validate_setup.py

# 5. Dry run — ALWAYS do this first. Inspect the output carefully.
#    15 questions, all 3 models, all 3 conditions. ~$3-5.
python experiments/run_experiment.py --models gpt52 claude gemini --dry-run

# 6. Inspect dry-run output:
#    - Confidence extraction rate ≥ 90%?
#    - Answers look reasonable?
#    - Cache files appearing in data/cache/?
#    - Cost estimate reasonable?
#    Only proceed if everything checks out.

# 7. Full run (~$40-50, ~150 questions × 3 conditions × 3 models)
python experiments/run_experiment.py --models gpt52 claude gemini

# 8. Compute metrics and statistical tests
python analysis/compute_metrics.py

# 9. Generate publication figures
python analysis/make_figures.py
```

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **A** | Single-turn baseline: one call per question |
| **B** | Self-anchoring multi-turn: 5 turns, full history passed each turn |
| **C** | Independent repetition control: 5 independent single-turn calls, no history |

## Metrics

- **CDS** (Confidence Drift Score): `confidence_T5 - confidence_T1` — positive = inflation
- **ECE** (Expected Calibration Error): standard binning-based calibration error at each turn
- **CDR** (Calibration Drift Rate): slope of ECE across turns — positive = worsening

## Question Bank

150 questions across 3 domains (50 each):
- **Factual**: capitals, historical dates, scientific constants, geography, literature
- **Technical**: algorithms, databases, networking, software engineering
- **Open-ended**: system design tradeoffs, architectural comparisons

## Models

All accessed via OpenRouter:

| Key | Model |
|-----|-------|
| `gpt52` | `openai/gpt-5.2` |
| `claude` | `anthropic/claude-sonnet-4-6` |
| `gemini` | `google/gemini-3.1-pro` |

## Critical Rules

- **Cache is sacred.** Never delete `data/cache/`. Never overwrite cache entries.
  If you need to change the prompt, change the prompt (new cache key) — do not touch existing files.
- **Temperature = 0.0 always.** No exceptions.
- **Dry run before full run.** The $3-5 dry run catches bugs before the $40+ full run.
- **Fail loudly.** If extraction rate < 90%, the pipeline halts and prints all failed responses.

## Repository Structure

```
calibration-drift/
├── env.example              # Copy to .env and add your API key
├── requirements.txt
├── README.md
├── data/
│   ├── questions.json       # 150-question bank (generated once)
│   └── cache/               # API response cache — NEVER delete
├── src/
│   ├── models.py            # OpenRouter client
│   ├── question_bank.py     # Question definitions and loader
│   ├── conversation.py      # Run single- and multi-turn conversations
│   ├── confidence_extractor.py  # Parse confidence % from model output
│   ├── correctness.py       # Check answer correctness per domain
│   ├── metrics.py           # CDS, ECE, CDR metric computation
│   └── utils.py             # Caching, logging, retry logic
├── experiments/
│   ├── run_experiment.py    # Main entry point
│   └── validate_setup.py    # Pre-flight checks
├── analysis/
│   ├── compute_metrics.py   # Load raw results, output metric tables
│   └── make_figures.py      # Generate all paper figures
└── results/
    ├── raw/                 # One JSON per model × condition × run
    ├── metrics/             # Computed metric tables (CSV)
    └── figures/             # PNG + PDF figures
```
