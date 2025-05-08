import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dynasor.core.evaluator import math_equal  # For answer equality check if needed

# --------------------------------------------------------------------------------------
# Helper utilities (adapted from the original post-process notebook so we can run stand-alone)
# --------------------------------------------------------------------------------------

def entropy(p_list: List[float]) -> float:
    """Compute the entropy of a probability distribution expressed as a list."""
    if not p_list:
        return 0.0
    return -sum(p * math.log(p, 2) for p in p_list if p > 0)


def _norm(counts: List[int]) -> List[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def _count(values: List[str]) -> List[int]:
    counter: Dict[str, int] = defaultdict(int)
    for v in values:
        counter[v] += 1
    return list(counter.values())


def item_entropy(answers: List[str]) -> float:
    """Entropy of a list of answers (Dynasor uses this to detect convergence)."""
    return entropy(_norm(_count(answers)))


# Same vocab Dynasor uses to detect uncertainty words in the probe completion
UNCERTAIN_WORDS = {"wait", "hold", "but", "okay", "no", "hmm"}

# --------------------------------------------------------------------------------------
# Loading & preprocessing the per-round JSON files that `run.py` produced
# --------------------------------------------------------------------------------------

def load_results(results_dir: str) -> Tuple[Dict[str, dict], int, int]:
    """Load every `question_X_tokens_Y.json` file under *results_dir*.

    Returns a nested dict keyed by (question, trial) where each value stores
    lists indexed by *step_id* (same ordering as the token budgets) with:
        answers, probabilities, responses, is_finished

    The function also infers *step_size* and *max_tokens* from the filenames.
    """
    pattern = os.path.join(results_dir, "question_*_tokens_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No result JSONs found in {results_dir}")

    # Infer step size by looking at the smallest positive token budget
    token_values = []
    for f in files:
        token_values.append(int(os.path.basename(f).split("_tokens_")[1].split(".")[0]))
    step_size = min(token_values)
    max_tokens = max(token_values)

    # Accumulator structure
    data: Dict[str, dict] = {}

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Parse question id & token budget from filename
        base = os.path.basename(fp)
        qid = int(base.split("question_")[1].split("_tokens")[0])
        tokens_budget = int(base.split("_tokens_")[1].split(".")[0])
        step_id = (tokens_budget // step_size) - 1  # zero-based step index

        # Retrieve lists for the *num_trials* runs
        answers = obj.get("probe_extracted_answers") or []
        probs = obj.get("probe_extracted_probabilities") or []
        responses = obj.get("probe_responses_text") or obj.get("probe_responses") or []
        finished_flags = obj.get("is_finished") or []

        assert len(answers) == len(probs) == len(responses) == len(finished_flags), "Inconsistent list lengths in JSON."
        num_trials = len(answers)

        for trial in range(num_trials):
            key = f"{qid}_{trial}"
            if key not in data:
                data[key] = {
                    "answers": [],
                    "probabilities": [],
                    "responses": [],
                    "is_finished": [],
                    "token_budgets": [],  # parallel list with step index
                    "target": obj["target"],
                }
            data[key]["answers"].append(answers[trial])
            data[key]["probabilities"].append(probs[trial])
            data[key]["responses"].append(responses[trial])
            data[key]["is_finished"].append(finished_flags[trial])
            data[key]["token_budgets"].append(tokens_budget)

    # Sort each list by token budget so we can iterate sequentially later
    for v in data.values():
        zipped = list(zip(v["token_budgets"], v["answers"], v["probabilities"], v["responses"], v["is_finished"]))
        zipped.sort(key=lambda t: t[0])
        v["token_budgets"], v["answers"], v["probabilities"], v["responses"], v["is_finished"] = map(list, zip(*zipped))

    return data, step_size, max_tokens

# --------------------------------------------------------------------------------------
# Early-exit decision following Dynasor (see earlyexit_accuracy in the notebook)
# --------------------------------------------------------------------------------------

def decide_early_exit(
    answers: List[str],
    probabilities: List[float],
    responses: List[str],
    finished_flags: List[bool],
    token_budgets: List[int],
    *,
    jump: int,
    bar: int,
    warmup_steps: int,
) -> Tuple[int, float]:
    """Return (tokens_used, probability_at_exit).

    probability_at_exit is None if the run finished naturally before we could probe.
    """

    for step_idx, tokens in enumerate(token_budgets):
        finished = finished_flags[step_idx]

        # Natural termination from the model itself (finish_reason != 'length')
        if finished:
            return tokens, None

        # --- Probe-based early-exit check (adapted from Dynasor) ---
        if tokens < warmup_steps:
            continue

        # clip every <jump> steps up to current step for entropy computation
        clip_answers = answers[: step_idx + 1 : jump]
        clip_responses = responses[: step_idx + 1 : jump]

        # Entropy on the last <bar> answers
        if len(clip_answers) < bar:
            continue
        recent_answers = clip_answers[-bar:]
        recent_responses = clip_responses[-bar:]

        ent = item_entropy(recent_answers)
        non_empty_cnt = sum(ans != "" for ans in recent_answers)
        certain_cnt = sum(
            all(word not in resp.lower() for word in UNCERTAIN_WORDS)
            for resp in recent_responses
        )

        if ent <= 0.01 and non_empty_cnt == bar and certain_cnt == bar:
            # We exit here and use the *last* probability available
            prob_at_exit = probabilities[step_idx] if step_idx < len(probabilities) else None
            return tokens, prob_at_exit

    # If never exited early, we use the final step's info (could be length-capped)
    return token_budgets[-1], None

# --------------------------------------------------------------------------------------
# Main analysis driver
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse Dynasor early-exit probabilities and generate a scatter plot.")
    parser.add_argument("--results-dir", required=True, help="Directory that contains question_*_tokens_*.json files (output of run.py)")
    parser.add_argument("--jump", type=int, default=4, help="Probe detection interval (see earlyexit_accuracy).")
    parser.add_argument("--bar", type=int, default=2, help="Number of consecutive consistent probes required to exit early.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warm-up tokens before early-exit logic kicks in.")
    parser.add_argument("--output", default="probability_vs_tokens.png", help="Filename for the saved plot.")

    args = parser.parse_args()

    data, step_size, max_tokens = load_results(args.results_dir)

    scatter_x = []  # tokens used
    scatter_y = []  # probability

    for key, v in data.items():
        tokens_used, prob = decide_early_exit(
            v["answers"],
            v["probabilities"],
            v["responses"],
            v["is_finished"],
            v["token_budgets"],
            jump=args.jump,
            bar=args.bar,
            warmup_steps=args.warmup_steps,
        )
        if prob is not None:
            scatter_x.append(tokens_used)
            scatter_y.append(prob)

    if not scatter_x:
        print("No early-exit events with available probabilities were found.")
        return

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(scatter_x, scatter_y, alpha=0.6)
    plt.xlabel("Tokens at early termination")
    plt.ylabel("Model self-rated probability")
    plt.title("Dynasor Early-Exit Probabilities vs. Tokens")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main() 