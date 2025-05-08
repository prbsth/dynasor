import argparse
from tqdm import tqdm
from utils import save_json, load_dataset, set_seed
from dynasor.core.evaluator import (
    extract_answer,
    extract_boxed_answer,
    strip_string,
    math_equal,
)
from clients import vllmClientModel
import re # Added for the new extraction function


# New helper function
def extract_boxed_answer_and_probability_local(text: str, data_name: str):
    """Return (answer, probability) from *text*.

    • `answer` is obtained by Dynasor's own `extract_boxed_answer`, so nested
      braces / extra TeX are handled exactly the same way as elsewhere.
    • `probability` is the first number appearing after the words
      "Probability" or "Confidence" (case-insensitive).  If a % sign is
      present we convert to the 0-1 range.
    """

    # ---------- answer ----------
    answer = extract_boxed_answer(text, data_name)

    # ---------- probability ----------
    probability = None
    if text:
        # Allow "Probability", "Prob.", "Confidence" … accept : or =
        prob_regex = re.compile(
            r"(?i)(?:probability|confidence)\s*[:=]?\s*([-+]?[0-9]*\.?[0-9]+)\s*%?"
        )
        m = prob_regex.search(text)
        if m:
            try:
                p_val = float(m.group(1))
                # If written as percentage (>1 or followed by %) → convert
                if "%" in m.group(0) or p_val > 1:
                    p_val /= 100.0
                # Clamp to [0,1]
                if 0.0 <= p_val <= 1.0:
                    probability = p_val
            except ValueError:
                pass

    return answer, probability


def parse_args():
    parser = argparse.ArgumentParser(description="Token Deprivation Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amc23", "aime24", "GPQADiamond", "math500"],
        help="Dataset to use (amc23 or aime24 or math500 or GPQADiamond)",
    )
    parser.add_argument(
        "--output", type=str, default="", help="Path to output results file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name or path of the model to use",
    )
    parser.add_argument(
        "--answer-probe",
        type=str,
        default=(
            "**Final Answer**\\n"
            "Respond ONLY with the answer inside one \\boxed{...}.\\n\\n"
            "\\boxed{"
        ),
        help="Prompt appended after the main generation to elicit ONLY the final answer.  Override to customise wording.",
    )
    parser.add_argument(
        "--answer-probe-tokens",
        type=int,
        default=40,
        help="Token budget for the answer probe completion",
    )
    parser.add_argument(
        "--prob-probe",
        type=str,
        default=(
            "Given your final answer above, what is the probability (0 to 1) that your answer is correct?\\n"
            "Respond with ONLY the number.\\nProbability: "
        ),
        help="Prompt used to ask the model for its confidence once it has already produced an answer.",
    )
    parser.add_argument(
        "--prob-probe-tokens",
        type=int,
        default=15,
        help="Token budget for the probability probe completion",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the model to use",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="token-abc123",
        help="API key of the model to use",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of the question"
    )
    parser.add_argument(
        "--end", type=int, default=10000, help="End index of the question"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens per request for main generation",
    )
    parser.add_argument(
        "--step", type=int, default=128, help="Step size for token budget"
    )
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Number of trials per question"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_model(model_name, url, api_key):

    return vllmClientModel(model_name, url, api_key)


def execute_question_reuse(
    model,
    prompt,
    target,
    dataset_type, # Added dataset_type
    max_tokens=[2048],
    answer_probe=None,
    answer_probe_tokens=10,
    prob_probe=None,
    prob_probe_tokens=5,
    num_trials=10,
    problem_id=None,
    output_dir=None,
    top_p=0.95,
    temperature=0.6,
):
    results = []
    current_prompts = [model.prepare_prompt(prompt) for _ in range(num_trials)]
    for i in tqdm(range(len(max_tokens)), desc="Executing questions"):
        # print(f"Executing question {i} with max tokens {max_tokens[i]}")

        # Track which trials are finished
        if i == 0:
            is_finished = [False] * num_trials
            responses = model.generate_batch(
                current_prompts,
                max_tokens=max_tokens[i],
                is_actives=[True] * num_trials,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            # Calculate remaining tokens needed
            remaining_tokens = max_tokens[i] - max_tokens[i - 1]
            # Stitch previous response to prompt
            current_prompts = [
                current_prompt + completion[0]
                for current_prompt, completion in zip(current_prompts, completions)
            ]
            # Only generate for unfinished trials
            responses = model.generate_batch(
                current_prompts,
                max_tokens=remaining_tokens,
                is_actives=[not finished for finished in is_finished],
                top_p=top_p,
                temperature=temperature,
            )

        # print(responses)
        completions = []
        for trial_idx in range(num_trials): # Renamed to trial_idx to avoid conflict
            if is_finished[trial_idx]:
                completions.append(("", None))  # Empty completion for finished trials
            else:
                response = responses[trial_idx]
                if response is None:
                    completions.append(("", None))
                else:
                    text = response.choices[0].text
                    finish_reason = response.choices[0].finish_reason
                    # logprobs = response.choices[0].logprobs # logprobs not used in current script
                    completions.append((text, finish_reason))
                    # Update finished status if LLM completed naturally
                    if finish_reason != "length":
                        is_finished[trial_idx] = True

        # Save results for this round
        round_results = {
            "round": i,
            "problem_id": problem_id,
            "max_tokens": max_tokens[i],
            "prompts": current_prompts, # These are prompts before adding current completion
            "new_tokens": [completion[0] for completion in completions],
            "finish_reasons": [completion[1] for completion in completions],
            "is_finished": is_finished,
            "target": target,
        }

        # ---------------- FIRST PROBE: ANSWER ----------------
        answer_probe_prompts = [
            current_prompts[trial_idx] + completions[trial_idx][0] + (answer_probe or "")
            for trial_idx in range(num_trials)
        ]

        answer_probe_responses = model.generate_batch_probe(
            answer_probe_prompts,
            max_tokens=answer_probe_tokens,
            is_actives=[not finished for finished in is_finished],
        )

        answer_probe_texts = []
        extracted_answers = []
        for trial_idx in range(num_trials):
            if is_finished[trial_idx] or answer_probe_responses[trial_idx] is None:
                answer_probe_texts.append("")
                extracted_answers.append(None)
            else:
                txt = answer_probe_responses[trial_idx].choices[0].text
                answer_probe_texts.append(txt)
                ans = extract_boxed_answer(txt, dataset_type)
                extracted_answers.append(ans)

        # ---------------- SECOND PROBE: PROBABILITY ----------------
        # Determine the answer string to reference in the probability probe
        answer_strings_for_prob = []
        for trial_idx in range(num_trials):
            if is_finished[trial_idx]:
                # Use answer from natural completion
                full_gen_text = current_prompts[trial_idx] + completions[trial_idx][0]
                ans = extract_answer(full_gen_text, dataset_type)
            else:
                ans = extracted_answers[trial_idx]
            answer_strings_for_prob.append(ans or "")

        prob_probe_prompts = []
        for trial_idx in range(num_trials):
            prompt_prefix = (
                current_prompts[trial_idx] + completions[trial_idx][0] + answer_probe_texts[trial_idx]
            )
            prob_prompt = prob_probe.replace("{ANSWER}", answer_strings_for_prob[trial_idx] or "")
            prob_probe_prompts.append(prompt_prefix + prob_prompt)

        prob_probe_responses = model.generate_batch_probe(
            prob_probe_prompts,
            max_tokens=prob_probe_tokens,
            is_actives=[not finished for finished in is_finished],
        )

        prob_probe_texts = []
        extracted_probabilities = []
        for trial_idx in range(num_trials):
            if is_finished[trial_idx] or prob_probe_responses[trial_idx] is None:
                prob_probe_texts.append("")
                extracted_probabilities.append(None)
            else:
                txt = prob_probe_responses[trial_idx].choices[0].text
                prob_probe_texts.append(txt)
                # Re-use local extractor just for probability
                _, prob_val = extract_boxed_answer_and_probability_local("Probability: " + txt, dataset_type)
                extracted_probabilities.append(prob_val)

        # -------- store in results --------
        round_results["answer_probe_prompts"] = answer_probe_prompts
        # Back-compat: keep old key name pointing at answer probe prompts
        round_results["probe_prompts"] = answer_probe_prompts
        round_results["answer_probe_responses_text"] = answer_probe_texts
        round_results["probe_extracted_answers"] = extracted_answers

        round_results["prob_probe_prompts"] = prob_probe_prompts
        round_results["prob_probe_responses_text"] = prob_probe_texts
        round_results["probe_extracted_probabilities"] = extracted_probabilities

        is_corrects = []
        is_corrects_original = []
        for trial_idx in range(num_trials):
            # Correctness based on natural stop or probe
            if is_finished[trial_idx]:
                # Answer from main generation if it finished naturally
                full_generation_text = current_prompts[trial_idx] + completions[trial_idx][0]
                finished_result = extract_answer(full_generation_text, dataset_type)
                is_corrects.append(math_equal(finished_result, target))
            else:
                # Answer from parsed probe output
                extracted_probe_answer = round_results["probe_extracted_answers"][trial_idx]
                is_corrects.append(math_equal(extracted_probe_answer, target))

            # Original correctness: based on the generation *before* probe, regardless of finish reason for this round
            is_corrects_original.append(
                math_equal(
                    extract_answer(
                        current_prompts[trial_idx] + completions[trial_idx][0], dataset_type
                    ),
                    target,
                )
            )

        round_results["is_corrects"] = is_corrects
        round_results["is_corrects_original"] = is_corrects_original

        # Save results for this round to a file
        if output_dir:
            round_filename = (
                f"{output_dir}/question_{problem_id}_tokens_{max_tokens[i]}.json"
            )
            save_json(round_results, round_filename)
        results.append(round_results) # Append to results list
    return results # Return all results for this question


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_dataset(args.dataset)

    num_trials = args.num_trials  # Number of trials per question

    import os
    from datetime import datetime

    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Create output directory with model name, dataset, parameters and date
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name_sanitized = args.model.replace("/", "-")
        output_dir = f"results/{model_name_sanitized}_{args.dataset}_step{args.step}_max{args.max_tokens}_trials{args.num_trials}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Save args to a metadata file in the output directory
    args_dict = vars(args)
    save_json(args_dict, os.path.join(output_dir, "experiment_args.json"))

    model = load_model(args.model, args.url, args.api_key)

    for problem_id, item in enumerate(data):
        if problem_id < args.start:
            continue
        if problem_id >= args.end:
            break
        # execute question
        prompt = item["problem"].strip()
        target = strip_string(item["answer"])

        print(f"Executing question {problem_id} (Dataset index {item.get('id', problem_id)}) with target [{target}]")
        print(f"Prompt: {prompt[:200]}...") # Print partial prompt
        print("-" * 100)
        token_budgets = list(range(args.step, args.max_tokens + args.step, args.step))
        # Ensure token budgets do not exceed args.max_tokens if max_tokens is not a multiple of step
        token_budgets = [tb for tb in token_budgets if tb <= args.max_tokens]
        if not token_budgets or token_budgets[-1] < args.max_tokens and args.max_tokens > 0 :
             if args.max_tokens not in token_budgets: # Add max_tokens if it's not already the last budget due to step size
                token_budgets.append(args.max_tokens)
        token_budgets = sorted(list(set(token_budgets))) # Deduplicate and sort

        execute_question_reuse( # Removed batch_results assignment as it's saved per round
            model,
            prompt,
            target,
            args.dataset, # Pass dataset type
            max_tokens=token_budgets,
            answer_probe=args.answer_probe,
            answer_probe_tokens=args.answer_probe_tokens,
            prob_probe=args.prob_probe,
            prob_probe_tokens=args.prob_probe_tokens,
            num_trials=num_trials,
            problem_id=problem_id,
            output_dir=output_dir,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    print("Saved results to", output_dir)


if __name__ == "__main__":
    main()
