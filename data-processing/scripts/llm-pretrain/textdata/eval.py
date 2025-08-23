import argparse
import os
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# -------- Helpers --------
RESPONSE_MARKER = "### Response:\n"

# Keys we care about most for fieldwise Jaccard (extend as needed)
FIELD_KEYS = [
    "EventID", "EventId", "Provider", "ProviderName", "Channel", "Computer",
    "User", "UserName", "UserId", "RemoteName", "Payload", "PayloadData1",
    "PayloadData2", "PayloadData3", "ScriptBlockId", "TimeCreated",
    "ProcessId", "ThreadId"
]

def extract_response(text: str) -> str:
    """Return only the continuation after the response marker."""
    if RESPONSE_MARKER in text:
        return text.split(RESPONSE_MARKER, 1)[-1].strip()
    return text.strip()

def extract_json_objects(s: str):
    """Extract all JSON objects from a string and return as list of dicts."""
    objs, stack, start = [], 0, -1
    for i, ch in enumerate(s):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    try:
                        objs.append(json.loads(s[start:i+1]))
                    except Exception:
                        pass
                    start = -1
    if not objs:
        # Fallback: each line may be a standalone JSON object
        for line in s.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    objs.append(json.loads(line))
                except Exception:
                    pass
    return objs

def jaccard_items(d1, d2) -> float:
    """Jaccard over (key,value) items for two dicts."""
    if not d1 or not d2:
        return 0.0
    a, b = set(d1.items()), set(d2.items())
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)

def fieldwise_jaccard(d1, d2, keys=FIELD_KEYS) -> float:
    """Jaccard over selected fields (key,value) pairs."""
    if not d1 or not d2:
        return 0.0
    a = {(k, str(d1.get(k))) for k in keys if k in d1}
    b = {(k, str(d2.get(k))) for k in keys if k in d2}
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)

def ngram_overlap(s1, s2, n=2) -> float:
    """Jaccard over n-grams (default bigram) for text strings."""
    def ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    t1, t2 = s1.split(), s2.split()
    n1, n2 = ngrams(t1, n), ngrams(t2, n)
    if not n1 or not n2:
        return 0.0
    return len(n1 & n2) / len(n1 | n2)

# -------- Main --------
def main(args):
    # Load model & tokenizer
    print("Loading model from", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # Load data (can be a Hugging Face dataset dir or a JSONL file with prompts)
    print(f"Loading dataset from {args.dataset}")
    try:
        data = datasets.load_from_disk(args.dataset)
        # Try 'test' split, else fall back to full dataset
        if "test" in data:
            samples = data["test"]
        else:
            samples = data[list(data.keys())[0]]
    except Exception:
        # Try to load as plain JSONL with list of dicts with 'prompt' key
        with open(args.dataset, "r") as f:
            samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} samples.")
    has_reference = "response" in samples[0]

    # Metric accumulators
    bleu_scores, jaccard_items_scores, jaccard_field_scores, bigram_scores = [], [], [], []
    valid_json_counts, exact_two_flags = [], []

    results = []

    for ex in tqdm(samples):
        prompt = ex["prompt"]
        ref_text = ex.get("response", "")

        # ---- Build inference prompt to avoid echo ----
        gen_prompt = f"### Prompt:\n{prompt.strip()}\n\n{RESPONSE_MARKER}"

        # ---- Generate ONLY the response continuation ----
        inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,               # deterministic for structured JSON
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_response = extract_response(decoded)

        # Collect
        record = {"prompt": prompt, "generated": generated_response}
        if has_reference:
            record["reference"] = ref_text

        # ---- JSON extraction (gen/ref can contain 1+ JSON objects) ----
        gen_objs = extract_json_objects(generated_response)
        ref_objs = extract_json_objects(ref_text) if has_reference else []

        valid_json_counts.append(len(gen_objs))
        exact_two_flags.append(int(len(gen_objs) == 2))

        # ---- Metrics (pairwise on first min(len(gen), len(ref)) pairs) ----
        if has_reference and len(gen_objs) and len(ref_objs):
            pair_n = min(len(gen_objs), len(ref_objs))
            sample_bleu, sample_j_full, sample_j_keys, sample_bigram = [], [], [], []

            for i in range(pair_n):
                g, r = gen_objs[i], ref_objs[i]
                g_str = json.dumps(g, sort_keys=True)
                r_str = json.dumps(r, sort_keys=True)

                # BLEU on JSON strings (optional but quick)
                sample_bleu.append(
                    sentence_bleu([r_str.split()], g_str.split(),
                                  smoothing_function=SmoothingFunction().method1)
                )
                # Jaccard (all items) and fieldwise
                sample_j_full.append(jaccard_items(r, g))
                sample_j_keys.append(fieldwise_jaccard(r, g, FIELD_KEYS))
                # Bigram overlap on JSON strings
                sample_bigram.append(ngram_overlap(r_str, g_str, n=2))

            # Aggregate per-sample
            if sample_bleu:   bleu_scores.append(sum(sample_bleu)/len(sample_bleu))
            if sample_j_full: jaccard_items_scores.append(sum(sample_j_full)/len(sample_j_full))
            if sample_j_keys: jaccard_field_scores.append(sum(sample_j_keys)/len(sample_j_keys))
            if sample_bigram: bigram_scores.append(sum(sample_bigram)/len(sample_bigram))

        results.append({
            **record,
            "gen_json_count": len(gen_objs),
            "ref_json_count": len(ref_objs) if has_reference else None
        })

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved all generations and metrics to {results_path}")

    # ---- Print aggregates ----
    if has_reference:
        def avg(x): return (sum(x)/len(x)) if x else 0.0
        print("\n========== Evaluation Metrics (on test set) ==========")
        print(f"Average BLEU (JSON strings): {avg(bleu_scores):.4f}")
        print(f"Average Jaccard (all items): {avg(jaccard_items_scores):.4f}")
        print(f"Average Jaccard (key subset): {avg(jaccard_field_scores):.4f}")
        print(f"Average Bigram Jaccard:      {avg(bigram_scores):.4f}")
        print("\n---------- Validity ----------")
        print(f"Avg # of generated JSON objs: {avg(valid_json_counts):.2f}")
        print(f"% samples with exactly 2 JSON: {100.0 * sum(exact_two_flags)/len(exact_two_flags):.1f}")
    else:
        print("\n[Info] No ground-truth reference responses found. Only synthetic log generation performed.")
        print(f"Avg # of generated JSON objs: { (sum(valid_json_counts)/len(valid_json_counts)) if valid_json_counts else 0:.2f}")
        print(f"% samples with exactly 2 JSON: { (100.0 * sum(exact_two_flags)/len(exact_two_flags)) if exact_two_flags else 0:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model checkpoint directory (merged or adapters path)")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset dir or JSONL file with 'prompt' and optional 'response'")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate")
    args = parser.parse_args()
    main(args)
