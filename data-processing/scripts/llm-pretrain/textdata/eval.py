import argparse
import os
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# =========================
# Config / constants
# =========================
RESPONSE_MARKER = "### Response:\n"
FIELD_KEYS = [
    "EventID", "EventId", "Provider", "ProviderName", "Channel", "Computer",
    "User", "UserName", "UserId", "RemoteName", "Payload", "PayloadData1",
    "PayloadData2", "PayloadData3", "ScriptBlockId", "TimeCreated",
    "ProcessId", "ThreadId"
]

# =========================
# Helpers
# =========================
def extract_response(text: str) -> str:
    """Return only the continuation after the response marker."""
    return text.split(RESPONSE_MARKER, 1)[-1].strip() if RESPONSE_MARKER in text else text.strip()

def extract_json_objects(s: str):
    """
    Extract all JSON objects from a string and return as list of dicts.
    Handles concatenated JSON objects and JSONL-style lines.
    """
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
        # Fallback: try line-by-line JSON
        for line in s.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    objs.append(json.loads(line))
                except Exception:
                    pass
    return objs

def _flatten_json(obj, parent=""):
    """
    Flatten nested dicts/lists into (keypath, value) pairs.
    - dict key 'a' under parent 'root' -> 'root.a'
    - list index 0 under 'arr' -> 'arr[0]'
    Only terminal scalars become values; everything else is recursed.
    """
    pairs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent}.{k}" if parent else k
            pairs.extend(_flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{parent}[{i}]"
            pairs.extend(_flatten_json(v, key))
    else:
        # scalar (int/float/bool/str/None)
        pairs.append((parent, "" if obj is None else str(obj)))
    return pairs

def _item_set(d):
    """Return a set of flattened (keypath, value) pairs for a dict-like JSON."""
    if not isinstance(d, dict):
        return set()
    return set(_flatten_json(d))

def jaccard_items(d1, d2) -> float:
    """
    Jaccard over flattened (keypath, value) pairs (handles lists/nesting).
    Fixes 'unhashable type: list' by avoiding raw dict.items().
    """
    a, b = _item_set(d1), _item_set(d2)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def fieldwise_jaccard(d1, d2, keys=FIELD_KEYS) -> float:
    """
    Jaccard over selected fields only; values are stringified.
    Works even if other parts of the JSON contain lists/dicts.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return 0.0
    a = {(k, str(d1.get(k))) for k in keys if k in d1}
    b = {(k, str(d2.get(k))) for k in d2 if k in keys}
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def ngram_overlap(s1, s2, n=2) -> float:
    """Jaccard over n-grams (default bigram) for text strings."""
    def ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    t1, t2 = s1.split(), s2.split()
    n1, n2 = ngrams(t1, n), ngrams(t2, n)
    if not n1 or not n2:
        return 0.0
    return len(n1 & n2) / len(n1 | n2)

def avg(x): 
    return (sum(x)/len(x)) if x else 0.0

# =========================
# Main
# =========================
def main(args):
    # Load model & tokenizer
    print("Loading model from", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load data (HF dataset dir or JSONL)
    print(f"Loading dataset from {args.dataset}")
    try:
        data = datasets.load_from_disk(args.dataset)
        samples = data["test"] if "test" in data else data[list(data.keys())[0]]
    except Exception:
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

        # ---------- Build inference prompt to avoid prompt echo ----------
        gen_prompt = f"### Prompt:\n{prompt.strip()}\n\n{RESPONSE_MARKER}"

        # ---------- Generate ONLY the response continuation ----------
        inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # deterministic for structured JSON
                temperature=None,
                top_p=None,               # unset sampling knobs to silence warnings
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_response = extract_response(decoded)

        # Collect basic record
        record = {"prompt": prompt, "generated": generated_response}
        if has_reference:
            record["reference"] = ref_text

        # ---------- JSON extraction (gen/ref can contain 1+ JSON objects) ----------
        gen_objs = extract_json_objects(generated_response)
        ref_objs = extract_json_objects(ref_text) if has_reference else []

        valid_json_counts.append(len(gen_objs))
        exact_two_flags.append(int(len(gen_objs) == 2))

        # ---------- Metrics (pairwise on first min(len(gen), len(ref)) pairs) ----------
        if has_reference and len(gen_objs) and len(ref_objs):
            pair_n = min(len(gen_objs), len(ref_objs))
            sample_bleu, sample_j_full, sample_j_keys, sample_bigram = [], [], [], []

            for i in range(pair_n):
                g, r = gen_objs[i], ref_objs[i]
                g_str = json.dumps(g, sort_keys=True)
                r_str = json.dumps(r, sort_keys=True)

                # BLEU on JSON strings (quick proxy)
                sample_bleu.append(
                    sentence_bleu([r_str.split()], g_str.split(),
                                  smoothing_function=SmoothingFunction().method1)
                )
                # Jaccard (flattened items) and selected-key Jaccard
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

    # ---------- Save results ----------
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved all generations and metrics to {results_path}")

    # ---------- Print aggregates ----------
    if has_reference:
        print("\n========== Evaluation Metrics (on test set) ==========")
        print(f"Average BLEU (JSON strings): {avg(bleu_scores):.4f}")
        print(f"Average Jaccard (all items): {avg(jaccard_items_scores):.4f}")
        print(f"Average Jaccard (key subset): {avg(jaccard_field_scores):.4f}")
        print(f"Average Bigram Jaccard:      {avg(bigram_scores):.4f}")
        print("\n---------- Validity ----------")
        print(f"Avg # of generated JSON objs: {avg(valid_json_counts):.2f}")
        print(f"% samples with exactly 2 JSON: {100.0 * sum(exact_two_flags)/len(exact_two_flags):.1f}%")
    else:
        print("\n[Info] No ground-truth reference responses found. Only synthetic log generation performed.")
        print(f"Avg # of generated JSON objs: {avg(valid_json_counts):.2f}")
        print(f"% samples with exactly 2 JSON: {100.0 * sum(exact_two_flags)/len(exact_two_flags):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model checkpoint directory (merged or adapters path)")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset dir or JSONL with 'prompt' and optional 'response'")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate")
    args = parser.parse_args()
    main(args)
