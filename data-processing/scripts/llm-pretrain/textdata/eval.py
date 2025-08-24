import argparse
import os
import json
import re
from tqdm import tqdm

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# =========================
# Config / constants
# =========================
RESPONSE_MARKER = "### Response:\n"

# Fields that matter most for Windows logs; extend as needed
FIELD_KEYS = [
    "EventID", "EventId", "Provider", "ProviderName", "Channel", "Computer",
    "Hostname", "User", "UserName", "UserId", "RemoteName", "Message",
    "Payload", "PayloadData1", "PayloadData2", "PayloadData3",
    "ScriptBlockId", "TimeCreated", "UtcTime", "@timestamp",
    "ProcessGuid", "ProcessId", "ThreadId", "Image", "TargetFilename", "Task"
]

# Normalize common key aliases to improve matching
KEY_ALIASES = {
    "EventId": "EventID",
    "Provider": "ProviderName",
    "Host": "Hostname",
    "User": "UserName",
    "UserID": "UserId"
}

SPECIAL_TOKEN_RE = re.compile(r"<\|[^>|]+\|>")  # e.g., <|eos|>, <|endoftext|>

# Fix a frequent typo: ProcessGuid ends with ")" instead of "}"
GUID_PAREN_FIX_RE = re.compile(
    r'("ProcessGuid"\s*:\s*"\{[0-9a-fA-F\-]{8}\-[0-9a-fA-F\-]{4}\-[0-9a-fA-F\-]{4}\-[0-9a-fA-F\-]{4}\-[0-9a-fA-F\-]{12})\)\s*"',
    flags=re.IGNORECASE
)

# =========================
# Helpers
# =========================
def avg(x): 
    return (sum(x)/len(x)) if x else 0.0

def extract_response(text: str) -> str:
    """Return only the continuation after the response marker."""
    return text.split(RESPONSE_MARKER, 1)[-1].strip() if RESPONSE_MARKER in text else text.strip()

def clean_generated_text(s: str) -> str:
    """
    1) Remove special tokens like <|eos|>.
    2) Trim whitespace.
    3) If the whole thing is one quoted string containing escaped JSON, unquote it.
    """
    s = SPECIAL_TOKEN_RE.sub("", s).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and "\\{" in s and "\\}" in s:
        try:
            s = json.loads(s)
        except Exception:
            s = s[1:-1].replace('\\"', '"').replace('\\\\', '\\')
    return s

def repair_common_json_issues(s: str) -> str:
    """
    Targeted fixes for frequent, small mistakes:
    - ProcessGuid ends with ')' instead of '}'.
    - Add newline between back-to-back objects '}{' to help parsing.
    """
    s = GUID_PAREN_FIX_RE.sub(r'\1}"', s)
    s = s.replace('}{', '}\n{')
    return s

def truncate_after_n_json(s: str, n: int = 2) -> str:
    """Return substring containing exactly the first n well-formed JSON objects (if present)."""
    count, stack, start = 0, 0, -1
    for i, ch in enumerate(s):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    count += 1
                    if count == n:
                        return s[:i+1]
    return s  # fewer than n -> return as-is

def salvage_longest_balanced_json(s: str):
    """
    Return a list with the longest balanced JSON object we can find (if any).
    This rescues cases where the generation ends mid-object.
    """
    objs = []
    if '{' in s and '}' in s and s.index('{') < s.rindex('}'):
        candidate = s[s.index('{'): s.rindex('}')+1]
        try:
            objs.append(json.loads(candidate))
        except Exception:
            pass
    return objs

def extract_json_objects_robust(s: str):
    """
    Robust extractor:
    - Clean & repair first
    - Try normal multi-object extraction
    - Fall back to JSONL lines
    - Last resort: salvage the longest balanced object
    """
    s = repair_common_json_issues(clean_generated_text(s))

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

    if objs:
        return objs

    # JSONL-style fallback
    for line in s.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                objs.append(json.loads(line))
            except Exception:
                pass
    if objs:
        return objs

    # Last resort: salvage a single longest balanced object
    return salvage_longest_balanced_json(s)

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
        pairs.append((parent, "" if obj is None else str(obj)))
    return pairs

def _item_set(d):
    """Return a set of flattened (keypath, value) pairs for a dict-like JSON."""
    if not isinstance(d, dict):
        return set()
    return set(_flatten_json(d))

def jaccard_items(d1, d2) -> float:
    """Jaccard over flattened (keypath, value) pairs (handles lists/nesting)."""
    a, b = _item_set(d1), _item_set(d2)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def normalize_keys(x):
    """Recursively normalize key names using KEY_ALIASES."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            kk = KEY_ALIASES.get(k, k)
            out[kk] = normalize_keys(v)
        return out
    if isinstance(x, list):
        return [normalize_keys(i) for i in x]
    return x

def fieldwise_jaccard(d1, d2, keys=FIELD_KEYS) -> float:
    """Jaccard over selected fields only; values are stringified."""
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

# =========================
# Main
# =========================
def main(args):
    # Guard / device selection
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    device_map = "auto" if cuda_ok else {"": "cpu"}

    # Load model & tokenizer
    print("Loading model from", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if cuda_ok else torch.float32,
        device_map=device_map
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
    bleu_scores, jacc_full_scores, jacc_field_scores, bigram_scores = [], [], [], []
    valid_json_counts, exact_two_flags = [], []

    results = []

    for ex in tqdm(samples):
        prompt = ex["prompt"]
        ref_text = ex.get("response", "")

        # Strong instruction to prevent chatter and quoting
        instruction_tail = (
            "Output exactly two JSON objects, one per line. "
            "Do not include any other text. Do not wrap JSON in quotes."
        )
        gen_prompt = f"### Prompt:\n{prompt.strip()}\n\n{instruction_tail}\n\n{RESPONSE_MARKER}"

        # Generate ONLY the response continuation
        inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # deterministic for structured JSON
                temperature=None,
                repetition_penalty=1.1,
                no_repeat_ngram_size=6,
                num_beams=3, 
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_response = extract_response(decoded)

        # Hard stop after 2 objects (prevents trailing chatter)
        generated_response = truncate_after_n_json(generated_response, n=2)

        # First parse attempt (robust)
        gen_objs = extract_json_objects_robust(generated_response)

        # Retry once with more tokens if we found < 2 objects
        if len(gen_objs) < 2:
            bigger = max(args.max_new_tokens * 2, args.max_new_tokens + 256)
            with torch.no_grad():
                out2 = model.generate(
                    **inputs,
                    max_new_tokens=bigger,
                    do_sample=False,
                    temperature=None,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=6,
                    num_beams=3, 
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            dec2 = tokenizer.decode(out2[0], skip_special_tokens=True)
            resp2 = extract_response(dec2)
            resp2 = truncate_after_n_json(resp2, n=2)
            gen_objs2 = extract_json_objects_robust(resp2)
            if len(gen_objs2) > len(gen_objs):
                generated_response = resp2
                gen_objs = gen_objs2

        valid_json_counts.append(len(gen_objs))
        exact_two_flags.append(int(len(gen_objs) == 2))

        record = {"generated": generated_response, "gen_json_count": len(gen_objs) }

        # Metrics (pairwise on first min(len(gen), len(ref)) pairs)
        if has_reference:
            record["reference"] = ref_text
            ref_objs = extract_json_objects_robust(ref_text)
            record["ref_json_count"] = len(ref_objs)

            if len(gen_objs) and len(ref_objs):
                pair_n = min(len(gen_objs), len(ref_objs))
                s_bleu, s_jfull, s_jfield, s_ngram = [], [], [], []

                for i in range(pair_n):
                    g = normalize_keys(gen_objs[i])
                    r = normalize_keys(ref_objs[i])

                    g_str = json.dumps(g, sort_keys=True)
                    r_str = json.dumps(r, sort_keys=True)

                    s_bleu.append(
                        sentence_bleu([r_str.split()], g_str.split(),
                                      smoothing_function=SmoothingFunction().method1)
                    )
                    s_jfull.append(jaccard_items(r, g))
                    s_jfield.append(fieldwise_jaccard(r, g, FIELD_KEYS))
                    s_ngram.append(ngram_overlap(r_str, g_str, n=2))

                if s_bleu:   bleu_scores.append(avg(s_bleu))
                if s_jfull:  jacc_full_scores.append(avg(s_jfull))
                if s_jfield: jacc_field_scores.append(avg(s_jfield))
                if s_ngram:  bigram_scores.append(avg(s_ngram))

        results.append(record)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved all generations and metrics to {results_path}")

    # Print aggregates
    if has_reference:
        print("\n========== Evaluation Metrics (on test set) ==========")
        print(f"Average BLEU (JSON strings): {avg(bleu_scores):.4f}")
        print(f"Average Jaccard (all items): {avg(jacc_full_scores):.4f}")
        print(f"Average Jaccard (key subset): {avg(jacc_field_scores):.4f}")
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
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model (merged or adapters path)")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset dir or JSONL with 'prompt' and optional 'response'")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write eval_results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Token budget for two logs (bump as needed)")
    args = parser.parse_args()
    main(args)
