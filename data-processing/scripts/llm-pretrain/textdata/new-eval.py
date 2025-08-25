import argparse, os, json, re
from collections import Counter
from tqdm import tqdm

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher

RESPONSE_MARKER = "### Response:\n"

# ---------------------------
# Text helpers / metrics
# ---------------------------
def avg(xs):
    return (sum(xs) / max(1, len(xs)))

def extract_response(text: str) -> str:
    return text.split(RESPONSE_MARKER, 1)[-1].strip() if RESPONSE_MARKER in text else text.strip()

def tokenize_simple(s: str):
    # lightweight tokenization good for log-like text
    return re.findall(r"[A-Za-z0-9_.:\\/\\\\-]+|[\{\}\[\]\(\):,\"'=]|\\n|\\r\\n", s)

def bigram_jaccard(a: str, b: str) -> float:
    def bigrams(tokens):
        return set(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))
    ta, tb = tokenize_simple(a), tokenize_simple(b)
    Ba, Bb = bigrams(ta), bigrams(tb)
    if not Ba or not Bb:
        return 0.0
    return len(Ba & Bb) / len(Ba | Bb)

def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def length_ratio(gen: str, ref: str) -> float:
    return (len(gen) + 1e-9) / (len(ref) + 1e-9)

def repetition_rate(gen: str, n: int = 3) -> float:
    """
    Fraction of repeated n-grams (how loopy the text is).
    0.0 = no repeats, higher = more repetition.
    """
    toks = tokenize_simple(gen)
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    c = Counter(grams)
    # count repeats beyond first occurrence
    repeats = sum(cnt for _, cnt in c.items() if cnt > 1) - len([1 for _, cnt in c.items() if cnt > 1])
    total = max(1, len(grams))
    return max(0.0, repeats / total)

# ---------------------------
# Indicator extraction
# ---------------------------
RX = {
    "eventid": re.compile(r"\bEventID\b[^0-9]{0,5}([0-9]{1,5})", re.I),
    "provider_full": re.compile(r"Microsoft-Windows-[A-Za-z0-9\-]+(?:/[A-Za-z0-9\-]+)?", re.I),
    "channel": re.compile(r"(Security|System|Application|Microsoft-Windows-[A-Za-z0-9\-]+/[A-Za-z0-9\-]+)", re.I),
    "username_dom": re.compile(r"\b([A-Za-z0-9_\-\.]+\\[A-Za-z0-9_\-\.]+)\b"),
    "username_kv": re.compile(r"\b(User|UserName|SubjectUserName)\b[^A-Za-z0-9\\]{0,6}([A-Za-z0-9_\-\.\\]+)", re.I),
    "proc_image": re.compile(r"[A-Za-z]:\\[^\s\"']+\.(?:exe|dll|sys|ps1|bat|vbs|msc|msi|cmd)", re.I),
    "path_any": re.compile(r"[A-Za-z]:\\[^\s\"']+", re.I),
    "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    "timestamp": re.compile(r"\b[12][0-9]{3}-[01][0-9]-[0-3][0-9][T ](?:[0-2][0-9]:[0-5][0-9]:[0-5][0-9](?:\.[0-9]{1,3})?Z?)", re.I),
    "tech_id": re.compile(r"\bT[0-9]{4}(?:\.[0-9]{3})?\b", re.I),
}

def extract_indicators(s: str):
    found = {}
    found["eventid"] = set(m.group(1) for m in RX["eventid"].finditer(s))
    found["provider_full"] = set(m.group(0).lower() for m in RX["provider_full"].finditer(s))
    found["channel"] = set(m.group(0).lower() for m in RX["channel"].finditer(s))
    dom_users = set(m.group(1).lower() for m in RX["username_dom"].finditer(s))
    kv_users = set(m.group(2).lower() for m in RX["username_kv"].finditer(s))
    found["username"] = dom_users | kv_users
    found["proc_image"] = set(m.group(0).lower() for m in RX["proc_image"].finditer(s))
    found["path_any"] = set(m.group(0).lower() for m in RX["path_any"].finditer(s))
    found["ipv4"] = set(m.group(0) for m in RX["ipv4"].finditer(s))
    found["timestamp"] = set(m.group(0) for m in RX["timestamp"].finditer(s))
    found["tech_id"] = set(m.group(0).upper() for m in RX["tech_id"].finditer(s))
    return found

def set_pr(recall_set, precision_set):
    # returns (precision, recall, f1)
    tp = len(recall_set & precision_set)
    p = tp / max(1, len(precision_set))
    r = tp / max(1, len(recall_set))
    f1 = 2*p*r / max(1e-9, (p+r)) if (p+r) else 0.0
    return p, r, f1

def indicator_scores(gen: str, ref: str):
    g = extract_indicators(gen)
    r = extract_indicators(ref)
    keys = ["eventid","provider_full","channel","username","proc_image","path_any","ipv4","timestamp","tech_id"]
    per = {}
    for k in keys:
        p, rc, f1 = set_pr(r[k], g[k])
        per[k] = {"precision": p, "recall": rc, "f1": f1, "ref_n": len(r[k]), "gen_n": len(g[k])}
    mp = avg([per[k]["precision"] for k in keys])
    mr = avg([per[k]["recall"] for k in keys])
    mf1 = avg([per[k]["f1"] for k in keys])
    return per, {"macro_precision": mp, "macro_recall": mr, "macro_f1": mf1}

# ---------------------------
# Counting + truncation
# ---------------------------
def count_top_level_json(text: str) -> int:
    """Count top-level {...} blocks (ignores braces inside strings)."""
    count, depth, in_str, esc = 0, 0, False, False
    for ch in text:
        if ch == '"' and not esc:
            in_str = not in_str
        if ch == '\\' and not esc:
            esc = True
            continue
        esc = False
        if not in_str:
            if ch == '{':
                if depth == 0:
                    count += 1
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
    return count

def truncate_after_n_json(text: str, n: int = 2) -> str:
    """Return the prefix containing the first n balanced top-level JSON objects."""
    depth, in_str, esc, started = 0, False, False, False
    count, end_idx = 0, None
    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str
        if ch == '\\' and not esc:
            esc = True
            continue
        esc = False
        if not in_str:
            if ch == '{':
                if depth == 0:
                    started = True
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                if started and depth == 0:
                    count += 1
                    if count == n:
                        end_idx = i + 1
                        break
    return text if end_idx is None else text[:end_idx]

# ---------------------------
# Full-response regeneration (no per-log)
# ---------------------------
def regenerate_two_logs_joint(model, tokenizer, prompt, max_new_tokens):
    """
    Retry generating BOTH logs together with progressively stronger settings.
    Returns best text observed (may still be <2 logs as a last resort).
    """
    # Stronger anti-boilerplate instruction for retries
    tail_variants = [
        # pass 1 (deterministic, concise)
        "Output exactly two Windows event logs in JSON, one per line. No explanations. Keep Message to one sentence.",
        # pass 2 (allow light sampling)
        "Output exactly two Windows event logs in JSON, one per line. No explanations. Avoid long GUID/hash sequences. Keep each value concise.",
        # pass 3 (even tighter)
        "Output exactly two Windows event logs in JSON, one per line. No explanations. Do not include documentation or multi-paragraph text. Limit strings to short values.",
    ]

    decode_cfgs = [
        dict(do_sample=False, num_beams=3, repetition_penalty=1.15, no_repeat_ngram_size=8),
        dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2, no_repeat_ngram_size=8),
        dict(do_sample=True, temperature=0.6, top_p=0.85, top_k=100, repetition_penalty=1.25, no_repeat_ngram_size=10),
        # final deterministic try with more beams
        dict(do_sample=False, num_beams=5, repetition_penalty=1.15, no_repeat_ngram_size=8),
    ]

    best_text = ""
    best_count = -1

    for tail in tail_variants:
        gen_prompt = f"### Prompt:\n{prompt.strip()}\n\n{tail}\n\n{RESPONSE_MARKER}"
        inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        for cfg in decode_cfgs:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **cfg
                )
            dec = tokenizer.decode(out[0], skip_special_tokens=True)
            txt = truncate_after_n_json(extract_response(dec), n=2)
            cnt = count_top_level_json(txt)
            if cnt > best_count:
                best_count = cnt
                best_text = txt
            if cnt >= 2:
                return txt

        # also try a slightly larger budget before switching instruction
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens * 1.5),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.25,
                no_repeat_ngram_size=10,
            )
        dec = tokenizer.decode(out[0], skip_special_tokens=True)
        txt = truncate_after_n_json(extract_response(dec), n=2)
        cnt = count_top_level_json(txt)
        if cnt > best_count:
            best_count = cnt
            best_text = txt
        if cnt >= 2:
            return txt

    return best_text  # best effort

# ---------------------------
# Main
# ---------------------------
def main(args):
    cuda_ok = torch.cuda.is_available()
    device_map = "auto" if cuda_ok else {"": "cpu"}
    print(f"CUDA available: {cuda_ok}")
    print(f"Using device map: {device_map}")

    print("Loading model from", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if cuda_ok else torch.float32,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    print(f"Loading dataset from {args.dataset}")
    try:
        data = datasets.load_from_disk(args.dataset)
        samples = data["test"] if "test" in data else data[list(data.keys())[0]]
    except Exception:
        with open(args.dataset, "r") as f:
            samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} samples.")
    has_reference = "response" in samples[0]

    # Accumulators
    bleu_scores, bigram_scores, char_sims = [], [], []
    len_ratios, rep_rates = [], []
    macro_p, macro_r, macro_f1 = [], [], []
    valid_counts, exact_two_flags = [], []

    results = []

    for ex in tqdm(samples):
        prompt = ex["prompt"]
        ref_text = ex.get("response", "")

        tail = (
            "Output exactly two Windows event logs in JSON, one per line. "
            "Treat values as plain text; do not add explanations."
        )
        gen_prompt = f"### Prompt:\n{prompt.strip()}\n\n{tail}\n\n{RESPONSE_MARKER}"

        inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.1,
                no_repeat_ngram_size=6,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        gen_text = extract_response(decoded)

        # Hard cut after the 2nd top-level JSON (prevents trailing chatter)
        gen_text = truncate_after_n_json(gen_text, n=2)

        # If we got <2 logs, regenerate BOTH logs together (no per-log)
        if count_top_level_json(gen_text) < 2:
            gen_text2 = regenerate_two_logs_joint(
                model, tokenizer,
                prompt,
                max_new_tokens=min(640, max(256, args.max_new_tokens))  # small budget bump allowed
            )
            if count_top_level_json(gen_text2) > count_top_level_json(gen_text):
                gen_text = gen_text2

        cnt = count_top_level_json(gen_text)
        valid_counts.append(cnt)
        exact_two_flags.append(int(cnt == 2))

        rec = {"generated": gen_text, "json_like_count": cnt}
        if has_reference:
            rec["reference"] = ref_text

            # --- Text metrics ---
            bleu = sentence_bleu([ref_text.split()], gen_text.split(),
                                 smoothing_function=SmoothingFunction().method1)
            bleu_scores.append(bleu)

            bigram_scores.append(bigram_jaccard(ref_text, gen_text))
            char_sims.append(char_similarity(ref_text, gen_text))
            len_ratios.append(length_ratio(gen_text, ref_text))
            rep_rates.append(repetition_rate(gen_text, n=3))

            # --- Indicator metrics ---
            per, macro = indicator_scores(gen_text, ref_text)
            macro_p.append(macro["macro_precision"])
            macro_r.append(macro["macro_recall"])
            macro_f1.append(macro["macro_f1"])
            rec["indicator_macro"] = macro
            rec["indicator_per_type"] = per

        results.append(rec)

    # Save per-sample
    os.makedirs(args.output_dir, exist_ok=True)
    out_p = os.path.join(args.output_dir, "eval_text_results.jsonl")
    with open(out_p, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved all generations and metrics to {out_p}")

    # Print aggregates
    if has_reference:
        print("\n========== Text-based Metrics ==========")
        print(f"Average BLEU:                 {avg(bleu_scores):.4f}")
        print(f"Average Bigram Jaccard:       {avg(bigram_scores):.4f}")
        print(f"Average Char Similarity:      {avg(char_sims):.4f}")
        print(f"Average Length Ratio (gen/ref){avg(len_ratios):.3f}  (target ~1.0)")
        print(f"Average Repetition Rate (3g): {avg(rep_rates):.3f}  (lower is better)")
        print("\n------ Indicator Macro Scores (content overlap) ------")
        print(f"Macro Precision:              {avg(macro_p):.4f}")
        print(f"Macro Recall:                 {avg(macro_r):.4f}")
        print(f"Macro F1:                     {avg(macro_f1):.4f}")
    else:
        print("\n[Info] No references found. Only text generation logged.")

    print("\n---------- Validity (counted as top-level JSON blocks) ----------")
    print(f"Avg # of logs in output:      {avg(valid_counts):.2f}")
    print(f"% samples with exactly 2 logs: {100.0 * sum(exact_two_flags)/max(1,len(exact_two_flags)):.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()
    main(args)
