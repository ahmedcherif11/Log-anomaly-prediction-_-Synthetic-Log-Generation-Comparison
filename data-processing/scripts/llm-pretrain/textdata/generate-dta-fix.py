# build_dataset_two_cols.py
import json
import re
import argparse
from datasets import Dataset, disable_progress_bars

def fix_backslashes(s: str) -> str:
    # Escape lone backslashes not part of valid JSON escapes
    return re.sub(r'(?<!\\)\\(?![\\nt"\\/bfru])', r'\\\\', s)

def load_and_fix_jsonl(path: str):
    fixed = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    obj = json.loads(fix_backslashes(line))
                except Exception:
                    bad += 1
                    print(f"⚠️ Skipping invalid line {i}")
                    continue
            fixed.append(obj)
    print(f"✅ Parsed {len(fixed)} lines. Skipped {bad} invalid lines.")
    return fixed

def to_examples(objs):
    examples = []
    dropped = 0
    for o in objs:
        prompt = o.get("prompt")
        response = o.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            dropped += 1
            continue
        examples.append({"prompt": prompt, "response": response})
    if dropped:
        print(f"⚠️ Dropped {dropped} lines missing 'prompt' or 'response'.")
    return examples

def main(args):
    disable_progress_bars()

    # Step 1: Read + fix JSONL
    rows = load_and_fix_jsonl(args.input)

    # Step 2: Keep only prompt & response
    examples = to_examples(rows)
    if not examples:
        raise RuntimeError("No valid examples to save.")

    ds = Dataset.from_list(examples)
    split = ds.train_test_split(test_size=args.test_size, seed=42, shuffle=True)
    split.save_to_disk(args.out)

    print("Columns:", split["train"].column_names)
    print("Train size:", len(split["train"]))
    print("Test size:", len(split["test"]))
    print("Sample:", split["train"][0])
    print("Number of columns:", len(split["train"].column_names))
    print("max length:", split["train"].map(lambda x: len(x["prompt"] + x["response"]), batched=True).max())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input JSONL file")
    p.add_argument("-o", "--out", required=True, help="Output dataset folder")
    p.add_argument("--test-size", type=float, default=0.09, help="Holdout fraction (default 0.09)")
    args = p.parse_args()
    main(args)
