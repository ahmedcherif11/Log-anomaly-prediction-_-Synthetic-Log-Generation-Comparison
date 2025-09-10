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
        technique = o.get("technique")
        if not isinstance(prompt, str) or not isinstance(technique, str):
            dropped += 1
            continue
        examples.append({"prompt": prompt, "technique": technique})
    if dropped:
        print(f"⚠️ Dropped {dropped} lines missing 'prompt' or 'technique'.")
    return examples

def main(args):
    disable_progress_bars()
    rows = load_and_fix_jsonl(args.input)
    examples = to_examples(rows)
    if not examples:
        raise RuntimeError("No valid examples to save.")
    ds = Dataset.from_list(examples)
    ds.save_to_disk(args.out)
    print("Columns:", ds.column_names)
    print("Dataset size:", len(ds))
    print("Sample:", ds[0])
    print("Number of columns:", len(ds.column_names))
    print("max length:", max(len(x["prompt"]) for x in ds))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input JSONL file")
    p.add_argument("-o", "--out", required=True, help="Output dataset folder")
    args = p.parse_args()
    main(args)