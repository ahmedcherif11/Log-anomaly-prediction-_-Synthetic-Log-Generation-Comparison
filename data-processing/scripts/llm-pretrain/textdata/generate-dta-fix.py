# Combines checks.py and generate-dta.py functionality

import json
import re
import argparse
from datasets import Dataset, disable_progress_bars

def fix_backslashes(s):
    return re.sub(r'(?<!\\)\\(?![\\nt"\\/bfru])', r'\\\\', s)

def process_file(input_path, output_path):
    fixed_lines = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                json.loads(line)
                fixed_lines.append(line)
            except json.JSONDecodeError:
                fixed_line = fix_backslashes(line)
                try:
                    json.loads(fixed_line)
                    fixed_lines.append(fixed_line)
                except Exception:
                    print("Line still invalid after fix:", line)
                    continue
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(fixed_lines)
    print(f"✅ All valid lines written to {output_path}")

def main(args):
    # Step 1: Validate/fix JSONL
    process_file(args.input, args.output)
    # Step 2: Load as HuggingFace Dataset and save
    disable_progress_bars()
    logs = Dataset.from_text(args.output)
    data = logs.train_test_split(test_size=0.001)
    data.save_to_disk(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix/validate JSONL and save as HuggingFace dataset.")
    parser.add_argument("--input", "-i", default="prompts.jsonl", help="Input JSONL file")
    parser.add_argument("--output", "-o", default="prompts_fixed.jsonl", help="Output JSONL file")
    parser.add_argument("--out", type=str, required=True, help="Folder to save processed dataset")
    args = parser.parse_args()
    main(args)