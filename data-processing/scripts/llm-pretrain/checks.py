import json
import re
import argparse

def fix_backslashes(s):
    # Replace single backslashes with double, but avoid already doubled ones
    # This regex finds single backslashes not already part of a double
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
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except Exception as e:
                print(f"Invalid JSON at line {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix and validate JSONL files.")
    parser.add_argument("--input", "-i", default="prompts.jsonl", help="Input JSONL file")
    parser.add_argument("--output", "-o", default="prompts_fixed.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    process_file(args.input, args.output)
