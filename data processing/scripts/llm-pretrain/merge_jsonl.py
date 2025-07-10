import argparse
from pathlib import Path
import json
import sys

def write_log(obj, out_f):
    """Write a log object or JSON string as a line to output."""
    if isinstance(obj, str):
        out_f.write(obj.strip() + '\n')
    else:
        out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def merge_logs(input_dir, output_file):
    root = Path(input_dir)
    count = 0
    files = list(root.glob("**/*"))
    log_files = [f for f in files if f.suffix.lower() in [".jsonl", ".json"]]
    total_files = len(log_files)
    print(f"Found {total_files} files to process.")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, log_path in enumerate(log_files, 1):
            if not log_path.exists():
                print(f"File not found, skipping: {log_path}")
                continue
            print(f"\rProcessing ({idx}/{total_files}): {log_path}", end="")
            try:
                if log_path.suffix.lower() == ".jsonl":
                    with open(log_path, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            if line.strip():
                                write_log(line, out_f)
                                count += 1
                elif log_path.suffix.lower() == ".json":
                    with open(log_path, "r", encoding="utf-8") as in_f:
                        try:
                            obj = json.load(in_f)
                            if isinstance(obj, dict):
                                write_log(obj, out_f)
                                count += 1
                            elif isinstance(obj, list):
                                for entry in obj:
                                    write_log(entry, out_f)
                                    count += 1
                        except json.JSONDecodeError as e:
                            if "Extra data" in str(e):
                                # Fallback: treat as JSON Lines!
                                in_f.seek(0)
                                for line in in_f:
                                    if line.strip():
                                        try:
                                            write_log(json.loads(line), out_f)
                                            count += 1
                                        except Exception:
                                            pass
                            else:
                                print(f"\nWarning: Skipping {log_path} due to error: {e}")
            except Exception as e:
                print(f"\nWarning: Skipping {log_path} due to error: {e}")

    print(f"\nDone. Merged {count} logs to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all .jsonl/.json logs into one .jsonl file recursively.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory with log files")
    parser.add_argument("-o", "--output_file", required=True, help="Path for merged output .jsonl file")
    args = parser.parse_args()
    merge_logs(args.input_dir, args.output_file)
