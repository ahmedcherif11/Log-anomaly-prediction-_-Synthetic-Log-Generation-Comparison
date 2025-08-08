import os
import json
import glob

# Folder containing .txt files (recursively)
input_folder = r"C:\Users\AHMED\Desktop\new-approch\dataset\prompts"      # change this to your folder
output_file = r"C:\Users\AHMED\Desktop\new-approch\dataset\all-prompts.jsonl"   # output path

# Get all .txt files in the folder and subfolders, sorted for reproducibility
txt_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.txt"), recursive=True))
if not txt_files:
    raise ValueError("No .txt files found in the specified folder or its subfolders.")

with open(output_file, "w", encoding="utf-8") as fout:
    for idx, txt_path in enumerate(txt_files, 1):
        with open(txt_path, "r", encoding="utf-8") as fin:
            content = fin.read().strip()
            obj = {"prompt": content}
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")
        if idx % 100 == 0:
            print(f"Processed {idx} prompts...")

print(f"Converted {len(txt_files)} prompts to {output_file}")
