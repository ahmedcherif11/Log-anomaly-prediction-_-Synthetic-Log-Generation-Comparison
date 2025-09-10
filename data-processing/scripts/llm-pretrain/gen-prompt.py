import os
import json

# Set the root directory containing technique subfolders
root_dir = r'c:\Users\AHMED\Desktop\new-approch\dataset\prompts-one'
output_file = os.path.join(root_dir, 'all_prompts.jsonl')

with open(output_file, 'w', encoding='utf-8') as out_f:
    for technique in os.listdir(root_dir):
        tech_path = os.path.join(root_dir, technique)
        if os.path.isdir(tech_path):
            for fname in os.listdir(tech_path):
                if fname.endswith('.txt'):
                    file_path = os.path.join(tech_path, fname)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    obj = {
                        "technique": technique,
                        "prompt": prompt
                    }
                    out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"Done! Output written to {output_file}")