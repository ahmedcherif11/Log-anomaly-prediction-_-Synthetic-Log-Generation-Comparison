import os
import pandas as pd
import json

# --- SETTINGS ---
input_root = r"C:\Users\AHMED\Desktop\new-approch\dataset\organized"           # Change this
output_root = r"C:\Users\AHMED\Desktop\new-approch\dataset\json"          # Change this

for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".csv"):
            input_path = os.path.join(root, file)
            
            # Create mirrored output directory
            relative_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            # Output JSONL file path
            output_filename = os.path.splitext(file)[0] + ".jsonl"
            output_path = os.path.join(output_dir, output_filename)

            try:
                df = pd.read_csv(input_path)

                with open(output_path, "w", encoding="utf-8") as f:
                    for record in df.to_dict(orient="records"):
                        json_str = json.dumps(record, separators=(",", ":"))  # no indent or extra whitespace
                        f.write(json_str + "\n")

                print(f"✅ Converted: {input_path} → {output_path}")
            except Exception as e:
                print(f"❌ Failed to convert {input_path}: {e}")
