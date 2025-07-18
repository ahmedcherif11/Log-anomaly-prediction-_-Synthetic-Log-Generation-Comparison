import json

input_path = "/scratch/cherif/dataset/data_fixed_shuffled_normalized.jsonl"
output_path = "/scratch/cherif/dataset/data_fixed_allfields_null.jsonl"

# 1. First pass: get the set of all fields in all records
all_fields = set()
with open(input_path, "r", encoding="utf-8") as fin:
    for line in fin:
        obj = json.loads(line)
        all_fields.update(obj.keys())

# 2. Second pass: add missing fields as null
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        # Fill missing fields with null
        for field in all_fields:
            if field not in obj:
                obj[field] = None
        fout.write(json.dumps(obj) + "\n")

print(f"Done. Wrote all records with all {len(all_fields)} fields, missing ones as null, to {output_path}")
