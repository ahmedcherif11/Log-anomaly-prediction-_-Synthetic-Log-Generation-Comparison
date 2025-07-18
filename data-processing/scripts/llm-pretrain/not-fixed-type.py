import json
from collections import defaultdict

# Path to your JSONL file
filename = "/scratch/cherif/dataset/data_fixed_shuffled.jsonl"

# Dictionary to track types seen for each field
field_types = defaultdict(set)

with open(filename, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            for k, v in obj.items():
                # Use Python type name (e.g., 'int', 'str', 'null')
                if v is None:
                    field_types[k].add('null')
                else:
                    field_types[k].add(type(v).__name__)
        except Exception as e:
            print(f"Line {i}: JSON decode error: {e}")

print("Fields with multiple types detected:")
for field, types in field_types.items():
    if len(types) > 1:
        print(f"- {field}: {types}")
