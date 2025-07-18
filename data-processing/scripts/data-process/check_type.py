import json
from collections import defaultdict, Counter

input_path = "/scratch/cherif/dataset/data_fixed_allfields_null.jsonl"  # <-- Change this to your file path

# Initialize stats for each field
field_stats = defaultdict(Counter)

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        for k, v in obj.items():
            if v is None:
                field_stats[k]["null"] += 1
            elif isinstance(v, str):
                field_stats[k]["str"] += 1
            elif isinstance(v, int):
                field_stats[k]["int"] += 1
            elif isinstance(v, float):
                field_stats[k]["float"] += 1
            elif isinstance(v, bool):
                field_stats[k]["bool"] += 1
            elif isinstance(v, list):
                field_stats[k]["list"] += 1
            elif isinstance(v, dict):
                field_stats[k]["dict"] += 1
            else:
                field_stats[k]["other"] += 1

# Print stats
for field, stats in field_stats.items():
    print(f"Field: {field}")
    for t, count in stats.items():
        print(f"  {t}: {count}")
    print("-" * 20)
