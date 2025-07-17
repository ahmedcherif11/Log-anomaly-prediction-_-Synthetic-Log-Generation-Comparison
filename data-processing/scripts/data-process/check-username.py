import json

path = r"/scratch/cherif/dataset/data_fixed.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            if "UserName" in obj:
                if not isinstance(obj["UserName"], str):
                    print(f"Line {idx}: UserName is not string:", obj["UserName"])
        except Exception as e:
            print(f"Line {idx}: JSON decode error: {e}")
