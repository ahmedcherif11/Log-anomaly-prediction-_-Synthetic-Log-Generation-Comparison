import json
with open("/scratch/cherif/dataset/data_fixed.jsonl") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        kw = obj.get("Keywords", None)
        if kw is not None and not isinstance(kw, str) and kw is not None:
            print(f"Line {i}: Keywords is not string or null: {kw} ({type(kw)})")
