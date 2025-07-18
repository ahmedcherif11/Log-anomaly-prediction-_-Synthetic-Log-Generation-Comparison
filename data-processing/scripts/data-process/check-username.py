import json
with open("/scratch/cherif/dataset/data_fixed.jsonl") as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        kw = obj.get("Keywords", None)
        if kw is not None and not isinstance(kw, str) and kw is not None:
            print(f"Line {i}: Keywords is not string or null: {kw} ({type(kw)})")

# If you want to check for other fields, you can add them here
FIELDS_TO_CHECK = ["Task", "UserName", "MapDescription", "RemoteHost"]  
for field in FIELDS_TO_CHECK:
    with open("/scratch/cherif/dataset/data_fixed_normalized.jsonl") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            val = obj.get(field, None)
            if val is not None and not isinstance(val, str) and val is not None:
                print(f"Line {i}: {field} is not string or null: {val} ({type(val)})")