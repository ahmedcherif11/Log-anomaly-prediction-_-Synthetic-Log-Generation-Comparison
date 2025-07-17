import json

fixed = []
with open("/scratch/cherif/dataset/data.jsonl", "r", encoding="utf-8") as fin:
    for idx, line in enumerate(fin, 1):
        try:
            obj = json.loads(line)
            if "UserName" in obj:
                # Replace nan/NaN/None with ""
                if str(obj["UserName"]).lower() in ("nan", "none"):
                    obj["UserName"] = ""
            fixed.append(obj)
        except Exception as e:
            print(f"Skipped line {idx}: {e}")

with open("/scratch/cherif/dataset/data_fixed.jsonl", "w", encoding="utf-8") as fout:
    for obj in fixed:
        fout.write(json.dumps(obj) + "\n")
