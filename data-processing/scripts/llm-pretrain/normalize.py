import json

input_path = "/scratch/cherif/dataset/data.jsonl"
output_path = "/scratch/cherif/dataset/data_fixed_normalized.jsonl"

# with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
#     for line in fin:
#         try:
#             # Replace unquoted NaN (from pandas) with null before parsing
#             fixed_line = line.replace(":NaN", ":null")
#             obj = json.loads(fixed_line)
#             # If you want to also fix quoted "NaN" as empty string:
#             obj = {k: ("" if v == "NaN" else v) for k, v in obj.items()}
#             fout.write(json.dumps(obj) + "\n")
#         except Exception as e:
#             print("Error parsing:", line)
#             print(e)


# with open(infile, "r") as fin, open(outfile, "w") as fout:
#     for line in fin:
#         obj = json.loads(line)
#         kw = obj.get("Keywords", None)
#         # Convert integer Keywords to string, or null if it's this value
#         if isinstance(kw, int):
#             # Option 1: convert ALL int keywords to string
#             obj["Keywords"] = str(kw)
#             # Option 2: or set just the weird value to null
#             # if kw == -9223372036854775808:
#             #     obj["Keywords"] = None
#             # else:
#             #     obj["Keywords"] = str(kw)
#         fout.write(json.dumps(obj) + "\n")
# List fields to normalize as string or null

FIELDS_TO_STRINGIFY = ["Task", "Keywords", "UserName", "MapDescription", "RemoteHost"]

def normalize_field(obj, key):
    val = obj.get(key, None)
    if val is None or val == "null":
        obj[key] = None
    else:
        obj[key] = str(val)
    return obj

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        obj = json.loads(line)
        for field in FIELDS_TO_STRINGIFY:
            obj = normalize_field(obj, field)
        outfile.write(json.dumps(obj) + "\n")

print(f"Normalized fields: {FIELDS_TO_STRINGIFY}\nOutput written to: {output_path}")
