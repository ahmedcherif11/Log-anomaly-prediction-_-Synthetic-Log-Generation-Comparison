import os
import json
from pathlib import Path

# Optionally, a mapping from technique ID to name/description
MITRE_MAPPING_FILE = r"C:\Users\AHMED\Desktop\new-approch\dataset\mitre_techniques.json"  # Set to None if you don't have it

BASE_LOGS_DIR = r"C:\Users\AHMED\Desktop\new-approch\dataset\base-logs\sub_variants"
PROMPT_DIR = r"C:\Users\AHMED\Desktop\new-approch\dataset\prompts"
NUM_VARIANTS = 5

# Optional: Load MITRE technique name/desc mapping
mitre_map = {}
if MITRE_MAPPING_FILE and Path(MITRE_MAPPING_FILE).exists():
    print(f"MITRE mapping file found: {MITRE_MAPPING_FILE}")
    with open(MITRE_MAPPING_FILE, "r", encoding="utf-8") as mf:
        mitre_map = json.load(mf)  # Format: { "T1059.001": { "name": "...", "desc": "..." }, ... }
    print(f"Loaded {len(mitre_map)} techniques from mapping file.")
else:
    print(f"MITRE mapping file not found: {MITRE_MAPPING_FILE}")


os.makedirs(PROMPT_DIR, exist_ok=True)

for tech_dir in Path(BASE_LOGS_DIR).iterdir():
    if not tech_dir.is_dir():
        continue
    technique_id = tech_dir.name

    technique_id_upper = technique_id.upper()
    technique_name = ""
    technique_desc = ""
    if technique_id_upper in mitre_map:
        technique_name = mitre_map[technique_id_upper].get("name", "")
        technique_desc = mitre_map[technique_id_upper].get("desc", "")

    out_dir = Path(PROMPT_DIR) / technique_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, log_file in enumerate(tech_dir.glob("*.json")):
        with open(log_file, "r", encoding="utf-8") as f:
            base_log = json.load(f)

        # Extract description from base log if present
        if not technique_desc and "description" in base_log:
            technique_desc = base_log["description"]

        # 🟦 Build the detailed prompt (copy-paste/modify as needed)
        prompt = f"""
You are an expert in Windows Event Log generation for cybersecurity detection. Your task is to create synthetic Windows event logs for a given MITRE ATT&CK technique using the provided base log template.

----
MITRE Technique Context:
- ID: {technique_id}
- Name: {technique_name or '[Unknown Technique Name]'}
- Description: { technique_desc or '[No Description Provided]'}
----

Instructions:
- Base Log: The following is a structured base log extracted from a detection rule or Sigma rule. It describes a specific adversarial behavior mapped to a MITRE ATT&CK technique.
Instructions for log generation:

- Use the “fields” section as the canonical definition of the event log. All generated logs must match the criteria or patterns in this section, as these define the critical field names, values, and detection logic (e.g., what file was created, which command was executed).
- You may refer to the “fieldguidance” section only to clarify the meaning or value patterns of fields (such as “contains,” “ends with,” etc.). This section is for your understanding and **should not** appear in output logs.
- Ignore the “condition” and “message” fields; they are not required for log generation.
- “tactic” and “technique” fields indicate MITRE ATT&CK context. The technique (e.g., T1001.003) will be further described below.
- Other fields such as "description", "channel", and additional metadata provide context, but only vary contextually appropriate fields such as usernames, hostnames, timestamps, and file paths in your outputs to simulate natural diversity.


Base Log Template:
{json.dumps(base_log, indent=2)}


- Your output: Generate {NUM_VARIANTS} realistic, unique Windows event logs in JSON, each conforming to the structure and critical fields of the base log and aligned to the MITRE technique below.
  - Each log should reflect possible real-world variations (e.g., file names, paths, user names, timestamps), but must keep the core malicious behavior described in "fields". Output exactly {NUM_VARIANTS} unique Windows event logs in JSON format, each as a single-line JSON object.

"""

        prompt_path = out_dir / f"prompt_{idx}.txt"
        with open(prompt_path, "w", encoding="utf-8") as pf:
            pf.write(prompt.strip())
        print(f"Generated prompt: {prompt_path}")

        
        prompt_path = out_dir / f"prompt_{idx}.txt"
        with open(prompt_path, "w", encoding="utf-8") as pf:
            pf.write(prompt)
        print(f"Generated prompt: {prompt_path}")
