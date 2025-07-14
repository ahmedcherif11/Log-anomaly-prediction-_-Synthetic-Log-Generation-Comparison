import json

with open(r"C:\Users\AHMED\Desktop\new-approch\dataset\enterprise-attack.json", "r", encoding="utf-8") as f:
    stix_data = json.load(f)

mitre_map = {}
for obj in stix_data['objects']:
    if obj.get('type') == 'attack-pattern' and 'external_references' in obj:
        for ref in obj['external_references']:
            if ref.get('source_name') == 'mitre-attack' and ref.get('external_id', '').startswith('T'):
                tid = ref['external_id']
                mitre_map[tid] = {
                    "name": obj.get("name", ""),
                    "desc": obj.get("description", "")
                }

with open(r"C:\Users\AHMED\Desktop\new-approch\dataset\mitre_techniques.json", "w", encoding="utf-8") as f:
    json.dump(mitre_map, f, indent=2)

print(f"Extracted {len(mitre_map)} techniques to mitre_techniques.json")
