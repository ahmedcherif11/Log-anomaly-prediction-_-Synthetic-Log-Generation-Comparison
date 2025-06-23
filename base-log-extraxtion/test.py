import yaml
import json
from pathlib import Path
import argparse
import itertools


class SigmaSummaryGenerator:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    @staticmethod
    def flatten_selections(selections):
        """Recursively yield all dicts from a list of selections."""
        for item in selections:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                yield from SigmaSummaryGenerator.flatten_selections(item)

    @staticmethod
    def generate_field_guidance_and_message(selection):
        guidance = {}
        message_parts = []

        templates = {
            "CommandLine": "Command executed: {}",
            "ImageLoaded": "Driver loaded: {}",
            "Image": "Executable loaded: {}",
            "TargetImage": "Executable loaded: {}",
            "Data": "Log references: {}",
            "Hashes": "Observed SHA256 hash: {}",
            "ParentImage": "Parent process: {}",
            "Provider_Name": "Event from provider: {}",
            "OriginalFileName": "Original file name: {}"
        }

        def is_operator_field(f):
            return any(op in f for op in ['|contains', '|endswith', '|startswith'])

        def operator_label(field):
            base = field.split('|')[0]
            patterns = [
            ('|contains|all|windash', f"{base} CommandLine contains all of (ignoring dash variants)"),
            ('|contains|windash', f"{base} CommandLine contains (ignoring dash variants)"),
            ('|contains|all', f"{base} contains all of"),
            ('|contains', f"{base} contains"),
            ('|endswith', f"{base} ends with"),
            ('|startswith', f"{base} starts with"),
            ('|re', f"{base} matches regex"),
            ('|clip', f"{base} contains (case-insensitive)"),
            ('|cpi', f"{base} contains path item (case-insensitive)"),
            ('|fieldref', f"{base} equals field"),
            ('cidr', f"{base} IpAddress matches network ranges"),
            ('base64offset|contains', f"{base} contains base64-encoded substring (with possible offset)"),
            ('|invoke', f"{base} field contains invocation of"),
            ]
            for pattern, label in patterns:
                if pattern in field:
                    return label
            return base

        dicts_to_process = []
        if isinstance(selection, dict):
            dicts_to_process = [selection]
        elif isinstance(selection, list):
            dicts_to_process = list(SigmaSummaryGenerator.flatten_selections(selection))

        for d in dicts_to_process:
            for field, value in d.items():
                base_field = field.split('|')[0]
                label = operator_label(field)
                if isinstance(value, list):
                    if is_operator_field(field):
                        if label in guidance:
                            if isinstance(guidance[label][0], list):
                                guidance[label].append(value)
                            else:
                                guidance[label] = [guidance[label], value]
                        else:
                            guidance[label] = value
                    if base_field in templates:
                        message_parts.append(templates[base_field].format(", ".join(str(v) for v in value[:2])))
                else:
                    if is_operator_field(field):
                        guidance[label] = [value]
                    if base_field in templates:
                        message_parts.append(templates[base_field].format(value))

        return guidance, " | ".join(message_parts)
    

    def create_structured_summary_json(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            rule = yaml.safe_load(f)

        summary = {}

        # Basic info
        logsource = rule.get('logsource', {})
        summary['channel'] = logsource.get('service') or logsource.get('category', 'unknown')
        summary['description'] = rule.get('description', '')

        # Extract tags, tactic, and technique
        tags = rule.get('tags', [])

        # Extract tactic (first tag starting with 'attack.' and not 'attack.t')
        tactic = None
        for tag in tags:
            if tag.startswith('attack.') and not tag.startswith('attack.t'):
                if 'tactic' not in summary:
                    summary['tactic'] = []
                summary['tactic'].append(tag.replace('attack.', '').replace('-', ' ').title())
        if 'tactic' in summary and len(summary['tactic']) == 1:
            summary['tactic'] = summary['tactic'][0]
        if tactic:
            summary['tactic'] = tactic

        # Extract techniques (all tags starting with 'attack.t')
        techniques = []
        for tag in tags:
            if tag.startswith('attack.t'):
                techniques.append(tag.replace('attack.', ''))
        if techniques:
            summary['technique'] = techniques

        # Collect all selection_* blocks and keep them as-is
        detection = rule.get('detection', {})
        selection_blocks = {}
        field_guidance = {}
        message_parts = []

        for key, value in detection.items():
            if key.startswith('condition') :
                break
            else:
                selection_blocks[key] = value
                # Optionally, generate guidance/message for each block
                guidance, message = self.generate_field_guidance_and_message(value)
                field_guidance[key] = guidance
                if message:
                    message_parts.append(message)

        summary['fields'] = selection_blocks
        summary['fieldguidance'] = field_guidance
        summary['message'] = " | ".join(message_parts)
        summary['condition'] = detection.get('condition', '')

        # Save to JSON
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)


    def parse_condition(self, condition, selection_blocks):
        import re

        tokens = re.findall(r'[\w\*]+(?:_[\w\*]+)*\*?', condition)
        keywords = {'and', 'or', 'not', '1', 'of', 'all', 'none'}
        block_tokens = [ t for t in tokens if t.lower() not in keywords ]

        expanded_blocks = []
        for token in block_tokens:
            if '*' in token:
                pattern = re.compile(token.replace('*', '.*'))
                matches = [ name for name in selection_blocks if pattern.fullmatch(name) ]
                expanded_blocks.extend(matches)
            else:
                expanded_blocks.append(token)

        expanded_blocks = list(set(expanded_blocks))

        excluded_blocks = []
        if 'not' in condition:
            not_matches = re.findall(r'not\s+(?:1 of|all of)?\s*([\w\*]+)', condition)
            for n in not_matches:
                if '*' in n:
                    pattern = re.compile(n.replace('*', '.*'))
                    matches = [ name for name in selection_blocks if pattern.fullmatch(name) ]
                    excluded_blocks.extend(matches)
                else:
                    excluded_blocks.append(n)

        included_blocks = [ b for b in expanded_blocks if b not in excluded_blocks ]

        # Detect fixed vs variable:
        variable_blocks = []
        fixed_blocks = []

        # If "1 of X"
        if '1 of' in condition:
            one_of_matches = re.findall(r'1 of\s*([\w\*]+)', condition)
            for token in one_of_matches:
                if '*' in token:
                    pattern = re.compile(token.replace('*', '.*'))
                    matches = [ name for name in included_blocks if pattern.fullmatch(name) ]
                    variable_blocks.extend(matches)
                else:
                    variable_blocks.append(token)

        # Fixed = all other included blocks
        fixed_blocks = [ b for b in included_blocks if b not in variable_blocks ]

        return fixed_blocks, variable_blocks, excluded_blocks

    def generate_base_logs_from_condition(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            rule = yaml.safe_load(f)

        detection = rule.get('detection', {})
        condition = detection.get('condition', '')

        # Step 1: collect all available selection block names
        selection_blocks = { key: value for key, value in detection.items() if key != 'condition' }
        fixed_blocks= []
        variable_blocks = []
        excluded_blocks = []
        if not condition:
            print("No condition found in the rule. Cannot generate base logs.")
            return []   
        # Step 2: parse the condition to identify included, excluded, fixed, and variable blocks
        fixed_blocks, variable_blocks, excluded_blocks = self.parse_condition(condition, selection_blocks)
        print(fixed_blocks, variable_blocks, excluded_blocks)
        # Step 3: determine which blocks to include in the base log


        # Step 4: generate combinations
        result_base_logs = []

        if '1 of' in condition and not 'not 1 of' in condition:
            for block in variable_blocks:
                block_value = selection_blocks.get(block, {})
                single_log = {
                    'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                    'description': rule.get('description', ''),
                    'fields': { block: block_value },
                    'fieldguidance': {},
                    'message': '',
                    'condition': f"{block}"
                }
                guidance, message = self.generate_field_guidance_and_message(block_value)
                single_log['fieldguidance'][block] = guidance
                if message:
                    single_log['message'] += message + " | "

                # Add fixed blocks from AND part:
                if fixed_blocks:
                    print(f"Adding fixed blocks: {fixed_blocks}")
                    for fb in fixed_blocks:
                        fb_value = selection_blocks.get(fb, {})
                        single_log['fields'][fb] = fb_value
                        guidance, message = self.generate_field_guidance_and_message(fb_value)
                        single_log['fieldguidance'][fb] = guidance
                        if message:
                            single_log['message'] += message + " | "

                    single_log['condition'] = ' and '.join([block] + fixed_blocks)

                result_base_logs.append(single_log)


        elif 'all of' in condition and not 'not all of' in condition:
            base_log = {
                'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                'description': rule.get('description', ''),
                'fields': {},
                'fieldguidance': {},
                'message': '',
                'condition': condition
            }
            for block in fixed_blocks:
                block_value = selection_blocks.get(block, {})
                base_log['fields'][block] = block_value
                guidance, message = self.generate_field_guidance_and_message(block_value)
                base_log['fieldguidance'][block] = guidance
                if message:
                    base_log['message'] += message + " | "
            result_base_logs.append(base_log)

        else:
            # Default: AND
            print("ena houni")
            base_log = {
                'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                'description': rule.get('description', ''),
                'fields': {},
                'fieldguidance': {},
                'message': '',
                'condition': condition
            }
            for block in fixed_blocks:
                block_value = selection_blocks.get(block, {})
                base_log['fields'][block] = block_value
                guidance, message = self.generate_field_guidance_and_message(block_value)
                base_log['fieldguidance'][block] = guidance
                if message:
                    base_log['message'] += message + " | "
            result_base_logs.append(base_log)

        return result_base_logs
    def generate_sub_variants(self, base_log):
        import itertools

        def get_guidance_label(field, fg_selection):
            field_base = field.split('|')[0].lower()
            op = None
            if 'endswith' in field: op = 'ends with'
            elif 'contains|all' in field: op = 'contains all of'
            elif 'contains' in field: op = 'contains'
            elif 'startswith' in field: op = 'starts with'
            if op:
                expected = f"{field_base.capitalize()} {op}"
                for fg_label in fg_selection:
                    if fg_label.lower() == expected.lower():
                        return fg_label
            for fg_label in fg_selection:
                if fg_label.lower().startswith(field_base):
                    return fg_label
            return None

        fields = base_log.get('fields', {})
        fieldguidance = base_log.get('fieldguidance', {})

        block_fields_dict = {}
        for block_name, block_content in fields.items():
            fg_selection = fieldguidance.get(block_name, {})
            block_fields = []
            if isinstance(block_content, list):
                block_items = block_content
            else:
                block_items = [block_content]
            for item in block_items:
                for field, value in item.items():
                    label = get_guidance_label(field, fg_selection)
                    is_all_of = label is not None and "contains all of" in label
                    values = value if isinstance(value, list) else [value]
                    block_fields.append((field, label, values, is_all_of))
            block_fields_dict[block_name] = block_fields

        # For each block, build split fields
        block_splits = {}
        for block_name, block_fields in block_fields_dict.items():
            split_fields = []
            for field, label, values, is_all_of in block_fields:
                if is_all_of:
                    split_fields.append([(field, label, values, is_all_of)])
                else:
                    split_fields.append([(field, label, [v], is_all_of) for v in values])
            block_splits[block_name] = split_fields

        # Cartesian product *per block*, then cross product of blocks
        block_combos = {}
        for block_name, splits in block_splits.items():
            block_combos[block_name] = list(itertools.product(*splits))
        all_combos = list(itertools.product(*block_combos.values()))

        sub_variants = []
        for combo_blocks in all_combos:
            fields_dict = {}
            fg_dict = {}
            selection_for_message = {}
            for i, block_name in enumerate(block_fields_dict.keys()):
                combo = combo_blocks[i]
                fields_dict[block_name] = {}
                fg_dict[block_name] = {}
                fg_selection = fieldguidance.get(block_name, {})
                for (field, label, value, is_all_of) in combo:
                    use_val = value if is_all_of else value[0]
                    fields_dict[block_name][field] = use_val
                    selection_for_message[field] = use_val
                    if label:
                        fg_dict[block_name][label] = use_val
            _, message = self.generate_field_guidance_and_message(selection_for_message)
            variant = {
                'channel': base_log.get('channel', ''),
                'description': base_log.get('description', ''),
                'fields': fields_dict,
                'fieldguidance': fg_dict,
                'message': message,
                'condition': base_log.get('condition', '')
            }
            sub_variants.append(variant)

        return sub_variants

def main():
    parser = argparse.ArgumentParser(description="Generate Sigma rule summary JSON and base log variants.")
    parser.add_argument("-i", "--input", required=True, help="Path to Sigma YAML rule file")
    parser.add_argument("-o", "--output", required=True, help="Path to output summary JSON file (base name)")
    args = parser.parse_args()

    generator = SigmaSummaryGenerator(args.input, args.output)

    # Step 1: Create your existing summary
    generator.create_structured_summary_json()
    print(f"Structured summary JSON created at {args.output}")

    # Step 2: Generate multiple base logs from condition
    base_logs = generator.generate_base_logs_from_condition()
    print(f"Found {len(base_logs)} base log variant(s) from condition")

    # Save each variant + sub-variants
    for i, base_log in enumerate(base_logs):
        variant_filename = f"{args.output.replace('.json', '')}_variant_{i}.json"
        with open(variant_filename, 'w', encoding='utf-8') as f:
            json.dump(base_log, f, indent=2)
        print(f"Base log variant saved: {variant_filename}")

        # Now generate sub-variants:
        sub_variants = generator.generate_sub_variants(base_log)
        print(f"Base log {i} → {len(sub_variants)} sub-variants")

        for j, sub in enumerate(sub_variants):
            sub_filename = f"{args.output.replace('.json', '')}_variant_{i}_sub_{j}.json"
            with open(sub_filename, 'w', encoding='utf-8') as f:
                json.dump(sub, f, indent=2)
            print(f"Sub-variant saved: {sub_filename}")

if __name__ == "__main__":
    main()