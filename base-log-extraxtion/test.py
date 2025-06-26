import yaml
import json
from pathlib import Path
import argparse
import re
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

        def expand_blocks(pattern, blocks):
            if '*' in pattern:
                regex = re.compile(pattern.replace('*', '.*'))
                return [b for b in blocks if regex.fullmatch(b)]
            else:
                return [pattern] if pattern in blocks else []

        def split_condition_on_top_level_and(cond):
            stack = []
            last = 0
            result = []
            i = 0
            while i < len(cond):
                if cond[i] == '(':
                    stack.append(i)
                    i += 1
                elif cond[i] == ')':
                    if stack:
                        stack.pop()
                    i += 1
                elif cond[i:i+3] == 'and' and not stack:
                    before = i == 0 or cond[i-1].isspace()
                    after = (i+3 == len(cond)) or cond[i+3].isspace()
                    if before and after:
                        result.append(cond[last:i].strip())
                        last = i+3
                        i += 3
                    else:
                        i += 1
                else:
                    i += 1
            result.append(cond[last:].strip())
            return result

        def split_condition_on_top_level_or(cond):
            stack = []
            last = 0
            result = []
            i = 0
            while i < len(cond):
                if cond[i] == '(':
                    stack.append(i)
                    i += 1
                elif cond[i] == ')':
                    if stack:
                        stack.pop()
                    i += 1
                elif cond[i:i+2] == 'or' and not stack:
                    before = i == 0 or cond[i-1].isspace()
                    after = (i+2 == len(cond)) or cond[i+2].isspace()
                    if before and after:
                        result.append(cond[last:i].strip())
                        last = i+2
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            result.append(cond[last:].strip())
            return result

        # ---- Recursive handling starts here ----

        # If the condition is wrapped in parentheses, unwrap it
        cond = condition.strip()
        if cond.startswith('(') and cond.endswith(')'):
            cond = cond[1:-1].strip()

        # If top-level ORs, process each branch and return as separate combos
        or_groups = split_condition_on_top_level_or(cond)
        if len(or_groups) > 1:
            # Collect all combinations from all OR branches
            all_fixed, all_variable, all_excluded = [], [], []
            for g in or_groups:
                fixed, var, excl = self.parse_condition(g, selection_blocks)
                all_fixed.append(fixed)
                all_variable.append(var)
                all_excluded.append(excl)

            return all_fixed, all_variable, all_excluded

        # If not, split on AND and process each part
        and_groups = split_condition_on_top_level_and(cond)
        variable_blocks = []
        fixed_blocks = []
        excluded_blocks = []

        # If any group inside AND is itself a ( ... or ... ), recurse!
        for token in and_groups:
            token = token.strip()
            if token.startswith('(') and token.endswith(')') and 'or' in token:
                # Recursively parse this OR group, producing all variants
                inner_fixed, inner_var, inner_excl = self.parse_condition(token[1:-1], selection_blocks)
                # Each branch is a variant, so for each, pair with rest of AND (cross product logic needed in main loop!)
                # return these as lists-of-lists, and the calling code must expand them
                fixed_blocks.append(inner_fixed)
                variable_blocks.append(inner_var)
                excluded_blocks.append(inner_excl)
            else:
                # Handle normal (not ...) and variable/one_of/all_of
                if token.lower().startswith('not '):
                    not_matches = re.findall(r'not\s+(?:1 of|all of)?\s*([\w\*]+)', token)
                    for n in not_matches:
                        excluded_blocks.extend(expand_blocks(n, selection_blocks))
                    continue
                m_one_of = re.match(r'1 of\s+(.+)', token)
                if m_one_of:
                    one_of_pattern = m_one_of.group(1).strip()
                    variable_blocks.append({'one_of': expand_blocks(one_of_pattern, selection_blocks)})
                    continue
                m_all_of = re.match(r'all of\s+(.+)', token)
                if m_all_of:
                    all_of_pattern = m_all_of.group(1).strip()
                    variable_blocks.append({'all_of': expand_blocks(all_of_pattern, selection_blocks)})
                    continue
                # Otherwise, treat as fixed block
                blocks = expand_blocks(token, selection_blocks)
                fixed_blocks.extend(blocks)

        # Remove excluded from fixed and variable
        fixed_blocks = [b for b in fixed_blocks if b not in excluded_blocks]
        for vb in variable_blocks:
            if isinstance(vb, dict):
                for k in vb:
                    vb[k] = [b for b in vb[k] if b not in excluded_blocks]

        return fixed_blocks, variable_blocks, excluded_blocks


    @staticmethod
    def split_condition_on_top_level_and(condition):
        # Splits on 'and' that are not inside parentheses
        cond = condition
        stack = []
        last = 0
        result = []
        i = 0
        while i < len(cond):
            if cond[i] == '(':
                stack.append(i)
                i += 1
            elif cond[i] == ')':
                if stack:
                    stack.pop()
                i += 1
            elif cond[i:i+3] == 'and' and not stack:
                before = i == 0 or cond[i-1].isspace()
                after = (i+3 == len(cond)) or cond[i+3].isspace()
                if before and after:
                    result.append(cond[last:i].strip())
                    last = i+3
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        result.append(cond[last:].strip())
        return result

        
    @staticmethod
    def split_condition_on_top_level_or(condition):
        # Splits on 'or' that are not inside parentheses
        cond = condition
        stack = []
        last = 0
        result = []
        i = 0
        while i < len(cond):
            if cond[i] == '(':
                stack.append(i)
                i += 1
            elif cond[i] == ')':
                if stack:
                    stack.pop()
                i += 1
            elif cond[i:i+2] == 'or' and not stack:
                # Only split if surrounded by whitespace (not part of a word)
                before = i == 0 or cond[i-1].isspace()
                after = (i+2 == len(cond)) or cond[i+2].isspace()
                if before and after:
                    result.append(cond[last:i].strip())
                    last = i+2
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        result.append(cond[last:].strip())
        return result

    def parse_condition_tree(self, condition, selection_blocks):

        import re

        # Remove outer parentheses
        condition = condition.strip()
        while condition.startswith('(') and condition.endswith(')'):
            # Only remove if parentheses match
            stack = 0
            for i, c in enumerate(condition):
                if c == '(': stack += 1
                if c == ')': stack -= 1
                if stack == 0 and i < len(condition) - 1: break
            else:
                condition = condition[1:-1].strip()
                continue
            break

        # Split top-level OR
        or_groups = self.split_condition_on_top_level_or(condition)
        if len(or_groups) > 1:
            return {'or': [self.parse_condition_tree(group, selection_blocks) for group in or_groups]}

        # Split top-level AND
        and_groups = self.split_condition_on_top_level_and(condition)
        if len(and_groups) > 1:
            return {'and': [self.parse_condition_tree(group, selection_blocks) for group in and_groups]}

        # Handle "not"
        if condition.lower().startswith("not "):
            return {'not': self.parse_condition_tree(condition[4:].strip(), selection_blocks)}

        # Handle "1 of" or "all of"
        m_one = re.match(r'1 of\s+(.+)', condition)
        if m_one:
            pattern = m_one.group(1).strip()
            return {'one_of': self.expand_blocks(pattern, selection_blocks)}
        m_all = re.match(r'all of\s+(.+)', condition)
        if m_all:
            pattern = m_all.group(1).strip()
            return {'all_of': self.expand_blocks(pattern, selection_blocks)}

        # Otherwise, treat as block or wildcard
        expanded = self.expand_blocks(condition, selection_blocks)
        if len(expanded) == 1:
            return expanded[0]
        elif len(expanded) > 1:
            return {'all_of': expanded}
        else:
            return condition  # fallback

    def expand_blocks(self, pattern, blocks):
        import re
        if '*' in pattern:
            regex = re.compile(pattern.replace('*', '.*'))
            return [b for b in blocks if regex.fullmatch(b)]
        else:
            return [pattern] if pattern in blocks else []

    def expand_condition_tree(self, tree):
        """
        Expand a condition tree into a list of combinations (each combination is a list of block names).
        """
        if isinstance(tree, str):
            return [[tree]]
        if isinstance(tree, list):
            # treat as AND of all
            combos = [[]]
            for subtree in tree:
                subcombos = self.expand_condition_tree(subtree)
                new_combos = []
                for c in combos:
                    for sc in subcombos:
                        new_combos.append(c + sc)
                combos = new_combos
            return combos
        if isinstance(tree, dict):
            if 'or' in tree:
                combos = []
                for option in tree['or']:
                    combos.extend(self.expand_condition_tree(option))
                return combos
            elif 'and' in tree:
                combos = [[]]
                for part in tree['and']:
                    part_combos = self.expand_condition_tree(part)
                    new_combos = []
                    for c in combos:
                        for pc in part_combos:
                            new_combos.append(c + pc)
                    combos = new_combos
                return combos
            elif 'not' in tree:
                # 'not' is tricky: we return an empty list (don't expand these blocks)
                return [[]]
            elif 'one_of' in tree:
                return [[block] for block in tree['one_of']]
            elif 'all_of' in tree:
                return [tree['all_of']]
        return [[]]

        
    def handle_variable_blocks(self, condition, variable_blocks, fixed_blocks, selection_blocks, rule):
        result_base_logs = []
        print(f"Handling condition: {condition}")
        print(f"Variable blocks: {variable_blocks}, Fixed blocks: {fixed_blocks}")

        if re.search(r'(?<!not\s)1 of', condition) and variable_blocks:
            for block in variable_blocks:
                print(f"Expanding variable_blocks: {variable_blocks}")
                print(f"Available selection_blocks: {list(selection_blocks.keys())}")
                block_value = selection_blocks.get(block, {})
                single_log = {
                    'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                    'description': rule.get('description', ''),
                    'tactic': rule.get('tactic', ''),
                    'technique': rule.get('technique', []),
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
                    for fb in fixed_blocks:
                        fb_value = selection_blocks.get(fb, {})
                        single_log['fields'][fb] = fb_value
                        guidance, message = self.generate_field_guidance_and_message(fb_value)
                        single_log['fieldguidance'][fb] = guidance
                        if message:
                            single_log['message'] += message + " | "

                    single_log['condition'] = ' and '.join([block] + fixed_blocks)

                result_base_logs.append(single_log)
        elif  re.search(r'(?<!not\s)all of', condition) :
            base_log = {
                'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                'description': rule.get('description', ''),
                'tactic': rule.get('tactic', ''),
                'technique': rule.get('technique', []),
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
                # For 'all of', the condition should reflect all fixed blocks joined by 'and'
                base_log['condition'] = ' and '.join(fixed_blocks)
            result_base_logs.append(base_log)
        else:
        # Default: AND
            base_log = {
                'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                'description': rule.get('description', ''),
                'tactic': rule.get('tactic', ''),
                'technique': rule.get('technique', []),
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
                base_log['condition'] = ' and '.join(fixed_blocks)

            result_base_logs.append(base_log)
        return result_base_logs
    

    def generate_base_logs_from_condition(self):

        with open(self.input_path, 'r', encoding='utf-8') as f:
            rule = yaml.safe_load(f)
                # Extract tags, tactic, and technique
        summary = {}
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
        detection = rule.get('detection', {})
        condition = detection.get('condition', '')
        print(f"Processing detection condition: {condition}")

        selection_blocks = { key: value for key, value in detection.items() if key != 'condition' }
        if not condition:
            print("No condition found in the rule. Cannot generate base logs.")
            return []   

        # NEW: Parse the tree and expand all combinations!
        cond_tree = self.parse_condition_tree(condition, selection_blocks)
        print("Parsed condition tree:", cond_tree)

        all_combos = self.expand_condition_tree(cond_tree)
        print("All expanded combinations:", all_combos)

        result_base_logs = []
        for combo in all_combos:
            fields = {}
            fieldguidance = {}
            message_parts = []
            for block in combo:
                block_value = selection_blocks.get(block, {})
                fields[block] = block_value
                guidance, message = self.generate_field_guidance_and_message(block_value)
                fieldguidance[block] = guidance
                if message:
                    message_parts.append(message)
            base_log = {
                'channel': rule.get('logsource', {}).get('service') or rule.get('logsource', {}).get('category', 'unknown'),
                'description': rule.get('description', ''),
                'tactic':  summary['tactic'],
                'technique': techniques,
                'fields': fields,
                'fieldguidance': fieldguidance,
                'message': " | ".join(message_parts),
                'condition': " and ".join(combo)
            }
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
                'tactic': base_log.get('tactic', ''),
                'technique': base_log.get('technique', []),
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