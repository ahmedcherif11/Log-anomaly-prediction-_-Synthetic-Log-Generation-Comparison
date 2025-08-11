import json
import re

def replace_instructions_in_jsonl(input_file, output_file):
    """
    Replace the log generation instructions and remove MITRE technique context in all lines of a JSONL file
    """
    
    # The new instructions text to replace with
    new_instructions = """Log Generation Rules

Use the fields section of the base log as the source of truth.

Only use fieldguidance for understanding value patterns; do not include it in the output.

Ignore condition and message fields.

Vary contextual fields (e.g., usernames, hostnames, timestamps, file paths) naturally.

Ensure all logs match the detection logic defined in fields."""

    # Pattern to match and remove the MITRE Technique Context section
    mitre_pattern = r'----\nMITRE Technique Context:.*?----\n\n'
    
    # Pattern to match the existing instructions section
    # This pattern looks for the instructions section that starts with "Instructions for log generation:"
    # and continues until the Base Log Template section
    old_pattern = r'Instructions for log generation:\s*\n\n.*?(?=\n\nBase Log Template:)'
    
    # Replacement text for instructions
    replacement = f"Instructions for log generation:\n\n{new_instructions}"
    
    processed_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse the JSON object
                    json_obj = json.loads(line.strip())
                    
                    # Check if this line has a 'prompt' field
                    if 'prompt' in json_obj:
                        # Replace the instructions in the prompt
                        original_prompt = json_obj['prompt']
                        
                        # First remove the MITRE Technique Context section
                        updated_prompt = re.sub(
                            mitre_pattern, 
                            '', 
                            original_prompt, 
                            flags=re.DOTALL
                        )
                        
                        # Then replace the instructions section
                        updated_prompt = re.sub(
                            old_pattern, 
                            replacement, 
                            updated_prompt, 
                            flags=re.DOTALL
                        )
                        
                        # Update the JSON object
                        json_obj['prompt'] = updated_prompt
                        
                        # Convert back to JSON string
                        updated_line = json.dumps(json_obj, ensure_ascii=False)
                        processed_lines.append(updated_line)
                        
                        print(f"Processed line {line_num}")
                    else:
                        # If no prompt field, keep the line as is
                        processed_lines.append(line.strip())
                        print(f"Line {line_num}: No prompt field found, keeping as is")
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    processed_lines.append(line.strip())
                    
        # Write the processed lines to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
                
        print(f"Successfully processed {len(processed_lines)} lines")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    input_file = r"c:\Users\AHMED\Desktop\new-approch\dataset\prompts-copy.jsonl"
    output_file = r"c:\Users\AHMED\Desktop\new-approch\dataset\prompts-copy-updated.jsonl"
    
    print("Starting replacement process...")
    replace_instructions_in_jsonl(input_file, output_file)
    print("Replacement process completed!")
