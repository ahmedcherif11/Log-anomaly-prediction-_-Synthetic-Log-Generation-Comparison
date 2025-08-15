#!/usr/bin/env python3
"""
Script to modify prompts-copy.jsonl file:
1. Remove MITRE context section from prompts
2. Add technique number at the beginning of the prompt
3. Keep only first 2 logs in responses instead of 5
"""

import json
import re
import sys
import os

def extract_technique_id(prompt_text):
    """Extract technique ID from the MITRE context section."""
    # Look for "- ID: " pattern in the MITRE context
    match = re.search(r'- ID:\s+([tT]\d+(?:\.\d+)*)', prompt_text)
    if match:
        return match.group(1).lower()
    return None

def remove_mitre_context(prompt_text):
    """Remove the MITRE Technique Context section from the prompt."""
    # Find the start and end of the MITRE context section
    start_pattern = r'----\nMITRE Technique Context:'
    end_pattern = r'----\n'
    
    # Find the start of MITRE context
    start_match = re.search(start_pattern, prompt_text)
    if not start_match:
        return prompt_text
    
    # Find the end after the start
    end_match = re.search(end_pattern, prompt_text[start_match.end():])
    if not end_match:
        return prompt_text
    
    # Remove the MITRE context section
    before_context = prompt_text[:start_match.start()]
    after_context = prompt_text[start_match.end() + end_match.end():]
    
    return before_context + after_context

def keep_first_two_logs(response_text):
    """Keep only the first 2 logs from the response instead of 5."""
    if not response_text:
        return response_text
    
    # Split the response by newlines to get individual log entries
    lines = response_text.strip().split('\n')
    
    # Keep only the first 2 lines (logs)
    if len(lines) >= 2:
        return '\n'.join(lines[:2])
    else:
        return response_text

def modify_prompt(prompt_text):
    """Modify the prompt according to requirements."""
    # Extract technique ID
    technique_id = extract_technique_id(prompt_text)
    if not technique_id:
        print("Warning: Could not extract technique ID from prompt")
        return prompt_text
    
    # Remove MITRE context
    modified_prompt = remove_mitre_context(prompt_text)
    
    # Add technique number at the beginning and modify the task description
    intro = "You are an expert in Windows Event Log generation for cybersecurity detection. Your task is to create synthetic Windows event logs for a  MITRE ATT&CK technique {} using the provided base log template.".format(technique_id)
    
    # Replace the original introduction
    original_intro = "You are an expert in Windows Event Log generation for cybersecurity detection. Your task is to create synthetic Windows event logs for a given MITRE ATT&CK technique using the provided base log template."
    
    modified_prompt = modified_prompt.replace(original_intro, intro)
    
    # Update the output instruction to generate 2 logs instead of 5
    modified_prompt = re.sub(
        r'Generate 5 realistic, unique Windows event logs',
        'Generate 2 realistic, unique Windows event logs',
        modified_prompt
    )
    
    modified_prompt = re.sub(
        r'Output exactly 5 unique Windows event logs',
        'Output exactly 2 unique Windows event logs',
        modified_prompt
    )
    
    return modified_prompt

def process_jsonl_file(input_file, output_file):
    """Process the JSONL file and modify each entry."""
    modified_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Modify the prompt
                if 'prompt' in data:
                    data['prompt'] = modify_prompt(data['prompt'])
                
                # Modify the response (keep only first 2 logs)
                if 'response' in data:
                    data['response'] = keep_first_two_logs(data['response'])
                
                # Write modified line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                modified_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully modified {modified_count} entries")
    return modified_count

def main():
    # Define file paths
    input_file = r"c:\Users\AHMED\Desktop\new-approch\dataset\prompts-copy.jsonl"
    output_file = r"c:\Users\AHMED\Desktop\new-approch\dataset\prompts-modified.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    try:
        # Process the file
        print(f"Processing file: {input_file}")
        count = process_jsonl_file(input_file, output_file)
        print(f"Output written to: {output_file}")
        print(f"Total entries processed: {count}")
        return 0
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
