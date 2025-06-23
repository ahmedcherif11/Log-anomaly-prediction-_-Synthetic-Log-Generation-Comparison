import os
import re

# Define the folder path containing your YAML files
folder_path = r'C:\Users\AHMED\Desktop\new-approch\Security-Datasets\datasets\atomic'
folder_path1 = r'C:\Users\AHMED\Desktop\new-approch\Security-Datasets\datasets\atomic\_metadata'


# Base GitHub URL to look for
github_base_url = 'https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic'

# Function to recursively walk through the folder and edit YAML files
for root, dirs, files in os.walk(folder_path1):
    for file in files:
        if file.endswith('.yaml') or file.endswith('.yml'):
            file_path = os.path.join(root, file)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace GitHub URLs with local path
            new_content = re.sub(
                rf'{re.escape(github_base_url)}(/[\w/.-]+)',
                lambda m: os.path.normpath(folder_path + m.group(1)),
                content
            )

            # Only write back if something was changed
            if content != new_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f'Updated: {file_path}')
