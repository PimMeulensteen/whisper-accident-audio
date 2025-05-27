from typing import Dict, Any, List, Optional, Tuple, Set
from audio_separator import separator
import os
import re

def process_token(token: str) -> str:
    """
    Insert a space between a lowercase and uppercase letter,
    and if the token is all lowercase, convert it to title case.
    """
    processed = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', token)
    if processed.islower():
        processed = processed.title()
    return processed

def clean_name(filename: str) -> str:
    """
    Remove the .ckpt extension, keep underscores for separation,
    improve capitals, and escape underscores for LaTeX.
    """
    name, _ = os.path.splitext(filename)
    if '_' in name:
        tokens = name.split('_')
        processed_tokens = [process_token(token) for token in tokens]
        cleaned = '_'.join(processed_tokens)
    else:
        cleaned = process_token(name)
    # Escape underscores for LaTeX
    cleaned = cleaned.replace('_', r'\_')
    return cleaned

# Initialize the separator and get the available models.
temp_sep = separator.Separator()  
available_models = temp_sep.list_supported_model_files()

# Begin the LaTeX table with three columns: Name, Stems, and Status.
print(r"\begin{tabular}{|l|l|l|}")
print(r"\hline")
print("Name & Stems & Status \\\\")
print(r"\hline")

# Iterate through the models and output each as a row in the table.
for model_type in available_models:
    for item in available_models[model_type]:
        model_data = available_models[model_type][item]
        filename = model_data.get('filename', 'N/A')
        name = clean_name(filename)
        stems = model_data.get('stems', [])
        if not stems:
            stems_str = model_data.get('target_stem', [])
        else:
            stems_str = ", ".join(stems)
        
        # Determine status: accepted if "no noise" or "vocals" is in stems.
        if "no noise" in stems or "vocals" in stems:
            status = "accepted"
        else:
            status = "rejected"
            
        print(f"{name} & {stems_str} & {status} \\\\")
        # print(r"\hline")

# End the LaTeX table.
print(r"\end{tabular}")
