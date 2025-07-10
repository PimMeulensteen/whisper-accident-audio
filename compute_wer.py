import argparse
import sys
import os
import re
from collections import defaultdict

# Import numpy for correlation calculation.
# If you don't have it, run: pip install numpy
import numpy as np

from helper import compare_transcriptions, print_colored_diff

def compute_wer(reference_text: str, hypothesis_text: str) -> tuple:
    wer_value, cer_value = compare_transcriptions(reference_text, hypothesis_text)
    return wer_value, cer_value

def read_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {path}: {e}", file=sys.stderr)
        sys.exit(1)

def parse_filename(filename: str) -> tuple:
    basename = os.path.basename(filename).replace('.txt', '')
    if basename == 'audio':
        return 'no fine-tune', None, False
    if basename == 'audio_denoised':
        return 'no fine-tune', None, True

    is_denoised_input = False
    core_part = basename
    if basename.startswith('audio_denoised-'):
        is_denoised_input = True
        core_part = basename.replace('audio_denoised-', '', 1)
    elif basename.startswith('audio-'):
        is_denoised_input = False
        core_part = basename.replace('audio-', '', 1)
    else:
        return None, None, False

    match = re.search(r'^(.*?)-(\d+)$', core_part)
    if match:
        model_name = match.group(1).replace('whisper-', '', 1)
        epoch = int(match.group(2))
        return model_name, epoch, is_denoised_input
    
    return None, None, False

def generate_latex_table(results: dict):
    """
    Generates a clean LaTeX table, bolding the first occurrence of the minimum 
    in each column, and also calculates the overall WER/CER correlation.
    """
    
    # Define the exact printing order for consistent tie-breaking.
    model_print_order = (['no fine-tune'] if 'no fine-tune' in results else []) + ['ls', 'atco', 'atcod', 'all', '']
    
    ordered_data_points = []
    for model_name in model_print_order:
        if model_name not in results: continue
        if model_name == 'no fine-tune':
            ordered_data_points.append({'model': model_name, 'epoch': None, **results[model_name]})
        else:
            for epoch in sorted(results[model_name].get('epochs', {}).keys()):
                ordered_data_points.append({'model': model_name, 'epoch': epoch, **results[model_name]['epochs'][epoch]})

    if not ordered_data_points:
        print("Warning: No valid data found to generate a table.", file=sys.stderr)
        return

    # --- NEW: Collect all WER and CER scores for correlation ---
    all_wers = []
    all_cers = []
    for point in ordered_data_points:
        if 'normal' in point and 'wer' in point['normal']:
            all_wers.append(point['normal']['wer'])
            all_cers.append(point['normal']['cer'])
        if 'denoised' in point and 'wer' in point['denoised']:
            all_wers.append(point['denoised']['wer'])
            all_cers.append(point['denoised']['cer'])
    # --- END NEW ---

    # Find the minimum value for each column.
    min_scores = {
        'normal_wer': float('inf'), 'normal_cer': float('inf'),
        'denoised_wer': float('inf'), 'denoised_cer': float('inf')
    }
    for point in ordered_data_points:
        if 'normal' in point and 'wer' in point['normal']:
            min_scores['normal_wer'] = min(min_scores['normal_wer'], point['normal']['wer'])
            min_scores['normal_cer'] = min(min_scores['normal_cer'], point['normal']['cer'])
        if 'denoised' in point and 'wer' in point['denoised']:
            min_scores['denoised_wer'] = min(min_scores['denoised_wer'], point['denoised']['wer'])
            min_scores['denoised_cer'] = min(min_scores['denoised_cer'], point['denoised']['cer'])

    # Mark the *first* occurrence of each minimum for bolding.
    cells_to_bold = set()
    min_found_flags = {k: False for k in min_scores}
    for point in ordered_data_points:
        model, epoch = point['model'], point['epoch']
        if 'normal' in point:
            if not min_found_flags['normal_wer'] and abs(point['normal']['wer'] - min_scores['normal_wer']) < 1e-9:
                cells_to_bold.add((model, epoch, 'normal', 'wer'))
                min_found_flags['normal_wer'] = True
            if not min_found_flags['normal_cer'] and abs(point['normal']['cer'] - min_scores['normal_cer']) < 1e-9:
                cells_to_bold.add((model, epoch, 'normal', 'cer'))
                min_found_flags['normal_cer'] = True
        if 'denoised' in point:
            if not min_found_flags['denoised_wer'] and abs(point['denoised']['wer'] - min_scores['denoised_wer']) < 1e-9:
                cells_to_bold.add((model, epoch, 'denoised', 'wer'))
                min_found_flags['denoised_wer'] = True
            if not min_found_flags['denoised_cer'] and abs(point['denoised']['cer'] - min_scores['denoised_cer']) < 1e-9:
                cells_to_bold.add((model, epoch, 'denoised', 'cer'))
                min_found_flags['denoised_cer'] = True

    def format_score(value, is_bold):
        if value is None: return ""
        val_str = f"{value:.3f}"
        return f"\\textbf{{{val_str}}}" if is_bold else val_str

    # Print the table.
    print("  \\begin{tabular}{rc|ll|ll}")
    print("        & & \\multicolumn{2}{c|}{Normal} & \\multicolumn{2}{c}{Denoised} \\\\")
    print("             Model       & epochs & WER & CER  & WER   & CER    \\\\ \\hline")
    str_to_add = ""

    for model_name in model_print_order:
        if model_name not in results: continue

        if 'epochs' in results[model_name]: # Fine-tuned models
            print(str_to_add,end='')
            str_to_add = "\\hline\n"

            model_data = results[model_name]['epochs']
            sorted_epochs = sorted(model_data.keys())
            for i, epoch in enumerate(sorted_epochs):
                epoch_data = model_data[epoch]
                n_wer = format_score(epoch_data.get('normal', {}).get('wer'), (model_name, epoch, 'normal', 'wer') in cells_to_bold)
                n_cer = format_score(epoch_data.get('normal', {}).get('cer'), (model_name, epoch, 'normal', 'cer') in cells_to_bold)
                d_wer = format_score(epoch_data.get('denoised', {}).get('wer'), (model_name, epoch, 'denoised', 'wer') in cells_to_bold)
                d_cer = format_score(epoch_data.get('denoised', {}).get('cer'), (model_name, epoch, 'denoised', 'cer') in cells_to_bold)

                if i == 0:
                    model_spec = f"\\multirow{{{len(sorted_epochs)}}}{{*}}{{{model_name}}}"
                    print(f"{model_spec:<24} & {epoch:<3} & {n_wer:<16} & {n_cer:<16} & {d_wer:<16} & {d_cer:<16}  \\\\")
                else:
                    print(f"                         & {epoch:<3} & {n_wer:<16} & {n_cer:<16} & {d_wer:<16} & {d_cer:<16}  \\\\")
            
        else: # 'no fine-tune' model
            data = results[model_name]
            n_wer = format_score(data.get('normal', {}).get('wer'), (model_name, None, 'normal', 'wer') in cells_to_bold)
            n_cer = format_score(data.get('normal', {}).get('cer'), (model_name, None, 'normal', 'cer') in cells_to_bold)
            d_wer = format_score(data.get('denoised', {}).get('wer'), (model_name, None, 'denoised', 'wer') in cells_to_bold)
            d_cer = format_score(data.get('denoised', {}).get('cer'), (model_name, None, 'denoised', 'cer') in cells_to_bold)
            print(f"{model_name:<24} &     & {n_wer:<16} & {n_cer:<16} & {d_wer:<16} & {d_cer:<16}  \\\\ \\hline")
        
    print("    \\end{tabular}")

    # --- NEW: Calculate and print the final correlation ---
    correlation = float('nan') # Default value in case of not enough data
    if len(all_wers) > 1: # Correlation requires at least 2 data points
        # np.corrcoef returns a 2x2 matrix. The correlation is at [0,1] or [1,0].
        correlation_matrix = np.corrcoef(all_wers, all_cers)
        correlation = correlation_matrix[0, 1]
    
    print(f"\n% Pearson correlation between all WER and CER values: {correlation:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute WER/CER for a directory of hypothesis files and output a LaTeX table."
    )
    parser.add_argument("hypothesis_path", help="Path to the directory containing hypothesis .txt files")
    args = parser.parse_args()

    hyp_path = args.hypothesis_path

    if not os.path.isdir(hyp_path):
        print(f"Error: Hypothesis path must be a directory: {hyp_path}", file=sys.stderr)
        sys.exit(1)
    
    ref_path = os.path.join(hyp_path, "transcript.txt")

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    reference_text = read_file(ref_path)
    reference_filename = os.path.basename(ref_path)

    word_count = len(reference_text.split())
    print(f"Reference transcript word count: {word_count}\n")

    for fname in sorted(os.listdir(hyp_path)):
        if not fname.lower().endswith('.txt') or fname == reference_filename: continue
        
        model_name, epoch, is_denoised_input = parse_filename(fname)
        
        if not model_name: continue

        full_path = os.path.join(hyp_path, fname)
        hypothesis_text = read_file(full_path)
        wer_score, cer_score = compute_wer(reference_text, hypothesis_text)
        # print_colored_diff(reference_text, hypothesis_text)
        if wer_score is None: continue
            
        result_type = 'denoised' if is_denoised_input else 'normal'
        scores = {'wer': wer_score, 'cer': cer_score}
        
        if epoch is not None:
            results[model_name]['epochs'][epoch][result_type] = scores
        else:
            results[model_name][result_type] = scores

    generate_latex_table(results)