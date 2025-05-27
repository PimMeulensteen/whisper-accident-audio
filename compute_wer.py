import argparse
import sys
import os
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

def process_pair(ref_path: str, hyp_path: str, show_diff: bool):
    reference_text = read_file(ref_path)
    hypothesis_text = read_file(hyp_path)

    wer_score, cer_score = compute_wer(reference_text, hypothesis_text)
    if wer_score is None:
        print(f"Error: couldn't compute WER for {hyp_path}.", file=sys.stderr)
        return

    print(f"{os.path.basename(hyp_path):>35} | WER: {wer_score:>6.2%} | CER: {cer_score:>6.2%}")
    if show_diff:
        print_colored_diff(reference_text, hypothesis_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute WER between a reference file and hypothesis file or all .txt files in a hypothesis directory, with optional colored diff."
    )
    parser.add_argument("reference_file", help="Path to the reference transcription file (.txt)")
    parser.add_argument("hypothesis_path", help="Path to the hypothesis file (.txt) or directory containing .txt files")
    parser.add_argument(
        "-d", "--diff",
        action="store_true",
        help="Print a colored word-level diff (additions in green, removals in red)"
    )
    args = parser.parse_args()

    ref_path = args.reference_file
    hyp_path = args.hypothesis_path
    show_diff = args.diff

    if os.path.isdir(hyp_path):
        for fname in sorted(os.listdir(hyp_path)):
            if not fname.lower().endswith('.txt'):
                continue
            full_path = os.path.join(hyp_path, fname)
            if os.path.isfile(full_path):
                process_pair(ref_path, full_path, show_diff)
    else:
        if hyp_path.lower().endswith('.txt'):
            process_pair(ref_path, hyp_path, show_diff)
        else:
            print(f"Error: hypothesis file must be a .txt file: {hyp_path}", file=sys.stderr)
            sys.exit(1)
