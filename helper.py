import difflib
import jiwer
import re

def clean_str(s):
    s = s.lower()
    s = re.sub("[^a-zA-Z]", ' ', s)
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s.strip()

def compare_transcriptions(reference_text: str, hypothesis_text: str):
    """
    Compute the Word Error Rate (WER) between the reference and hypothesis transcriptions.
    If they differ only slightly, print the key differences in color.
    """
    reference_text = clean_str(reference_text)
    hypothesis_text = clean_str(hypothesis_text)

    try:
        w = min(1.0, float(jiwer.wer(reference_text, hypothesis_text)))
        c = min(1.0, float(jiwer.cer(reference_text, hypothesis_text)))
        return (w, c)
    except Exception as e:
        return (None, None)
    

def print_colored_diff(ref: str, hyp: str) -> None:
    """
    Prints a word-level diff between the reference and hypothesis transcripts,
    highlighting removals in red and additions in green.
    """
    ref = clean_str(ref)
    hyp = clean_str(hyp)

    ref_words = ref.split()
    hyp_words = hyp.split()
    diff = list(difflib.ndiff(ref_words, hyp_words))
    colored_tokens = []
    for token in diff:
        if token.startswith('- '):
            colored_tokens.append("\033[91m" + token + "\033[0m")  # red for removals
        elif token.startswith('+ '):
            colored_tokens.append("\033[92m" + token + "\033[0m")  # green for additions
        elif token.startswith('? '):
            pass
        else:
            colored_tokens.append(token)
    print(" ".join(colored_tokens))