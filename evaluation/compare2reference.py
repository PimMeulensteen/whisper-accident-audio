#!/usr/bin/env python3
"""
This script compares processed audio files to a reference audio file and transcript.
It calculates metrics such as Mean Squared Error (MSE), correlation, and Word Error Rate (WER)
by comparing audio signals and transcripts.
"""

import os
import csv
import re
import difflib
import librosa
import numpy as np
from jiwer import cer, wer
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
wer_list = []

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalizes an audio signal.
    """
    return (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

def align_audio(ref: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two audio signals by trimming them to the same length.
    """
    min_len = min(len(ref), len(test))
    return ref[:min_len], test[:min_len]

def compute_mse(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Compute the Mean Squared Error between two normalized audio signals.
    """
    ref_norm = normalize_audio(ref)
    test_norm = normalize_audio(test)
    return float(np.mean((ref_norm - test_norm) ** 2))

def compute_correlation(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Compute the correlation coefficient between two normalized audio signals.
    Returns 0.0 if one of the signals is constant.
    """
    ref_norm = normalize_audio(ref)
    test_norm = normalize_audio(test)
    if np.std(ref_norm) == 0 or np.std(test_norm) == 0:
        return 0.0
    return float(np.corrcoef(ref_norm, test_norm)[0, 1])

def load_transcript(transcript_path: Union[str, Path]) -> Optional[str]:
    """
    Load a transcript from a file.
    """
    try:
        with open(str(transcript_path), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.error(f"Error reading transcript file {transcript_path}: {e}")
        return None

def get_transcript(file_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Get the transcript corresponding to an audio file.
    It looks for a .txt file with the same base name as the audio file.
    """
    file = Path(file_path)
    transcript_path = file.with_suffix(".txt")
    transcript = load_transcript(transcript_path)
    if transcript:
        # logging.info(f"Using existing WhisperX transcript: {transcript_path}")
        return transcript, str(transcript_path)
    # logging.error(f"No WhisperX transcript found for {file_path}")
    return "", ""

def clean_str(s) -> str:
    s = s.lower()
    s = s.replace("-", " ")
    regex_repl = {
        r"""(?<=[0-9])\.(?=[0-9])""": "decimal ",
        "juliet[^\t]": "juliett",
        "[^a-zA-Z]ft[^\W]": "feet",
        "clear[^e]": "cleared ",
    }
    for old_word, new_word in regex_repl.items():
        s = re.sub(old_word, new_word, s)

    replacements = {
        "ew7ab": "eurowings seven alpha bravo",
        " ew ": "eurowings",
        " thats ": "that is",
        " ok ": " okay ",
        "push back": "pushback",
        "all right": "alright",
        "take off": "takeoff",
        "goodbye": "good bye",
        " than ": " then ",
        " ft ": " feet ",
        "descent": "descend",
        "call sign": "callsign",
        "geo ocean": "geoocean",
        " past ": " passed ",
        "'": "",
        "airtaxi": "air taxi",
        "portside": "port side",
        "tugboat": "tug boat",
        "alfa": "alpha",
        "000": " thousand ",
        "00": " hundred ",
        "10 ": "ten ",
        "0": " zero ",
        "1": " one ",
        "2": " two ",
        "3": " three ",
        "4": " four ",
        "5": " five ",
        "6": " six ",
        "7": " seven ",
        "8": " eight ",
        "9": " nine ",
        "helikopter": "helicopter",
        "engined": "engine",
        " d ":  "delta ",
        " p ": "papa",
        " b ": "bravo",
        " e ":  "echo ",
        " g ":  "golf ",
        " eco ": " echo ",
        " left ": " l ", 
        " j ": " juliett ",
        " k ":  "kilo ",
        " f ": " foxtrot ",
        " oskar ": " oscar ",
        " up on ": " upon ",
        "kenneth": "kennet",
        "  ": " ",
    }

    for old_word, new_word in replacements.items():
        s = s.replace(old_word, new_word)

    rx = re.compile('\W+')
    return rx.sub(' ', s).strip().lower()

def print_colored_diff(ref: str, hyp: str) -> None:
    """
    Prints a word-level diff between the reference and hypothesis transcripts,
    highlighting removals in red and additions in green.
    """
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
    
def compare_transcriptions(reference_text: str, hypothesis_text: str):
    """
    Compute the Word Error Rate (WER) between the reference and hypothesis transcriptions.
    If they differ only slightly, print the key differences in color.
    """
    reference_text = clean_str(reference_text)
    hypothesis_text = clean_str(hypothesis_text)

    try:
        w = min(1.0, float(wer(reference_text, hypothesis_text)))
        c = min(1.0, float(cer(reference_text, hypothesis_text)))

        wer_list.append(w)
        return (w, c)
    except Exception as e:
        return (None, None)

def load_audio(file_path: Union[str, Path], sr: Optional[int] = 16000) -> Optional[np.ndarray]:
    """
    Load an audio file, resample it if needed, and normalize the volume.
    """
    return []
    try:
        audio, orig_sr = librosa.load(str(file_path), sr=None)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return librosa.util.normalize(audio) * 0.9
    except Exception as e:
        # logging.error(f"Error loading {file_path}: {e}")
        return None

def find_reference_audio(segment_path: Union[str, Path]) -> Optional[Path]:
    """
    Find a reference audio file in the given segment directory.
    Only .wav files are considered.
    """
    segment = Path(segment_path)
    candidate_files = [f for f in segment.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]
    if not candidate_files:
        return None
    # Return the file with the shortest basename as a heuristic
    return min(candidate_files, key=lambda x: len(x.stem))

def process_single_file(
    file_path: Union[str, Path],
    ref_audio: np.ndarray,
    sr: int,
    reference_text: str,
):
    """
    Process a single audio file:
    - Load and align the audio.
    - Compute MSE and correlation with the reference audio.
    - Load and compare the transcript to compute WER.
    Extracts metadata (segment, noise_type, denoiser) from the file path.
    """
    file_path = Path(file_path)
    test_audio = load_audio(file_path, sr=sr)
    if test_audio is None:
        return (file_path.name, "", "", float("nan"), float("nan"), float("nan"), float("nan"))

    # Uncomment below if you wish to compute audio metrics:
    # ref_aligned, test_aligned = align_audio(ref_audio, test_audio)
    # mse_value = compute_mse(ref_aligned, test_aligned)
    mse_value = 0
    # corr_value = compute_correlation(ref_aligned, test_aligned)
    corr_value = 0

    transcript_whisperx, _ = get_transcript(file_path)
    wer, cer = compare_transcriptions(reference_text, transcript_whisperx)

    parts = file_path.parts
    segment = parts[-5] if len(parts) >= 5 else ""
    noise_type = parts[-3] if len(parts) >= 3 else ""
    denoiser = parts[-2] if len(parts) >= 2 else ""

    return (segment, noise_type, denoiser, mse_value, corr_value, wer, cer)

def compare_audio_files(
    input_dir: Union[str, Path],
    output_csv_base: str = "comparison_results",
) -> None:
    """
    Compare processed audio files to the reference in each segment directory.
    Results are written to CSV files.
    """
    input_path = Path(input_dir)
    for segment_dir in input_path.iterdir():
        if not segment_dir.is_dir():
            continue
        if "segment" not in segment_dir.name or segment_dir.name.endswith("out"):
            continue


        reference_file = find_reference_audio(segment_dir)
        if reference_file is None:
            # logging.error(f"Error: No reference audio in {segment_dir}")
            continue

        ref_audio = load_audio(reference_file)
        if ref_audio is None:
            continue

        sr = librosa.get_samplerate(str(reference_file))

        # Load reference transcript from a .txt file with the same base name as the reference audio.
        candidate_files = [f for f in segment_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt" and (f.name.startswith('segment_') or f.name.startswith('transcript'))]
        ref_transcript_path = min(candidate_files, key=lambda x: len(x.stem))
        if not ref_transcript_path.exists():
            logging.error(f"Reference transcript not found: {ref_transcript_path}")
            continue
        reference_text = ref_transcript_path.read_text(encoding="utf-8").strip()

        out_dir = segment_dir / "out"
        if not out_dir.is_dir():
            continue

        for audio_file_dir in out_dir.iterdir():
            if not audio_file_dir.is_dir():
                continue
            print(audio_file_dir)
            csv_filename = f"{output_csv_base}_{segment_dir.name}_{audio_file_dir.name}.csv"
            print(csv_filename)
            csv_path = audio_file_dir / csv_filename
            results = []

            for model_output_dir in audio_file_dir.iterdir():
                if not model_output_dir.is_dir():
                    continue
                for file in model_output_dir.glob("*.wav"):
                    fname = file.name.lower()
                    if "(vocals)" in fname or "(no noise)" in fname or "(" not in fname:
                        result = process_single_file(file, ref_audio, sr, reference_text)
                        results.append(result)

            # Process file with no denoiser to compare the results to
            file_with_no_denoiser = segment_dir / f"{audio_file_dir.name}.wav"
            segment_val, noise_type_val, denoiser, mse_value, corr_value, wer, cer = process_single_file(
                file_with_no_denoiser, ref_audio, sr, reference_text
            )
            # Use the segment and noise type from previous result if available, otherwise fallback
            if results:
                segment_val = results[-1][0]
                noise_type_val = results[-1][1]
            results.append((segment_val, noise_type_val, "reference", mse_value, corr_value, wer, cer))

            if results:
                # Sort results by WER (last element of tuple)
                results.sort(key=lambda x: x[-1])
                try:
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["segment", "noise_type", "denoiser", "mse", "correlation", "wer_whisperx", "cer_whisperx"])
                        writer.writerows(results)
                except Exception as e:
                    logging.error(f"Error writing CSV file {csv_path}: {e}")
            else:
                logging.info(f"No files to compare for: {segment_dir.name}/{audio_file_dir.name}")

def main():
    parser = argparse.ArgumentParser(description="Compare processed audio files to a reference.")
    parser.add_argument("input_dir", type=str, help="Path to the main input directory.")
    parser.add_argument("--output_csv_base", type=str, default="comparison_results", help="Base name for output CSVs.")
    args = parser.parse_args()

    compare_audio_files(args.input_dir, args.output_csv_base)
    # print(sum(wer_list) / len(wer_list))

if __name__ == "__main__":
    main()