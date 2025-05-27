#!/usr/bin/env python3
"""
Audio separation script (WAV only), rejecting models that don't produce vocals or noise.
This version runs entirely in a single process.
"""

import os
import shutil
import argparse
import logging
import re
from pathlib import Path
from typing import List, Set
from audio_separator import separator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants for marker filenames
REJECT_FILE = "model_reject_list.txt"   
COMPLETE_FILE = "model_complete_file.txt"
PROCESS_MARKER = "processing_complete.txt"


def setup_tmp_dir(input_dir: str) -> Path:
    tmp_dir = Path(input_dir) / "_tmp_output"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir



def get_wav_files(input_dir: str) -> List[Path]:
    wav_files = []
    # Use os.scandir for faster directory iteration
    with os.scandir(input_dir) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith("segment_") and not entry.name.endswith("out"):
                with os.scandir(entry.path) as seg_entries:
                    for file_entry in seg_entries:
                        if file_entry.is_file() and file_entry.name.endswith(".wav"):
                            wav_files.append(Path(file_entry.path))
    return wav_files

VOCALS_NOISE_REGEX = re.compile(r'\(vocals\)|\(no.?noise\)', re.IGNORECASE)
def contains_vocals_or_noise(filename: str) -> bool:
    return bool(VOCALS_NOISE_REGEX.search(filename))


def has_vocals_or_noise(tmp_dir: Path) -> bool:
    return any(f.is_file() and contains_vocals_or_noise(f.name) for f in tmp_dir.rglob("*"))


def clean_tmp_dir(tmp_dir: Path) -> None:
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(exist_ok=True)


def move_output_files(input_dir: str, segment: str, audio_name: str, model_name: str, tmp_dir: Path) -> int:
    if not has_vocals_or_noise(tmp_dir):
        logger.debug(f"Model {model_name} did not produce vocals/noise for {audio_name}.")
        return 0

    out_dir = Path(input_dir) / segment / "out" / Path(audio_name).stem / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    base = Path(audio_name).stem

    for file in tmp_dir.rglob(f"{base}*"):
        if file.is_file() and contains_vocals_or_noise(file.name):
            try:
                shutil.move(str(file), str(out_dir / file.name))
                moved += 1
            except Exception as e:
                logger.error(f"Error moving {file}: {e}")
    (out_dir / PROCESS_MARKER).write_text("Processing complete")
    logger.info(f"Moved {moved} files for {audio_name} with model {model_name}")
    return moved


def load_list(file: Path) -> Set[str]:
    return {line.strip() for line in file.read_text().splitlines()} if file.exists() else set()


def append_to_file(file: Path, model: str) -> None:
    with file.open("a") as f:
        f.write(model + "\n")

def process_files(input_dir: str, model_dir: str, force: bool) -> None:
    tmp_dir = setup_tmp_dir(input_dir)
    audio_files = get_wav_files(input_dir)
    reject_list = load_list(Path(input_dir) / REJECT_FILE)
    processed_models = set()
    seg_size = 900
    sep = separator.Separator(
        output_dir=str(tmp_dir),
        model_file_dir=model_dir,
        # mdx_params={"segment_size": seg_size},
        # demucs_params={"segment_size": seg_size},
        # vr_params={"segment_size": seg_size},
        mdxc_params={
            "segment_size": seg_size,
            "override_model_segment_size": False,
        },
    )
    available_models = sep.list_supported_model_files()

    for model_type, models in available_models.items():
        for model in models.values():
            model_filename = model["filename"]
            model_name = Path(model_filename).stem

            if model_name in reject_list:
                logger.info(f"Skipping model {model_name} (in reject list)")
                continue

            # Check if any audio file requires processing for this model.
            audio_to_process = []
            for audio in audio_files:
                segment = audio.parent.name
                marker = Path(input_dir) / segment / "out" / audio.stem / model_name / PROCESS_MARKER
                if not (marker.exists() and not force):
                    audio_to_process.append(audio)

            if not audio_to_process:
                logger.info(
                    f"Skipping model {model_name}: No audio files require processing (already processed)."
                )
                continue

            # Now load the model since processing is needed.
            logger.info(f"Loading model: {model_filename}")
            try:
                sep.load_model(model_filename)
            except Exception as e:
                logger.error(f"Error loading model {model_filename}: {e}")
                continue

            model_failed = False
            for audio in audio_to_process:
                segment = audio.parent.name
                marker = Path(input_dir) / segment / "out" / audio.stem / model_name / PROCESS_MARKER
                logger.info(f"Processing {audio} with model {model_name}")
                try:
                    clean_tmp_dir(tmp_dir)
                    sep.separate(str(audio))
                    moved = move_output_files(input_dir, segment, audio.name, model_name, tmp_dir)
                    if not moved:
                        logger.warning(
                            f"Model {model_name} produced output without vocals/noise for {audio}. Rejecting model."
                        )
                        append_to_file(Path(input_dir) / REJECT_FILE, model_name)
                        model_failed = True
                        break
                except Exception as e:
                    logger.error(f"Error processing {audio} with model {model_name}: {e}")
                    model_failed = True

                if model_failed:
                    break

            processed_models.add(model_name)
            # append_to_file(Path(input_dir) / COMPLETE_FILE, model_name)

    clean_tmp_dir(tmp_dir)
    logger.info("-" * 50)
    logger.info(f"Processing complete. Models used successfully: {', '.join(sorted(processed_models))}")
    logger.info("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process .wav audio files and organize outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--model_dir", type=str, default="/home/pim/rp/denoisers/_uvr_models", help="Model directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    process_files(args.input_dir, args.model_dir, args.force)


if __name__ == "__main__":
    main()
