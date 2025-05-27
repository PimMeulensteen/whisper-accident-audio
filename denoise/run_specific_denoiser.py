#!/usr/bin/env python3
"""
Audio separation script (WAV only), using a single specified model (denoiser) by filename.
Outputs denoised files next to input files with '_denoised' suffix.
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import List
from audio_separator import separator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def setup_tmp_dir(input_dir: str) -> Path:
    tmp_dir = Path(input_dir) / "_tmp_output"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def get_wav_files(input_dir: str) -> List[Path]:
    wav_files = []
    with os.scandir(input_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".wav"):
                wav_files.append(Path(entry.path))
    return wav_files


def clean_tmp_dir(tmp_dir: Path) -> None:
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(exist_ok=True)


def write_output(audio: Path, tmp_dir: Path) -> None:
    outputs = list(tmp_dir.glob("*.wav"))
    if not outputs:
        logger.warning(f"No output for {audio}")
        return
    for file in outputs:
        dest = audio.parent / f"{audio.stem}_denoised{file.suffix}"
        try:
            shutil.move(str(file), str(dest))
            logger.info(f"Wrote denoised file: {dest}")
        except Exception as e:
            logger.error(f"Error moving {file} to {dest}: {e}")


def process_files(input_dir: str, model_dir: str, force: bool, denoiser: str) -> None:
    tmp_dir = setup_tmp_dir(input_dir)
    audio_files = get_wav_files(input_dir)
    seg_size = 900

    sep_kwargs = {
        "output_dir": str(tmp_dir),
        "model_file_dir": model_dir,
        "mdxc_params": {"segment_size": seg_size, "override_model_segment_size": False}
    }
    sep = separator.Separator(**sep_kwargs)
    supported = sep.list_supported_model_files()

    # Flatten all models into a list
    all_models = [m for models in supported.values() for m in models.values()]
    # Find the specified model by filename
    model_entry = next((m for m in all_models if Path(m["filename"]).name == denoiser), None)
    if not model_entry:
        logger.error(f"Model '{denoiser}' not found. Supported models: {[Path(m['filename']).name for m in all_models]}")
        return

    model_filename = model_entry["filename"]
    model_name = Path(model_filename).stem
    logger.info(f"Using model: {model_filename}")

    try:
        sep.load_model(model_filename)
    except Exception as e:
        logger.error(f"Error loading model {model_filename}: {e}")
        return

    for audio in audio_files:
        dest = audio.parent / f"{audio.stem}_denoised.wav"
        if dest.exists() and not force:
            logger.info(f"Skipping {audio}, output already exists.")
            continue

        logger.info(f"Processing {audio} with model {model_name}")
        try:
            clean_tmp_dir(tmp_dir)
            sep.separate(str(audio))
            write_output(audio, tmp_dir)
        except Exception as e:
            logger.error(f"Error processing {audio}: {e}")

    clean_tmp_dir(tmp_dir)
    logger.info("Processing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process .wav audio files using a specified model and write denoised outputs next to inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--model_dir", type=str, default="/home/pim/rp/denoisers/_uvr_models", help="Model directory")
    parser.add_argument(
        "--denoiser", type=str,
        default="mel_band_roformer_vocals_fv4_gabox.ckpt",
        help="Filename of the denoiser model to use"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    process_files(args.input_dir, args.model_dir, args.force, args.denoiser)

if __name__ == "__main__":
    main()
