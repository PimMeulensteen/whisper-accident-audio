#!/usr/bin/env python3
"""
Audio separation script (WAV only), prioritizing simplicity.

Adds models to reject list ONLY if they produce output (for WAV files),
but that output doesn't contain "vocal" or "noise".
"""

import os
import shutil
import argparse
import logging
import re
from typing import List, Set
from pathlib import Path

from audio_separator import separator
import torch
import torchaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROCESSING_COMPLETE_FILE = "processing_complete.txt"


def setup_tmp_directory(input_dir: str) -> str:
    """Create and return the path to the temporary output directory."""
    tmp_output_folder = os.path.join(input_dir, "_tmp_output")
    os.makedirs(tmp_output_folder, exist_ok=True)
    return tmp_output_folder


def get_audio_files(input_dir: str) -> List[str]:
    """Find all .wav files in all segment directories."""
    audio_files = []
    input_path = Path(input_dir)
    for segment_dir in input_path.glob("segment_*"):
        if segment_dir.is_dir() and not segment_dir.name.endswith('out'):
            audio_files.extend([str(f) for f in segment_dir.glob("*.wav")])  # Only .wav files
    return audio_files


def clean_tmp_directory(tmp_output_folder: str) -> None:
    """Remove all files and directories in the temporary directory."""
    try:
        shutil.rmtree(tmp_output_folder)
        os.makedirs(tmp_output_folder, exist_ok=True)
        logger.debug(f"Cleaned temporary directory: {tmp_output_folder}")
    except OSError as e:
        logger.error(f"Error cleaning up {tmp_output_folder}: {e}")

from df.enhance import enhance, init_df, load_audio, save_audio
df_model, df_state, _ = init_df() 
def DeepFilterNet_wrapper(inp,output):
     # Load default model
    audio, _ = load_audio(inp, sr=df_state.sr())
    enhanced_audio = enhance(df_model, df_state, audio)
    save_audio(output, enhanced_audio, df_state.sr())

from resemble_enhance.enhancer.inference import denoise

def resemble_enhance_wrapper(inp, outp):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dwav, sr = torchaudio.load(inp)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    if wav1.dim() == 1:
        wav1 = wav1.unsqueeze(0)    
    torchaudio.save(outp, wav1, new_sr, format='wav')

from speechbrain.inference.enhancement import SpectralMaskEnhancement
enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

def metricgan_plus_voicebank_wrapper(inp, output):
    # Load and add fake batch dimension
    noisy = enhance_model.load_audio(inp).unsqueeze(0)

    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

    # Saving enhanced signal on disk
    torchaudio.save(output, enhanced.cpu(), 16000)

import os
import shutil
import tempfile
from attenuate import Denoiser

def attenuate_wrapper(inp, output):
    if os.path.splitext(os.path.basename(output))[1]:
        output_folder = os.path.dirname(output)
    else:
        output_folder = output
    
    # If the output folder exists, check if any .wav file is already present.
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            if file.lower().endswith('.wav') and os.path.isfile(os.path.join(output_folder, file)):
                print(f"Skipping; {file} already exists.")
                return
    else:
        os.makedirs(output_folder, exist_ok=True)

    model = Denoiser()
    model.from_pretrained("PeaBrane/aTENNuate")
    # Check if inp is a file or directory
    if os.path.isfile(inp):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get the input file name
            input_filename = os.path.basename(inp)
            # Copy the input file to the temporary directory
            temp_input_path = os.path.join(temp_dir, input_filename)
            shutil.copy2(inp, temp_input_path)
            
            # Run denoising on the temporary directory
            print(output_folder)
            model.denoise(temp_dir, denoised_dir=output_folder)
    else:
        # If inp is already a directory, proceed as normal
        model.denoise(inp, denoised_dir=output_folder)

    
def process_audio_files(input_dir: str, force: bool = False) -> None:
    """Main processing function for WAV audio files.
    
    Args:
        input_dir: Directory containing audio files to process
        force: If True, reprocess files even if output already exists
    """
    # Define model configurations
    models = [
        {
            "name": "ResembleEnhance",
            "processor": resemble_enhance_wrapper
        },
        {
            "name": "metricgan-plus",
            "processor": metricgan_plus_voicebank_wrapper
        },
        { "name": "DeepFilterNet",
         "processor": DeepFilterNet_wrapper
         },
        {
            "name": "aTENNuate",
            "processor": attenuate_wrapper
        },
    ]
    
    # Get list of audio files only once
    audio_files = get_audio_files(input_dir)
    total_files = len(audio_files) * len(models)
    processed_count = 0
    
    # Process each file with each model
    
    for model in models:
        model_name = model["name"]
        processor = model["processor"]
        for audio_file in audio_files:
   
            # Build output path
            segment_dir = Path(audio_file).parent.name
            output_path = Path(input_dir) / segment_dir / "out" / Path(audio_file).stem / model_name
            output_file = output_path / "denoised.wav"
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Check if the file exists. If so, skip it unless force is True
            if not force and output_file.exists():
                logger.info(f"{output_file} already exists; skipping!")
                processed_count += 1
                continue
                
            # Process the audio file
            try:
                processor(audio_file, output_file)
                logger.info(f"Processed {audio_file} with {model_name}")
            except Exception as e:
                logger.error(f"Error processing {audio_file} with {model_name}: {str(e)}")
            
            processed_count += 1
            logger.info(f"Progress: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%)")
    
    logger.info("-" * 50)
    logger.info(f"Processing complete. Processed {processed_count} out of {total_files} files.")
    logger.info("-" * 50)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process .wav audio files and organize outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--force", "-f", action="store_true", help="Enable force rerunning.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    process_audio_files(args.input_dir, args.force)


if __name__ == "__main__":
    main()