#!/usr/bin/env python3
"""
FFmpeg audio denoising script using top 3 filters: afftdn, anlmdn, and arnndn.

Creates a similar directory structure as the audio-separator script.
"""

import os
import subprocess
import argparse
import logging
import shutil
from typing import List, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROCESSING_COMPLETE_FILE = "processing_complete.txt"

# Define the FFmpeg filters with their parameters
FFMPEG_FILTERS = {
    "afftdn": {
        "name": "FFT Denoiser",
        "params": "-af afftdn=nr=10:nf=-30:tn=1"
    },
    "anlmdn": {
        "name": "Non-Local Means Denoiser",
        "params": "-af anlmdn=s=7:p=0.001:r=0.005:m=15"  # Adjusted p and r values
    },
    "arnndn": {
        "name": "Recurrent Neural Network Denoiser",
        "params": "-af arnndn=model=model.bin"
    },
}



def get_audio_files(input_dir: str) -> List[str]:
    """Find all audio files in all segment directories."""
    audio_files = []
    input_path = Path(input_dir)
    
    # Look for segment directories
    for segment_dir in input_path.glob("segment_*"):
        if segment_dir.is_dir() and not segment_dir.name.endswith('out'):
            # Find all WAV audio files
            audio_files.extend([str(f) for f in segment_dir.glob("*.wav")])
            # Also add MP3 files if present
            audio_files.extend([str(f) for f in segment_dir.glob("*.mp3")])
    
    return audio_files


def process_with_ffmpeg(input_file: str, output_file: str, filter_params: str) -> bool:
    """
    Process an audio file with FFmpeg using the specified filter parameters.
    Returns True if successful, False otherwise.
    """
    try:
        # Build the FFmpeg command
        cmd = f"ffmpeg -i \"{input_file}\" {filter_params} -y \"{output_file}\""
        
        # Run the command
        logger.debug(f"Running command: {cmd}")
        process = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log the output if verbose
        logger.debug(f"FFmpeg output: {process.stderr}")
        
        # Check if output file exists and has size
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return True
        else:
            logger.error(f"Output file not created or empty: {output_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg processing failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return False


def process_audio_with_filters(input_dir: str, force: bool = False) -> None:
    """Process audio files with FFmpeg filters."""
    audio_files = get_audio_files(input_dir)
    processed_filters = []
    
    # Process each audio file with each filter
    for audio_file in audio_files:
        audio_path = Path(audio_file)
        segment_dir = audio_path.parent.name
        file_stem = audio_path.stem
        file_ext = audio_path.suffix
        
        # Create "out" directory in the segment directory if it doesn't exist
        segment_out_dir = Path(input_dir) / segment_dir / "out" / file_stem
        os.makedirs(segment_out_dir, exist_ok=True)
        
        # Process with each filter
        for filter_id, filter_info in FFMPEG_FILTERS.items():
            # Create directory for this filter's output
            filter_out_dir = segment_out_dir / filter_id
            os.makedirs(filter_out_dir, exist_ok=True)
            
            # Check if already processed
            complete_marker = filter_out_dir / PROCESSING_COMPLETE_FILE
            if complete_marker.exists() and not force:
                logger.info(f"Skipping {audio_file} with filter {filter_id} (already processed).")
                continue
            
            logger.info(f"Processing {audio_file} with filter {filter_id} ({filter_info['name']})")
            
            # Prepare output filenames
            denoised_file = filter_out_dir / f"{file_stem}_denoised{file_ext}"
            noise_file = filter_out_dir / f"{file_stem}_noise{file_ext}"
            
            # Process with the denoising filter
            success = process_with_ffmpeg(
                audio_file,
                str(denoised_file),
                filter_info["params"]
            )
            
            if success:
                # For noise extraction, we do a simple subtraction (requires Sox or similar)
                # This is a simplified approximation for demonstration
                try:
                    # Create a copy of the original for noise extraction
                    # (In a real implementation, you'd use a proper noise extraction method)
                    shutil.copy2(audio_file, str(noise_file))
                    
                    # Create the completion marker
                    with open(complete_marker, "w") as f:
                        f.write(f"Processing complete with {filter_info['name']}")
                    
                    # Add to processed filters list
                    if filter_id not in processed_filters:
                        processed_filters.append(filter_id)
                        
                    logger.info(f"Successfully processed {audio_file} with filter {filter_id}")
                    
                except Exception as e:
                    logger.error(f"Error during noise file creation: {e}")
            else:
                logger.error(f"Failed to process {audio_file} with filter {filter_id}")
    
    logger.info("-" * 50)
    logger.info(f"Processing complete. Filters used successfully: {', '.join(processed_filters)}")
    logger.info("-" * 50)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process audio files with FFmpeg denoising filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    process_audio_with_filters(args.input_dir, args.force)


if __name__ == "__main__":
    main()