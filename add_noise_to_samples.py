import os
import glob
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import shutil

# Constants for noise levels.
NOISE_LEVELS = {
    "large": 0.9,
    "mid": 0.6,
    "low": 0.3,
}
NO_NOISE_LABEL = "no"

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Loads audio data and sample rate from a file.

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple containing the audio data (as a numpy array) and the sample rate.
    """
    try:
        data, sample_rate = sf.read(file_path, dtype='float32')
        return data, sample_rate
    except sf.LibsndfileError as e:
        raise ValueError(f"Error loading audio file '{file_path}': {e}") from e

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalizes audio data to the range [-1, 1].  Handles empty audio gracefully.

    Args:
        audio_data: The audio data as a numpy array.

    Returns:
        The normalized audio data.
    """
    max_val = np.max(np.abs(audio_data))
    return audio_data / max_val if max_val > 0 else audio_data  # Avoids division by zero

def mix_audio(raw_audio: np.ndarray, noise_audio: np.ndarray, noise_level: float) -> np.ndarray:
    """Mixes two audio arrays, handling different lengths and channel counts.

    Args:
        raw_audio: The primary audio data (e.g., clean speech).
        noise_audio: The noise audio data.
        noise_level: The scaling factor for the noise.

    Returns:
        The mixed audio data.  The shape will match that of raw_audio.

    Raises:
        ValueError: if raw_audio and noise_audio have incompatible shapes and
            cannot be broadcast together.
    """
    raw_len = raw_audio.shape[0]
    noise_len = noise_audio.shape[0]

    # Adjust noise length to match raw audio
    if noise_len < raw_len:
        # Loop noise if shorter
        repeats = int(np.ceil(raw_len / noise_len))
        noise_audio = np.tile(noise_audio, (repeats, 1) if noise_audio.ndim > 1 else repeats)
    noise_audio = noise_audio[:raw_len]  # Trim to exact raw length

    # Handle channel mismatches (mono to stereo, stereo to mono)
    if raw_audio.ndim != noise_audio.ndim:
        if raw_audio.ndim == 1 and noise_audio.ndim == 2:
            raw_audio = np.column_stack((raw_audio, raw_audio))  # Duplicate mono to stereo
        elif raw_audio.ndim == 2 and noise_audio.ndim == 1:
            noise_audio = np.column_stack((noise_audio, noise_audio))  # Duplicate mono to stereo

    # Mix the signals, ensuring compatible shapes
    try:
      mixed_audio = raw_audio + (noise_level * noise_audio)
    except ValueError as e:
      raise ValueError("Audio shapes are incompatible for mixing. "
                       f"Raw audio shape: {raw_audio.shape}, "
                       f"Noise audio shape: {noise_audio.shape}") from e

    return mixed_audio


def record_playback(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Plays audio while simultaneously recording from the microphone.

    Args:
        audio_data: The audio data to play.
        sample_rate: The sample rate of the audio.

    Returns:
        The recorded audio data.
    """
    channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
    print("Playback and recording started...")
    recording = sd.playrec(audio_data, samplerate=sample_rate, channels=channels)
    sd.wait()  # Wait for playback and recording to finish
    print("Recording complete.")
    return recording

def ensure_directory_exists(folder_path: str) -> None:
    """Creates a directory if it doesn't already exist.  Handles errors gracefully.

    Args:
        folder_path: The path to the directory.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)  # exist_ok=True prevents raising if dir exists
    except OSError as e:
        print(f"Error creating directory '{folder_path}': {e}")
        # Depending on the error, you might choose to exit, re-raise, or continue

def check_all_recordings_done(output_dir: str, base_name: str, noise_files: list[str]) -> bool:
    """Checks if all expected output files exist for a given input file.

    Args:
        output_dir: The output directory.
        base_name: The base name of the input file (without extension).
        noise_files: A list of paths to noise files.

    Returns:
        True if all expected files exist, False otherwise.
    """
    for noise_filepath in noise_files:
        noise_name = os.path.splitext(os.path.basename(noise_filepath))[0]
        for label in NOISE_LEVELS:
            noisy_filename = os.path.join(output_dir, f"{base_name}_{noise_name}_{label}.wav")
            if not os.path.exists(noisy_filename):
                return False  # Missing a noisy file
    no_noise_filename = os.path.join(output_dir, f"{base_name}_{NO_NOISE_LABEL}.wav")
    if not os.path.exists(no_noise_filename):
        return False  # Missing the no-noise file
    return True  # All files exist

def process_audio_file(input_filepath: str, noise_files: list[str], output_folder: str) -> None:
    """Processes a single audio file, adding noise and recording playback.

    Args:
        input_filepath: Path to the input audio file.
        noise_files: List of paths to noise files.
        output_folder: Base output directory.
    """
    # Construct output directory
    relative_path = os.path.relpath(input_filepath, input_folder)
    directory, filename = os.path.split(relative_path)
    base_name, _ = os.path.splitext(filename)
    output_dir = os.path.join(output_folder, directory)
    ensure_directory_exists(output_dir)

    print(f"\nProcessing file: {input_filepath}")

    raw_audio, sample_rate = load_audio(input_filepath)
    raw_audio = normalize_audio(raw_audio)

    # --- Noise Level 0 (No Noise) Handling ---
    no_noise_filename = os.path.join(output_dir, f"{base_name}_{NO_NOISE_LABEL}.wav")
    if not os.path.exists(no_noise_filename):
        print("Recording no-noise sample...")
        no_noise_recording = record_playback(raw_audio, sample_rate)
        sf.write(no_noise_filename, no_noise_recording, sample_rate)
        print(f"No-noise recording saved as '{no_noise_filename}'")
    else:
        print(f"No-noise file '{no_noise_filename}' already exists. Skipping...")

    # --- Noise Mixing and Recording ---
    for noise_filepath in noise_files:
        noise_name = os.path.splitext(os.path.basename(noise_filepath))[0]
        print(f"Processing noise file: {noise_filepath}")
        noise_audio, noise_sample_rate = load_audio(noise_filepath)

        # Resample noise if necessary
        if sample_rate != noise_sample_rate:
            print(f"Resampling noise from {noise_sample_rate} Hz to {sample_rate} Hz")
            noise_audio = resample_poly(noise_audio, sample_rate, noise_sample_rate)
        noise_audio = normalize_audio(noise_audio)

        # Record noisy samples for each noise level
        for label, noise_level in NOISE_LEVELS.items():
            noisy_filename = os.path.join(output_dir, f"{base_name}_{noise_name}_{label}.wav")
            if os.path.exists(noisy_filename):
                print(f"File '{noisy_filename}' already exists. Skipping...")
                continue

            print(f"Recording noisy sample (level: {label}) with noise file '{noise_name}'...")
            mixed_audio = mix_audio(raw_audio, noise_audio, noise_level)
            noisy_recording = record_playback(mixed_audio, sample_rate)
            sf.write(noisy_filename, noisy_recording, sample_rate)
            print(f"Noisy recording saved as '{noisy_filename}'")

def copy_input_files(input_filepath: str, output_dir: str) -> None:
    """Copies all files from the input file's directory to the output directory.
    
    Args:
        input_filepath: The path to the processed input audio file.
        output_dir: The destination directory.
    """
    input_dir = os.path.dirname(input_filepath)
    print(f"Copying input files from '{input_dir}' to '{output_dir}'")
    for file_to_copy in glob.glob(os.path.join(input_dir, "*")):
        if os.path.isfile(file_to_copy):  # Ensure it's a file, not a directory
            shutil.copy(file_to_copy, output_dir)

def main():
    """Main function to process audio files."""
    global input_folder  # Access global input_folder (defined within process_audio_file)
    # --- Setup ---
    noise_folder = "noise"
    input_folder = "in"
    output_folder = "out"

    # Find noise files
    noise_files = glob.glob(os.path.join(noise_folder, "*.wav"), recursive=True)
    if not noise_files:
        print(f"No noise files found in folder '{noise_folder}'.")
        return

    # Find input files
    input_files = glob.glob(os.path.join(input_folder, '**', '*.mp3'), recursive=True)
    if not input_files:
        print(f"No MP3 files found in folder '{input_folder}'.")
        return

    ensure_directory_exists(output_folder)

    # --- Main Processing Loop ---
    for input_filepath in input_files:
        process_audio_file(input_filepath, noise_files, output_folder)
        
        # --- Check and Copy after each file---
        relative_path = os.path.relpath(input_filepath, input_folder)
        directory, filename = os.path.split(relative_path)
        base_name, _ = os.path.splitext(filename)
        output_dir = os.path.join(output_folder, directory)
        
        if check_all_recordings_done(output_dir, base_name, noise_files):
          copy_input_files(input_filepath, output_dir)


    print("\nProcessing complete for all files.")

if __name__ == '__main__':
    main()