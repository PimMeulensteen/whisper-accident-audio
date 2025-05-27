import argparse
from pathlib import Path
import torch
import whisperx

def transcribe_file(wav_path: Path, model, batch_size: int = 16) -> None:
    """
    Transcribe a single WAV file and save the output as a TXT file with the same basename.
    """
    print(f"Transcribing: {wav_path.name}")
    # Load audio at 16kHz
    audio = whisperx.load_audio(str(wav_path))
    # Perform batched transcription via model
    result = model.transcribe(audio, batch_size=batch_size)
    # Extract full text from segments
    segments = result.get("segments", [])
    text = "".join(seg.get("text", "") for seg in segments)
    # Write to .txt file with same basename
    txt_path = wav_path.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print(f"Written transcript to: {txt_path.name}\n")

def transcribe_path(path: Path, model_size: str = "large-v3", batch_size: int = 16) -> None:
    """
    Transcribe a single WAV file or all WAV files in a directory using WhisperX.
    """
    # Determine device based on PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load WhisperX model
    model = whisperx.load_model(model_size, device=device)

    if path.is_file() and path.suffix.lower() == ".wav":
        transcribe_file(path, model, batch_size)
    elif path.is_dir():
        for wav_file in sorted(path.glob("*.wav")):
            transcribe_file(wav_file, model, batch_size)
    else:
        raise ValueError(f"Path must be a .wav file or a directory: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe WAV file(s) using WhisperX (large-v3) and save .txt outputs."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .wav file or folder containing .wav files to transcribe."
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription (default: 16)."
    )
    args = parser.parse_args()

    transcribe_path(args.path, batch_size=args.batch_size)