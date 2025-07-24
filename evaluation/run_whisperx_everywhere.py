#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import whisperx

def transcribe_audio(audio_path: Path, model, device: str, language: str = None) -> str:
    """
    Transcribes a given audio file using the WhisperX model.
    Returns the concatenated transcription text from all segments.
    """
    if language:
        result = model.transcribe(str(audio_path), language=language, batch_size=16)
    else:
        result = model.transcribe(str(audio_path))
    
    # Concatenate the text strings from all segments.
    segments = result.get("segments", [])
    full_text = "\n".join(s['text'].strip() for s in segments)
    return full_text

def main():
    parser = argparse.ArgumentParser(
        description="Recursively transcribe WAV audio files using WhisperX large-v3 model."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="The root folder where the WAV files are located."
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Specify the language for transcription (e.g., 'en' for English)"
    )
    parser.add_argument(
        "-g", "--gpu",
        type=int,
        default=0,
        help="If the GPU is used, this is the device id (i.e. 0 is first GPU, 1 is second GPU, etc.)"
    )
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        help="Force a new transcript."
    )
    args = parser.parse_args()
    
    root_folder = Path(args.folder)
    if not root_folder.exists() or not root_folder.is_dir():
        print("Error: Provided path does not exist or is not a directory.")
        return

    # Determine the device (CUDA if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Recursively find all WAV files in the folder.
    wav_files = list(root_folder.rglob("*.wav")) 
    if not wav_files:
        print("No WAV files found.")
        return
    print(f"Found {len(wav_files)} audio files.")
    # Filter files that need transcribing (transcription file doesn't exist or is empty).
    files_to_transcribe = []
    for audio_path in wav_files:
        txt_path = audio_path.with_suffix(".txt")
        if args.force:
            files_to_transcribe.append(audio_path)
            continue

        if txt_path.exists() and txt_path.stat().st_size > 5:
            with open(txt_path, "r") as f:
                first_char = f.read(1)
            if first_char == "[":
                files_to_transcribe.append(audio_path)
        else:
            files_to_transcribe.append(audio_path)

    if not files_to_transcribe:
        print("No files need transcribing.")
        return


    # Load the WhisperX model (large-v3) only if needed.
    model = whisperx.load_model("large-v3", device=device, device_index=args.gpu)
    print(f"Transcribing {len(files_to_transcribe)} file(s).")
    # files_to_transcribe.sort(key=lambda x: (str(x).count('/')))
    files_to_transcribe = list(filter(lambda x: 'out' in str(x), files_to_transcribe))
    if args.gpu != 0:
        files_to_transcribe.reverse()

    for audio_path in  tqdm(files_to_transcribe):
        # print(audio_path)
        txt_path = audio_path.with_suffix(".txt")
        
        try:
            transcription_text = transcribe_audio(audio_path, model, device, language=args.language)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcription_text)
            # print(f"Transcription saved to: {txt_path}")/


        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    main()
