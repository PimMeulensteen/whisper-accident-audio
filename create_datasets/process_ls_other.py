#!/usr/bin/env python3

import argparse
from pathlib import Path
# from pydub import AudioSegment

# List of audio file extensions to process
SUPPORTED_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac',
    '.wav', '.wma', '.alac', '.aiff'
}

# def convert_audio_files(input_dir: Path, output_dir: Path):
#     """
#     Recursively find all supported audio files in input_dir,
#     convert each to WAV, and save as audio.wav in a folder named
#     after the original file (without extension) inside output_dir.
#     """
#     for src_path in input_dir.rglob('*'):
#         if not src_path.is_file() or src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
#             continue

#         dest_subdir = output_dir / src_path.stem
#         dest_subdir.mkdir(parents=True, exist_ok=True)

#         dest_file = dest_subdir / 'audio.wav'
#         audio = AudioSegment.from_file(src_path)
#         audio.export(dest_file, format='wav')

#         print(f"Converted: {src_path} → {dest_file}")

def distribute_text_files(input_dir: Path, output_dir: Path):
    """
    Read all .txt files under input_dir. Each line should start with a folder name
    (no spaces), followed by a space, then the text. For each line, create a
    .txt file containing the text, and place it inside output_dir/<FOLDER_NAME>/.
    Filenames are generated as <source-txt-filename>_<line-number>.txt.
    """
    for txt_path in input_dir.rglob('*.txt'):
        with txt_path.open('r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    # skip lines without a space
                    continue
                folder_name, content = parts
                dest_subdir = output_dir / folder_name
                dest_subdir.mkdir(parents=True, exist_ok=True)

                dest_file = dest_subdir / f"transcript.txt"
                with dest_file.open('w', encoding='utf-8') as out_f:
                    out_f.write(content)

                print(f"Wrote text: {dest_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to WAV and distribute text lines into folders."
    )
    parser.add_argument(
        'input_folder',
        type=Path,
        help="Path to the input folder (will search recursively)"
    )
    parser.add_argument(
        'output_folder',
        type=Path,
        help="Path to the output folder (will be created if it doesn't exist)"
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Input folder does not exist or is not a directory: {args.input_folder}")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    # convert_audio_files(args.input_folder, args.output_folder)
    distribute_text_files(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main()
