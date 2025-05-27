#!/usr/bin/env python3
import os
import glob
import pandas as pd

def extract_atco_parquet(data_dir='atco2-asr/data', output_dir='atco'):
    """
    Process all .parquet files in data_dir.
    For each row (expected to have columns: 'audio', 'text', and 'info'),
    create a folder under output_dir with a unique id,
    save the audio file (audio.wav) and transcript (transcript.txt).
    """
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}.")
        return

    row_id = 1
    for parquet_file in parquet_files:
        print(f"Processing file: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        for idx, row in df.iterrows():
            folder_name = f"segment_{row_id}"
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Save the audio file.
            audio_data = row['audio']
            # If audio_data is a dict, try extracting the underlying bytes.
            if isinstance(audio_data, dict):
                # Adjust the key as necessary ('data' or 'bytes')
                audio_data = audio_data.get('data') or audio_data.get('bytes')
                if audio_data is None:
                    raise ValueError("Audio data dict does not contain expected key 'data' or 'bytes'.")
                # If the result is a list, convert it to bytes.
                if isinstance(audio_data, list):
                    audio_data = bytes(audio_data)
            
            # At this point, audio_data should be a bytes-like object.
            audio_file = os.path.join(folder_path, "audio.wav")
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            
            # Save the transcript text.
            transcript = row['text']
            transcript_file = os.path.join(folder_path, "transcript.txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            print(f"Saved segment {row_id} to {folder_path}")
            row_id += 1

if __name__ == '__main__':
    extract_atco_parquet()
