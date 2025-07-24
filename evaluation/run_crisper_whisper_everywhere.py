#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output

def transcribe_audio(audio_path: Path, pipe, language: str = None) -> str:
    """
    Transcribes a given audio file using the CrisperWhisper model.
    Returns the transcription text.
    """
    print(f"Processing: {audio_path}")
    # Call the pipeline; language parameter is ignored for now.
    result = pipe(str(audio_path))
    # Optionally adjust pauses/timestamps if needed.
    result = adjust_pauses_for_hf_pipeline_output(result)
    # Extract and return the transcription text.
    full_text = result.get("text", "")
    return full_text

def main():
    parser = argparse.ArgumentParser(
        description="Recursively transcribe WAV audio files using the CrisperWhisper model."
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
        help="Specify the language for transcription (currently not used)"
    )
    parser.add_argument(
        "-g", "--gpu",
        type=int,
        default=0,
        help="If the GPU is used, this is the device id (i.e. 0 for first GPU, 1 for second GPU, etc.)"
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
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    print(f"Using device: {device}")

    # Recursively find all WAV files in the folder.
    wav_files = list(root_folder.rglob("*.wav")) 
    if not wav_files:
        print("No WAV files found.")
        return

    # Filter files that need transcribing (transcription file doesn't exist or is empty).
    files_to_transcribe = []
    for audio_path in wav_files:
        txt_path = audio_path.with_suffix(".cwtxt")
        if args.force:
            files_to_transcribe.append(audio_path)
            continue

        if txt_path.exists() and txt_path.stat().st_size > 0:
            with open(txt_path, "r") as f:
                first_char = f.read(1)
            if first_char == "[":
                files_to_transcribe.append(audio_path)
            else:
                print(f"Skipping {audio_path} as {txt_path} already exists and is not empty.")
        else:
            files_to_transcribe.append(audio_path)

    if not files_to_transcribe:
        print("No files need transcribing.")
        return

    # Load the CrisperWhisper model using the Hugging Face transformers pipeline.
    model_id = "nyrahealth/CrisperWhisper"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager"
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps='word',
        torch_dtype=torch_dtype,
        device=device
    )
    print(f"Transcribing {len(files_to_transcribe)} file(s).")

    for audio_path in files_to_transcribe:
        txt_path = audio_path.with_suffix(".txt")
        try:
            transcription_text = transcribe_audio(audio_path, pipe, language=args.language)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcription_text)
            print(f"Transcription saved to: {txt_path}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    main()
