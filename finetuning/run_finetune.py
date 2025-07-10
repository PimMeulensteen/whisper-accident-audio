#!/usr/bin/env python3
import argparse
import glob
import os
import torch
import whisperx
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_file(
    file_path: str,
    processor: WhisperProcessor,
    model: PeftModel,
    align_model,
    metadata: dict,
    device: str,
    chunk_sec: int,
    suffix: str,
    output_dir: str,
):
    """Transcribe a single audio file and save word-aligned timestamps."""
    audio = whisperx.load_audio(file_path)
    sr = 16_000
    chunk_size = chunk_sec * sr

    segments = []
    for start in range(0, len(audio), chunk_size):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        inputs = processor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True
        )
        feats = inputs.input_features.to(device)
        mask = inputs.attention_mask.to(device)
        # model.config.forced_decoder_ids = None

        generated_ids = model.generate(
            feats,
            attention_mask=mask,
            max_length=model.config.max_target_positions,
            num_beams=8,
            no_repeat_ngram_size=3,
            language="en"
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        segments.append({"start": start/sr, "end": end/sr, "text": text})

    basename = os.path.splitext(os.path.basename(file_path))[0]
    out_name = f"{basename}{'-' if suffix else ''}{suffix}.txt"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as fout:
        for seg in segments:
            print(seg['text'])
            fout.write(f"{seg['text']}\n")

    print(f"[✓] Saved transcript: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe .wav files with one or multiple Whisper-LoRA adapters."
    )
    parser.add_argument("input_dir", type=str,
                        help="Root directory to search for .wav files")
    parser.add_argument("--model-dir", type=str, default="./whisper-lora",
                        help="Path to a LoRA adapter directory or parent folder for multiple adapters")
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3",
                        help="HuggingFace name or path of the base Whisper model")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where transcripts are saved; defaults to each audio file’s folder")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for transcript filenames (overrides auto naming)")
    parser.add_argument("--chunk-sec", type=int, default=15,
                        help="Length (in seconds) of each decoding chunk")
    parser.add_argument("--multi-models", action="store_true",
                        help="Treat --model-dir as parent containing multiple adapters")
    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load alignment model once
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    # gather model directories
    if args.multi_models:
        candidates = sorted([
            os.path.join(args.model_dir, d)
            for d in os.listdir(args.model_dir)
            if os.path.isdir(os.path.join(args.model_dir, d))
        ])
        if not candidates:
            print(f"[!] No subdirectories found in {args.model_dir}")
            return
        model_dirs = candidates
    else:
        model_dirs = [args.model_dir]

    # find .wav files
    wav_pattern = os.path.join(args.input_dir, '**', '*.wav')
    files = glob.glob(wav_pattern, recursive=True)
    if not files:
        print(f"[!] No .wav files found in {args.input_dir}")
        return

    for model_path in model_dirs:
        name = os.path.basename(model_path.rstrip(os.sep))
        suffix = args.suffix or name

        print(f"\n=== Processing with adapter: {name} ===")
        processor = WhisperProcessor.from_pretrained(model_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        peft_model.to(device).eval()

        for fp in sorted(files):
            out_dir = args.output_dir or os.path.dirname(fp)
            transcribe_file(
                file_path=fp,
                processor=processor,
                model=peft_model,
                align_model=align_model,
                metadata=metadata,
                device=device,
                chunk_sec=args.chunk_sec,
                suffix=suffix,
                output_dir=out_dir,
            )

if __name__ == "__main__":
    main()
