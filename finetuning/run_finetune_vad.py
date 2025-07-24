#!/usr/bin/env python3
import argparse
import glob
import os
import torch
import whisperx
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import webrtcvad
import collections
import numpy as np

class Frame:
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / (sample_rate * 2))
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    Filters out non-voiced audio frames.
    Yields start and end timestamps of voiced segments.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    
    start_time = 0.0 # Initialize start_time

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_time = ring_buffer[0][0].timestamp
                voiced_frames.extend(f for f, s in ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = frame.timestamp + frame.duration
                yield (start_time, end_time)
                triggered = False
                ring_buffer.clear()
                voiced_frames = []
    
    if triggered:
        end_time = frame.timestamp + frame.duration
        yield (start_time, end_time)


def transcribe_file_with_vad(
    file_path: str,
    processor: WhisperProcessor,
    model: PeftModel,
    device: str,
    suffix: str,
    output_dir: str,
    vad_aggressiveness: int,
    vad_frame_ms: int,
    vad_padding_ms: int,
):
    """Transcribe a single audio file chunk-by-chunk using VAD."""
    audio = whisperx.load_audio(file_path)
    sr = 16_000
    vad = webrtcvad.Vad(vad_aggressiveness)
    audio_int16 = (audio * 32767).astype(np.int16)
    raw_audio = audio_int16.tobytes()
    frames = frame_generator(frame_duration_ms=vad_frame_ms, audio=raw_audio, sample_rate=sr)
    segments_times = list(vad_collector(
        sample_rate=sr,
        frame_duration_ms=vad_frame_ms,
        padding_duration_ms=vad_padding_ms,
        vad=vad,
        frames=frames
    ))

    if not segments_times:
        print(f"[!] No speech detected in {os.path.basename(file_path)}")
        return

    print(f"--- Transcribing {os.path.basename(file_path)} with VAD ({len(segments_times)} segments) ---")
    basename = os.path.splitext(os.path.basename(file_path))[0]
    suffix = "-".join(suffix.split('-')[:-2] + ["vad_" + suffix.split('-')[-2],suffix.split('-')[-1]])
    out_name = f"{basename}{'-' if suffix else ''}{suffix}.txt"
    print(f"Will save as {out_name}")

    segments = []
    for i, (start_time, end_time) in enumerate(segments_times):
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chunk = audio[start_sample:end_sample]
        
        if len(chunk) < 100:
            continue

        inputs = processor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        feats = inputs.input_features.to(device, dtype=torch.float16)
        
        mask = inputs.attention_mask.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=feats,
                attention_mask=mask,
                language="en",
                task="transcribe",
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8), 
                no_repeat_ngram_size=6, 
                logprob_threshold=-1.0, 
                max_new_tokens=256,
            )
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        if text:
            segments.append({"start": start_time, "end": end_time, "text": text})
            print(f"  [{start_time:.2f} - {end_time:.2f}] {text}")


    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    
    with open(out_path, "w", encoding="utf-8") as fout:
        for seg in segments:
            print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
            fout.write(seg['text'])

    print(f"[✓] Saved VAD-segmented transcript: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe .wav files with VAD and a Whisper-LoRA model."
    )
    parser.add_argument("input_dir", type=str,
                        help="Root directory to search for .wav files")
    parser.add_argument("--model-dir", type=str, default="../../finetuned_models/whisper-atcod-2",
                        help="Path to a LoRA adapter directory or parent folder for multiple adapters")
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3",
                        help="HuggingFace name or path of the base Whisper model")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where transcripts are saved; defaults to each audio file’s folder")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for transcript filenames (overrides auto naming)")
    parser.add_argument("--multi-models", action="store_true",
                        help="Treat --model-dir as parent containing multiple adapters")
    parser.add_argument("--vad-aggressiveness", type=int, default=3, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness (0=least, 3=most aggressive)")
    parser.add_argument("--vad-frame-ms", type=int, default=30,
                        help="VAD frame duration in milliseconds")
    parser.add_argument("--vad-padding-ms", type=int, default=300,
                        help="VAD padding duration in milliseconds")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.multi_models:
        model_dirs = sorted([
            os.path.join(args.model_dir, d)
            for d in os.listdir(args.model_dir)
            if os.path.isdir(os.path.join(args.model_dir, d))
        ])
        if not model_dirs:
            print(f"[!] No subdirectories found in {args.model_dir}")
            return
    else:
        model_dirs = [args.model_dir]

    wav_pattern = os.path.join(args.input_dir, '**', '*.wav')
    files = glob.glob(wav_pattern, recursive=True)
    if not files:
        print(f"[!] No .wav files found in {args.input_dir}")
        return

    print(f"Loading base model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)

    for model_path in model_dirs:
        name = os.path.basename(model_path.rstrip(os.sep))
        suffix = args.suffix or name 

        print(f"\n=== Processing with adapter: {name} ===")
        peft_model = PeftModel.from_pretrained(base_model, model_path).eval()

        for fp in sorted(files):
            out_dir = args.output_dir or os.path.dirname(fp)
            transcribe_file_with_vad(
                file_path=fp,
                processor=processor,
                model=peft_model,
                device=device,
                suffix=suffix,
                output_dir=out_dir,
                vad_aggressiveness=args.vad_aggressiveness,
                vad_frame_ms=args.vad_frame_ms,
                vad_padding_ms=args.vad_padding_ms,
            )

if __name__ == "__main__":
    main()