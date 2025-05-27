import os
import torch
from pydub import AudioSegment
import whisperx

# 1. Load the audio file and transcribe with WhisperX
audio_file = "test.mp3"

# Load a WhisperX model (change "medium" to your preferred model size)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device=device)

# First transcription pass
result = model.transcribe(audio_file, language='nl')

# Load and run the alignment model to refine segment boundaries
alignment_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], alignment_model, metadata, audio_file, device=device)

# 2. Load the full audio with pydub (pydub works in milliseconds)
audio = AudioSegment.from_file(audio_file)

# Get the segments with start/end times (in seconds) and transcript text
segments = result["segments"]

# Define minimum snippet duration in seconds (1 minute = 60 seconds)
min_duration = 60.0

# Merge consecutive segments until each snippet is at least one minute
combined_segments = []
current_snippet = None

for seg in segments:
    seg_start = seg["start"]
    seg_end = seg["end"]
    seg_text = seg["text"].strip()

    if current_snippet is None:
        # Start a new snippet
        current_snippet = {"start": seg_start, "end": seg_end, "text": seg_text}
    else:
        # Calculate current snippet duration
        current_duration = current_snippet["end"] - current_snippet["start"]
        if current_duration < min_duration:
            # Merge the current segment with the snippet
            current_snippet["end"] = seg_end
            current_snippet["text"] += " " + seg_text
        else:
            # If the snippet is long enough, save it and start a new snippet with the current segment
            combined_segments.append(current_snippet)
            current_snippet = {"start": seg_start, "end": seg_end, "text": seg_text}

# Append the last snippet (even if it is shorter than min_duration)
if current_snippet:
    combined_segments.append(current_snippet)

# 3. Export each snippet's audio and transcript into its own folder
for i, snippet in enumerate(combined_segments, start=1):
    folder_name = f"pc/segment_{i}"
    os.makedirs(folder_name, exist_ok=True)

    # Convert seconds to milliseconds for slicing with pydub
    start_ms = int(snippet["start"] * 1000)
    end_ms = int(snippet["end"] * 1000)
    snippet_audio = audio[start_ms:end_ms]

    # Save the audio snippet (you can change the format if needed)
    audio_filename = os.path.join(folder_name, f"segment_{i}.mp3")
    snippet_audio.export(audio_filename, format="mp3")
    
    # Save the transcript text
    transcript_filename = os.path.join(folder_name, f"segment_{i}.txt")
    with open(transcript_filename, "w", encoding="utf-8") as f:
        f.write(snippet["text"])

    print(f"Exported snippet {i}: {audio_filename} and {transcript_filename}")
