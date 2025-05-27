#!/usr/bin/env python3
import os
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment

def extract_transcript_segments(url):
    """
    Fetches the transcript page and returns a list of tuples (timestamp, text)
    for each <p> tag that has a data-timestamp attribute.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve page. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    # Assuming the transcript is inside an <article> tag.
    transcript_container = soup.find('article')
    segments = []
    if transcript_container:
        p_tags = transcript_container.find_all('p', attrs={'data-timestamp': True})
        for p in p_tags:
            ts_str = p.get('data-timestamp').strip()
            try:
                ts = float(ts_str)
            except ValueError:
                continue  # Skip if timestamp is not a valid float
            # Get transcript text without additional formatting.
            text = p.get_text(separator=' ', strip=True)
            segments.append((ts, text))
    # Ensure segments are sorted by timestamp
    segments.sort(key=lambda x: x[0])
    return segments

def compute_boundaries(transcript_segments, segment_duration=60.0):
    """
    Computes segmentation boundaries using the rule:
    Start at the first transcript timestamp, then every time a transcript timestamp
    is at least 'segment_duration' seconds after the previous boundary, add it.
    """
    boundaries = []
    if not transcript_segments:
        return boundaries
    # Start at the first transcript timestamp.
    last_boundary = transcript_segments[0][0]
    boundaries.append(last_boundary)
    for ts, _ in transcript_segments:
        if ts - last_boundary >= segment_duration:
            boundaries.append(ts)
            last_boundary = ts
    return boundaries

def segment_audio(audio_path, transcript_segments, segment_duration=60.0):
    """
    Loads the audio file, cuts it into segments based on computed boundaries, and saves:
      - The audio segment (as audio.mp3)
      - A transcript file (transcript.txt) containing transcript texts (without timestamps)
    into an output folder structure: output/segment_<id>/.
    """
    boundaries = compute_boundaries(transcript_segments, segment_duration)
    if not boundaries:
        raise Exception("No segmentation boundaries found in transcript segments.")
    
    # Load the audio file.
    audio = AudioSegment.from_mp3(audio_path)
    audio_length_sec = len(audio) / 1000.0  # pydub works in milliseconds
    
    output_dir = f"tal_{n}"
    os.makedirs(output_dir, exist_ok=True)
    
    num_segments = len(boundaries)
    for i in range(num_segments):
        seg_start = boundaries[i]
        # Use the next boundary if available; otherwise, use the full length of the audio.
        seg_end = boundaries[i+1] if i+1 < num_segments else audio_length_sec
        
        # Convert seconds to milliseconds.
        start_ms = int(seg_start * 1000)
        end_ms = int(seg_end * 1000)
        
        # Cut the audio segment.
        segment_audio = audio[start_ms:end_ms]
        
        # Create a unique folder for this segment.
        seg_folder = os.path.join(output_dir, f"segment_{i+1}")
        os.makedirs(seg_folder, exist_ok=True)
        
        # Save the audio segment.
        audio_filename = os.path.join(seg_folder, "audio.mp3")
        segment_audio.export(audio_filename, format="mp3")
        
        # Gather transcript texts that fall within this segment.
        segment_transcripts = [text for ts, text in transcript_segments if seg_start <= ts < seg_end]
        transcript_text = "\n".join(segment_transcripts)
        
        # Save the transcript file (without timestamps).
        transcript_filename = os.path.join(seg_folder, "transcript.txt")
        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        print(f"Segment {i+1}: {seg_start:.2f}s to {seg_end:.2f}s saved to {seg_folder}")

if __name__ == '__main__':
    n = 848
    transcript_url = f"https://www.thisamericanlife.org/{n}/transcript"
    audio_file = f"{n}.mp3"
    try:
        # Step 1: Extract transcript segments from the webpage.
        transcript_segments = extract_transcript_segments(transcript_url)
        if not transcript_segments:
            raise Exception("No transcript segments found.")
        
        # Step 2: Process the audio file and segment it.
        segment_audio(audio_file, transcript_segments)
    except Exception as e:
        print(f"An error occurred: {e}")
