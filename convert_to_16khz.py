
#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile

def convert_wav(file_path):
    dir_name = os.path.dirname(file_path)
    # create temporary output file in same directory
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=dir_name, delete=False) as tmp:
        tmp_path = tmp.name
    # run ffmpeg to resample to 16 kHz
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", tmp_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # overwrite original (removes old file)
    os.replace(tmp_path, file_path)

def main(folder):
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".wav"):
                full_path = os.path.join(root, fname)
                try:
                    convert_wav(full_path)
                    print(f"Converted: {full_path}")
                except Exception as e:
                    print(f"Failed: {full_path} → {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_folder>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
