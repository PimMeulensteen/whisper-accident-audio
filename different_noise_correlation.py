import os
import glob
import soundfile as sf
import numpy as np


def compute_normalized_cross_correlation(file1: str, file2: str) -> float:
    data1, sr1 = sf.read(file1)
    data2, sr2 = sf.read(file2)
    if sr1 != sr2:
        raise ValueError(f"Sampling rates differ: {sr1} vs {sr2}")
    if data1.ndim > 1:
        data1 = np.mean(data1, axis=1)
    if data2.ndim > 1:
        data2 = np.mean(data2, axis=1)
    n = min(len(data1), len(data2))
    n = sr1
    a = data1[:n] - np.mean(data1[:n])
    b = data2[:n] - np.mean(data2[:n])
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0:
        return 0.0
    corr = np.correlate(a, b, mode='full') / denom
    return float(np.max(corr))



def loop_over_segments(root_dir: str) -> dict:
    # Summary: noise_key -> denoiser_key -> list of scores across segments
    summary = {}
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.aif', '.aiff', '.m4a'}

    for seg in glob.glob(os.path.join(root_dir, '*')):
        print(seg)
        if not os.path.isdir(seg):
            continue
            
        seg_name = os.path.basename(seg)
        # find clean original
        orig_file = next((f for f in os.listdir(seg) if f.lower().endswith('_no.wav')), None)
        if not orig_file:
            continue
        orig_path = os.path.join(seg, orig_file)
        out_root = os.path.join(seg, 'out')
        if not os.path.isdir(out_root):
            continue

        prefix = seg_name + '_'
        # iterate noise directories for this segment
        for noise_dir in os.listdir(out_root):
            if not noise_dir.startswith(prefix):
                continue
            noise_key = noise_dir[len(prefix):]
            full_noise_dir = os.path.join(out_root, noise_dir)
            if not os.path.isdir(full_noise_dir):
                continue

            # inspect denoised files
            for filepath in glob.glob(os.path.join(full_noise_dir, '**', '*'), recursive=True):
                if not os.path.isfile(filepath) or os.path.splitext(filepath)[1].lower() not in audio_exts:
                    continue
                try:
                    score = compute_normalized_cross_correlation(orig_path, filepath)
                except Exception:
                    score = None
                base = os.path.splitext(os.path.basename(filepath))[0]
                # strip prefix from filename key
                if base.startswith(noise_dir + '_'):
                    denoiser_key = base[len(noise_dir) + 1:]
                else:
                    denoiser_key = base

                # append score to summary list
                nd = summary.setdefault(noise_key, {})
                nd.setdefault(denoiser_key, []).append(score)

    return summary


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    results = loop_over_segments(input_dir)
    import json
    with open("out.json", 'w') as f:
        json.dump(results, f, indent=2)
    # print grouped by noise type with lists of scores per denoiser
    for noise_key, denoiser_map in results.items():
        print(f"{noise_key}: {{")
        for denoiser_key, scores in denoiser_map.items():
            print(f"  {denoiser_key}: {scores}")
        print("}")
