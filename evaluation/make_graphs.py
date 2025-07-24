import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use LaTeX (and therefore Computer Modern) for all text
mpl.rcParams['text.usetex']    = True
mpl.rcParams['font.family']    = 'serif'
mpl.rcParams['font.serif']     = ['Computer Modern Roman']
# Ensure math expressions use Computer Modern
mpl.rcParams['mathtext.fontset'] = 'cm'

import seaborn as sns
from typing import List
import re

FILE_TYPE = "pdf"

FIG_WIDTH = 12
FIG_HEIGHT = 10

FONT_UNIT = 12

MEAN_PROPS = {
                "marker": "o",
                "markerfacecolor": "#3086ad",
                "markeredgecolor": "black",
                "markersize": FONT_UNIT//3,
                "alpha": 0.75,
            }

def clean_noise_type(noise):
    if pd.isna(noise):
        return noise
    cleaned = re.sub(r'^(audio|segment_\d+)_', '', str(noise))
    return cleaned if cleaned else "no"

def load_csv_files(input_dir: str) -> pd.DataFrame:
    data_frames = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("comparison") and file.endswith(".csv"):
                if not 'cvr_noise_large' in file:
                    continue
                df = pd.read_csv(os.path.join(root, file))
                if 'segment' not in df.columns:
                    pass  # Modify as necessary
                df['sample_name'] = df['noise_type']
                if len(df) > 1:
                    data_frames.append(df)
                    # print(file)
                    print(len(df))
    if not data_frames:
        raise ValueError("No valid CSV files found.")
    return pd.concat(data_frames, ignore_index=True)

def setup_plot(style="whitegrid", context="talk", figsize=(12, 8)):
    sns.set_style(style)
    sns.set_context(context)
    return plt.figure(figsize=figsize)

def save_and_close_plot(output_dir, filename, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=dpi, transparent=True)
    plt.close()

def create_boxplot(data, x, y, order, title, xlabel, ylabel, palette=None, rotation=90):
    if palette is None:
        palette = {denoiser: ("#3086ad" if denoiser == "reference" else "#e9f9ff")
                for denoiser in data["denoiser"]}
    sns.boxplot(data=data, x=x, y=y, order=order, showmeans=True, palette=palette, meanprops  = MEAN_PROPS)
    sns.despine(left=True, bottom=True, top=True, right=True)
    plt.title(title, fontsize=FONT_UNIT*1.25)
    plt.xlabel(xlabel, fontsize=FONT_UNIT)
    plt.ylabel(ylabel, fontsize=FONT_UNIT)
    plt.xticks(rotation=rotation, ha="right", fontsize=FONT_UNIT*0.8)

def generate_analysis(df, output_dir, inp_dir, group_cols, metric, analysis_name, ylabel, lower_better=True):
    df['cleaned_noise'] = df['noise_type'].apply(clean_noise_type)
    if metric == "denoiser_rank":
        df[metric] = df.groupby(group_cols)["wer_whisperx"].rank(method='average', ascending=lower_better)
    elif metric == "avg_wer":
        df = df.groupby(group_cols + ['denoiser'], as_index=False)['wer_whisperx'].mean().rename(columns={'wer_whisperx': metric})
    elif metric == "avg_cer":
        df = df.groupby(group_cols + ['denoiser'], as_index=False)['cer_whisperx'].mean().rename(columns={'cer_whisperx': metric})
    elif metric == "avg_cer_large":
        df['cer_whisperx'] = np.where(
            df['denoiser'].str.startswith('mel_band_roformer_vocals'),
            df['cer_whisperx'] - 0,
            df['cer_whisperx']
        )
        df = df.groupby(group_cols + ['denoiser'], as_index=False)['cer_whisperx'].mean().rename(columns={'cer_whisperx': metric})
        df = df.loc[df['cleaned_noise'] == 'cvr_noise_large']



    order_metric = df.groupby("denoiser")[metric].mean().sort_values(ascending=lower_better).index

    noises = df["cleaned_noise"].dropna().unique()
    setup_plot(figsize=(FIG_WIDTH, FIG_HEIGHT * len(noises)))
    fig, axes = plt.subplots(len(noises), 1, figsize=(FIG_WIDTH, FIG_HEIGHT * len(noises)))
    axes = np.atleast_1d(axes)

    for ax, noise in zip(axes, noises):
        palette = {denoiser: ("#3086ad" if denoiser in ["reference", "mel_band_roformer_vocals_fv4_gabox"] else "#e9f9ff")
                for denoiser in df["denoiser"]}
        sns.boxplot(
            data=df[df["cleaned_noise"] == noise],
            x="denoiser", y=metric, order=order_metric,
            palette=palette, showmeans=True, ax=ax, boxprops = None,
        meanprops  = MEAN_PROPS)
        sns.despine(ax=ax, left=True, bottom=True, top=True, right=True)
        ax.set_title(f"{analysis_name} for Noise: {noise} (n={df['denoiser'].value_counts().values.tolist()[0]})", fontsize=14)
        ax.set_xlabel("Denoiser", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=8)

    prefix = inp_dir.rstrip('/').split('/')[-1]
    save_and_close_plot(output_dir, f"{prefix}_{analysis_name.lower().replace(' ', '_')}_per_noise.{FILE_TYPE}")

    setup_plot()
    create_boxplot(
        df, "denoiser", metric, order_metric,
        f"Global {analysis_name}", "Denoiser", ylabel
    )
    save_and_close_plot(output_dir, f"{prefix}_global_{analysis_name.lower().replace(' ', '_')}.{FILE_TYPE}")

def preliminary_analysis(df, output_dir, inp_dir):
    sorted_denoisers = df.groupby("denoiser")["wer_whisperx"].mean().sort_values().index
    setup_plot(figsize=(FIG_WIDTH, FIG_HEIGHT))
    create_boxplot(
        df, "denoiser", "wer_whisperx", sorted_denoisers,
        "WER (WhisperX) for Different Denoisers (Sorted by Mean)",
        "Denoiser", "WER (WhisperX)"
    )
    prefix = inp_dir.rstrip('/').split('/')[-1]
    save_and_close_plot(output_dir, f"{prefix}_wer_whisperx_boxplot_sorted.{FILE_TYPE}", dpi=200)

def ranking_analysis(df, output_dir, inp_dir):
    generate_analysis(
        df, output_dir, inp_dir,
        ["segment", "cleaned_noise"],
        "denoiser_rank",
        "Denoiser Ranking Distribution",
        "Ranking Score (Lower is Better)"
    )

def average_wer_analysis(df, output_dir, inp_dir):
    generate_analysis(
        df, output_dir, inp_dir,
        ["segment", "cleaned_noise"],
        "avg_wer",
        "Denoiser Average WER",
        "Average WER (Lower is Better)"
    )

def average_cer_analysis(df, output_dir, inp_dir):
    generate_analysis(
        df, output_dir, inp_dir,
        ["segment", "cleaned_noise"],
        "avg_cer",
        "Denoiser Average CER",
        "Average CER (Lower is Better)"
    )

def graph_large_noise_only(df, output_dir, inp_dir):
    generate_analysis(
        df, output_dir, inp_dir,
        ["segment", "cleaned_noise"],
        "avg_cer_large",
        "Denoiser Average CER",
        "Average CER (Lower is Better)"
    )

def average_boxplot_per_noise(df, output_dir, inp_dir):
    # df['cleaned_noise'] = df['noise_type'].apply(clean_noise_type)
    # df_mean = df.groupby(["cleaned_noise", "denoiser"], as_index=False)["cer_whisperx"].mean().rename(columns={"cer_whisperx": "avg_cer"})
    df = df[df.noise_type != 'original']

    noise_order = df.groupby("noise_type")["cer"].mean().sort_values().index
    pretty_labels = [
        f"{nt.replace('_', ' ').title().replace('Vdr','VDR').replace('Cvr','CVR').replace('Large', 'High')}"
        for nt in noise_order
    ]

    setup_plot(figsize=(12, 6))
    palette = {noise_type: ("#3086ad" if noise_type == "cvr_noise_large" else "#e9f9ff")
                for noise_type in df["noise_type"]}
    sns.boxplot(
        data=df, x="noise_type", y="cer",
        order=noise_order, palette=palette, showmeans=True, meanprops=MEAN_PROPS
    )


    sns.despine(left=True, bottom=True, top=True, right=True)
    plt.title("Character Error Rate Distribution by Noise Condition", fontsize=16)
    plt.xlabel("Cleaned Noise Type", fontsize=14)
    plt.ylabel("Average cer", fontsize=14)
    plt.xticks(
        ticks=range(len(noise_order)),
        labels=pretty_labels,
        rotation=45,
        ha="right",
        fontsize=10
    )
    prefix = inp_dir.rstrip('/').split('/')[-1]
    save_and_close_plot(output_dir, f"{prefix}_average_cer_boxplot_per_noise_sorted.{FILE_TYPE}")

def main():
    parser = argparse.ArgumentParser(description="Analyze audio comparison results.")
    parser.add_argument("input_dir", type=str, help="Input directory with segment folders.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Output directory for results.")
    args = parser.parse_args()

    df = load_csv_files(args.input_dir)
    # average_boxplot_per_noise(df, args.output_dir, args.input_dir)
    # preliminary_analysis(df, args.output_dir, args.input_dir)
    # ranking_analysis(df, args.output_dir, args.input_dir)
    # average_wer_analysis(df, args.output_dir, args.input_dir)
    # average_cer_analysis(df, args.output_dir, args.input_dir)

    graph_large_noise_only(df, args.output_dir, args.input_dir)

if __name__ == "__main__":
    main()
