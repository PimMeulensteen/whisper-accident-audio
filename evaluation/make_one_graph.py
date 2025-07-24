#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def highlight_reference(ax, sorted_denoisers, reference):
    """
    Draws a vertical red line exactly at the x-position of the reference denoiser.
    """
    if reference in sorted_denoisers:
        ref_idx = sorted_denoisers.index(reference)
        # Seaborn places categories at integer locations: 0, 1, 2, ...
        # So a vertical line at x=ref_idx will line up exactly with the reference category.
        ax.axvline(x=ref_idx, color='red', linestyle='--', linewidth=2, alpha=0.8)
    else:
        print(f"Warning: Reference denoiser '{reference}' not found in the data.")

def plot_cer_wer(df, reference, output_path):
    """
    Generates a combination of boxplot + stripplot for the CER and WER metrics 
    (cer_whisperx and wer_whisperx) across different denoisers.

    - Denoisers are sorted by the average of mean(CER) and mean(WER).
    - CER is blue, WER is orange.
    - A vertical red line is drawn exactly at the reference denoiser on the x-axis.
    """

    # Ensure required columns are present
    required_cols = ["denoiser", "cer_whisperx", "wer_whisperx"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: {col}")

    # Compute sorting based on average of mean CER and WER
    stats = df.groupby("denoiser").agg({
        "cer_whisperx": "mean",
        "wer_whisperx": "mean"
    })
    stats["avg_cer_wer"] = (stats["cer_whisperx"] + stats["wer_whisperx"]) / 2
    sorted_denoisers = stats.sort_values("avg_cer_wer").index.tolist()

    # Reshape to long format for plotting with a 'metric' hue
    df_long = pd.melt(
        df,
        id_vars=["denoiser"],
        value_vars=["cer_whisperx", "wer_whisperx"],
        var_name="metric",
        value_name="value"
    )
    df_long["denoiser"] = pd.Categorical(
        df_long["denoiser"], categories=sorted_denoisers, ordered=True
    )

    # Seaborn style and context
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Distinct colors for CER vs. WER
    metric_palette = {
        "cer_whisperx": "#1f77b4",  # Blue
        "wer_whisperx": "#ff7f0e"   # Orange
    }
    hue_order = ["cer_whisperx", "wer_whisperx"]

    plt.figure(figsize=(16, 10))

    # Boxplot without outliers (they will be shown in the stripplot)
    ax = sns.boxplot(
        data=df_long,
        x="denoiser",
        y="value",
        hue="metric",
        order=sorted_denoisers,
        hue_order=hue_order,
        palette=metric_palette,
        showmeans=False,
        fliersize=0
    )

    # Overlay stripplot to show individual points colored by hue
    sns.stripplot(
        data=df_long,
        x="denoiser",
        y="value",
        hue="metric",
        order=sorted_denoisers,
        hue_order=hue_order,
        palette=metric_palette,
        dodge=True,
        marker="o",
        alpha=0.7,
        ax=ax
    )

    # Remove duplicate legend entries (boxplot + stripplot)
    handles, labels = ax.get_legend_handles_labels()
    # We only want two unique legend items (CER and WER)
    unique_handles = [handles[0], handles[1]]
    unique_labels  = ["CER (WhisperX)", "WER (WhisperX)"]
    ax.legend(unique_handles, unique_labels, title="Metric", loc="upper left")

    sns.despine(left=True, bottom=True, top=True, right=True)

    # Set axis labels and title
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Denoiser", fontsize=14)
    ax.set_ylabel("Error Rate", fontsize=14)
    ax.set_title(
        "CER and WER (WhisperX) for Different Denoisers\n(Sorted by Mean CER+WER)",
        fontsize=16
    )

    # Draw a vertical line at the reference denoiser's x-position
    highlight_reference(ax, sorted_denoisers, reference)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot CER and WER (WhisperX) for different denoisers from a single CSV file."
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save the plot."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="reference",
        help="Name of the reference denoiser to highlight."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = (
        os.path.splitext(os.path.basename(args.csv_path))[0]
        + "_cer_wer_boxplot.png"
    )
    output_path = os.path.join(args.output_dir, output_filename)

    df = pd.read_csv(args.csv_path)
    plot_cer_wer(df, args.reference, output_path)

if __name__ == "__main__":
    main()
