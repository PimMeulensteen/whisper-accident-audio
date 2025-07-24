import os
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Configure matplotlib fonts
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

n_csv = 0

def load_csv_files(input_dir: str) -> pd.DataFrame:
    """
    Loads and combines CSV files from the specified directory structure.
    Each CSV file must have at least 10 rows and an average 'wer_whisperx' value <= 0.98.
    A constant 'noise_type' and a 'sample_name' are set, and a segment identifier is assigned 
    based on the file name.
    """
    input_path = Path(input_dir)
    data_frames = []

    for file_path in input_path.rglob("comparison*.csv"):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Ensure required columns exist
        if "denoiser" not in df.columns or "wer_whisperx" not in df.columns:
            print(f"Skipping {file_path}: required columns missing.")
            continue

        df = df.drop_duplicates(subset="denoiser", keep="first")

        # Set constant noise type and sample name
        df["noise_type"] = "noise"
        df["sample_name"] = df["noise_type"]

        # Assign segment identifier from the file name (stem)
        df["segment"] = file_path.stem

        # Accept only files with sufficient data quality
        if len(df) >= 10 and df["wer_whisperx"].mean() <= 0.98:
            data_frames.append(df)

    if not data_frames:
        raise ValueError("No valid CSV files found.")

    return pd.concat(data_frames, ignore_index=True)


def keep_relevant_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only rows for denoisers that are present in all segments.
    This ensures that comparisons are made on complete data.
    """
    denoiser_counts = df["denoiser"].value_counts()
    max_count = denoiser_counts.max()
    complete_denoisers = denoiser_counts[denoiser_counts == max_count].index
    return df[df["denoiser"].isin(complete_denoisers)]


def configure_plot():
    """Sets common style settings for all plots."""
    sns.set_style("whitegrid")
    sns.set_context("talk")


def save_plot(plot_path: Path):
    """Saves the current plot to a file and closes the figure."""
    plt.savefig(plot_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_boxplot(df: pd.DataFrame, output_dir: str, total_count: int):
    """
    Creates boxplots for WER and CER metrics.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = [
        ("wer_whisperx", "WER", "Word Error Rate (WhisperX large-v3 vs. transcript)", "atco_whisperx_boxplot_wer", (0.1, 0.5)),
        ("cer_whisperx", "CER", "Character Error Rate (WhisperX large-v3 vs. transcript)", "atco_whisperx_boxplot_cer", None)
    ]
    configure_plot()

    for metric, metric_label, y_label, filename, ylim in metrics:
        sorted_denoisers = df.groupby("denoiser")[metric].mean().sort_values().index
        plt.figure(figsize=(15, 10))
        palette_dict = {denoiser: ("#3086ad" if denoiser == "reference" else "#e9f9ff")
                        for denoiser in df["denoiser"]}
        sns.boxplot(
            data=df,
            x="denoiser",
            y=metric,
            order=sorted_denoisers,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
                "alpha": 0.6,
            },
            palette=palette_dict,
            flierprops={"marker": "o", "markersize": 3},
        )

        sns.despine(left=True, bottom=True, top=True, right=True)
        if ylim:
            plt.ylim(*ylim)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Denoiser", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(
            f"{metric_label} for Different Denoisers on {total_count} samples from the ATCO2 dataset.",
            fontsize=14,
        )

        # Add baseline reference line if "reference" denoiser exists
        baseline_df = df[df["denoiser"] == "reference"]
        if not baseline_df.empty:
            baseline_mean = baseline_df[metric].mean()
            plt.axhline(
                y=baseline_mean,
                color="#f8b032",
                linestyle="-",
                linewidth=1,
                label="Mean no denoiser",
            )
            plt.legend(frameon=False, bbox_to_anchor=(0.26, 0.95))

        plot_file = Path(output_dir) / f"{filename}.pdf"
        save_plot(plot_file)


def plot_ranking_boxplot(df: pd.DataFrame, output_dir: str, total_count: int):
    """
    For each segment, ranks the denoisers based on the selected metric (WER or CER) 
    and assigns ranking points (best = 1, second = 2, etc.). A boxplot of these ranking 
    points is then created for each denoiser.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = [
        ("wer_whisperx", "WER", "Ranking Points (Lower is Better)", "atco_ranking_boxplot_wer"),
        ("cer_whisperx", "CER", "Ranking Points (Lower is Better)", "atco_ranking_boxplot_cer"),
    ]
    configure_plot()
    df_ranked = df.copy()

    for metric, metric_label, y_label, filename in metrics:
        rank_col = f"rank_{metric}"
        df_ranked[rank_col] = df_ranked.groupby("segment")[metric].rank(method="min", ascending=True)
        sorted_denoisers = df_ranked.groupby("denoiser")[rank_col].mean().sort_values().index
        palette_dict = {denoiser: ("#3086ad" if denoiser == "reference" else "#e9f9ff")
                        for denoiser in df["denoiser"]}
        plt.figure(figsize=(15, 10))
        sns.boxplot(
            data=df_ranked,
            x="denoiser",
            y=rank_col,
            order=sorted_denoisers,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
                "alpha": 0.6,
            },
            palette=palette_dict,
            flierprops={"marker": "o", "markersize": 3},
        )

        baseline_df = df_ranked[df_ranked["denoiser"] == "reference"]
        if not baseline_df.empty:
            baseline_mean_rank = baseline_df[rank_col].mean()
            plt.axhline(
                y=baseline_mean_rank,
                color="#f8b032",
                linestyle="-",
                linewidth=1,
                label="Mean no denoiser",
            )
            plt.legend(frameon=False, bbox_to_anchor=(0.12, 0.925))

        sns.despine(left=True, bottom=True, top=True, right=True)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Denoiser", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(
            f"Denoiser Ranking Based on Average {metric_label} per Segment on {total_count} samples from the ATCO2 dataset.",
            fontsize=14,
        )

        plot_file = Path(output_dir) / f"{filename}.pdf"
        save_plot(plot_file)

def plot_sum_metric(df: pd.DataFrame, output_dir: str, total_count: int):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = [
        ("wer_whisperx", "WER", "Sum of WER across all samples", "atco_sum_across_samples_wer"),
        ("cer_whisperx", "CER", "Sum of CER across all samples", "atco_sum_across_samples_cer"),
    ]
    configure_plot()

    for metric, metric_label, y_label, filename in metrics:
        sum_metric = df.groupby("denoiser")[metric].sum().sort_values()
        plt.figure(figsize=(15, 10))
        palette_dict = {denoiser: ("#3086ad" if denoiser == "reference" else "#e9f9ff")
                        for denoiser in df["denoiser"]}
        sns.barplot(
            x=sum_metric.index,
            y=sum_metric.values,
            palette=palette_dict,
        )
        sns.despine(left=True, bottom=True, top=True, right=True)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Denoiser", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(
            f"Total {metric_label} (WhisperX) across all samples ({total_count} segments) for each Denoiser",
            fontsize=14,
        )

        plot_file = Path(output_dir) / f"{filename}.pdf"
        save_plot(plot_file)


def plot_weighted_boxplot(df: pd.DataFrame, output_dir: str, total_count: int):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = [
        ("wer_whisperx", "WER", "Weighted Score (Lower is Better)", "atco_weighted_denoiser_ranking_boxplot_wer"),
        ("cer_whisperx", "CER", "Weighted Score (Lower is Better)", "atco_weighted_denoiser_ranking_boxplot_cer"),
    ]
    configure_plot()
    df_weighted = df.copy()

    for metric, metric_label, y_label, filename in metrics:
        weighted_col = f"weighted_score_{metric}"

        def compute_weighted(x):
            if x.max() == x.min():
                return pd.Series(1, index=x.index)
            return 1 + (x - x.min()) / (x.max() - x.min()) * (len(x) - 1)

        df_weighted[weighted_col] = df_weighted.groupby("segment")[metric].transform(compute_weighted)
        sorted_denoisers = df_weighted.groupby("denoiser")[weighted_col].mean().sort_values().index
        palette_dict = {denoiser: ("#3086ad" if denoiser == "reference" else "#e9f9ff")
                        for denoiser in df["denoiser"]}
        plt.figure(figsize=(15, 10))
        sns.boxplot(
            data=df_weighted,
            x="denoiser",
            y=weighted_col,
            order=sorted_denoisers,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
                "alpha": 0.6,
            },
            palette=palette_dict,
            flierprops={"marker": "o", "markersize": 3},
        )

        baseline_df = df_weighted[df_weighted["denoiser"] == "reference"]
        if not baseline_df.empty:
            baseline_mean_score = baseline_df[weighted_col].mean()
            plt.axhline(
                y=baseline_mean_score,
                color="#f8b032",
                linestyle="-",
                linewidth=1,
                label="Mean no denoiser",
            )
            plt.legend(frameon=False, bbox_to_anchor=(0.26, 0.95))

        sns.despine(left=True, bottom=True, top=True, right=True)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Denoiser", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(
            f"Weighted Denoiser Ranking Based on Average {metric_label} per Segment ({total_count} segments)",
            fontsize=14,
        )

        plot_file = Path(output_dir) / f"{filename}.pdf"
        save_plot(plot_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze audio comparison results.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing CSV files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Output directory for plots and results.",
    )
    args = parser.parse_args()

    df = load_csv_files(args.input_dir)
    df = keep_relevant_data(df)
    total_count = df["segment"].nunique()
    print(f"Read {total_count} CSV files.")

    plot_boxplot(df, args.output_dir, total_count)
    plot_ranking_boxplot(df, args.output_dir, total_count)
    plot_sum_metric(df, args.output_dir, total_count)
    plot_weighted_boxplot(df, args.output_dir, total_count)


if __name__ == "__main__":
    main()
