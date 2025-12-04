import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper")

DATA_FILE = Path("./data/Table_COMPass - latency_performance_gpt5.csv")
OUTPUT_PATH = Path("./plots/latency_performance_gpt5.png")

TYPE_ORDER = [
    "ZS-DP",
    "CR-DP",
    "PR-DP",
    "ZS-CoT",
    "CR-CoT",
    "CS-CoT",
    "PR-CoT",
    "HB-CoT",
    "PR-SF",
]

BAR_COLORS = {
    "ICL Selection": "#FDB462",
    "Reasoning": "#80B1D3",
}

LINE_COLOR = "#4E79A7"


def compute_overall_f1(row: pd.Series) -> float:
    values = []
    for col in ("Depression F1", "Anxiety F1", "Stress F1"):
        if col in row and pd.notna(row[col]) and row[col] != "":
            values.append(float(row[col]))
    return float(np.mean(values)) if values else np.nan


def load_latency_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = ["ICL Selection", "Reasoning", "Depression F1", "Anxiety F1", "Stress F1"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["overall_f1"] = df.apply(compute_overall_f1, axis=1)
    return df


def prepare_series(df: pd.DataFrame, dataset: str) -> dict:
    subset = df[df["DATA"] == dataset].set_index("TYPE")
    series = {}
    for col in ["ICL Selection", "Reasoning", "overall_f1"]:
        series[col] = [subset.at[t, col] if t in subset.index else np.nan for t in TYPE_ORDER]
    return series


def plot_latency_performance(df: pd.DataFrame) -> None:
    datasets = ["CES", "GLOBEM", "Mental IoT"]
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 8.5), sharex=True)
    if n_rows == 1:
        axes = [axes]

    latency_totals = (df["ICL Selection"].fillna(0) + df["Reasoning"].fillna(0)).values
    latency_ylim = (0, max(latency_totals.max(), 1.0) * 1.15)

    f1_values = df["overall_f1"].dropna()
    if f1_values.empty:
        f1_ylim = (0.0, 1.0)
    else:
        f1_min, f1_max = f1_values.min(), f1_values.max()
        margin = max((f1_max - f1_min) * 0.15, 0.02)
        f1_ylim = (max(f1_min - margin, 0), min(f1_max + margin, 1.0))

    x = np.arange(len(TYPE_ORDER))

    for ax, dataset in zip(axes, datasets):
        series = prepare_series(df, dataset)

        icl = np.nan_to_num(series["ICL Selection"], nan=0.0)
        reasoning = np.nan_to_num(series["Reasoning"], nan=0.0)
        overall_f1 = series["overall_f1"]

        ax.bar(
            x,
            icl,
            color=BAR_COLORS["ICL Selection"],
            width=0.58,
            label="ICL Selection (sec)",
            zorder=2,
        )
        ax.bar(
            x,
            reasoning,
            bottom=icl,
            color=BAR_COLORS["Reasoning"],
            width=0.58,
            label="Reasoning (sec)",
            zorder=2,
        )
        ax.set_ylabel("Latency (sec)")
        ax.set_ylim(latency_ylim)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

        ax2 = ax.twinx()
        ax2.plot(
            x,
            overall_f1,
            color=LINE_COLOR,
            marker="o",
            linewidth=2.0,
            markersize=6,
            label="Overall F1",
            zorder=3,
        )
        ax2.set_ylim(f1_ylim)
        ax2.set_ylabel("Overall F1")
        ax2.grid(False)

        ax.set_title(dataset)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax.set_xticks(x)
        if dataset != datasets[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(TYPE_ORDER, rotation=0, ha="center")

    handles = [
        Patch(facecolor=BAR_COLORS["ICL Selection"], label="ICL Selection (sec)"),
        Patch(facecolor=BAR_COLORS["Reasoning"], label="Reasoning (sec)"),
        Line2D([0], [0], color=LINE_COLOR, marker="o", linewidth=2.0, markersize=6, label="Overall F1"),
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )

    fig.supxlabel("Strategy Type")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"[INFO] Saved to: {OUTPUT_PATH.resolve()}")


def main() -> None:
    df = load_latency_frame(DATA_FILE)
    plot_latency_performance(df)


if __name__ == "__main__":
    main()
