import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper")

# DATA_FILE = Path("./data/Table_COMPass - latency_performance_gpt5.csv")
DATA_FILE = Path("./data/Table_COMPass - latency_performance_gpt5_modified.csv")

TYPE_ORDER = [
    "ZS-DP",
    "CR-DP",
    "PR-DP",
    "ZS-CoT",
    "CR-CoT",
    "CS-CoT",
    "PR-CoT",
    "HB-CoT",
    # "PR-SF",
]

TYPE_DISPLAY = {
    "ZS-DP": "Zero-\nShot",
    "CR-DP": "Cross\nRandom",
    "PR-DP": "Personal\nRecent",
    "ZS-CoT": "Zero-\nShot",
    "CR-CoT": "Cross\nRandom",
    "CS-CoT": "Cross\nRetrieval",
    "PR-CoT": "Personal\nRecent",
    "HB-CoT": "Hybrid",
    # "PR-SF": "Personal\nRecent",
}

GROUP_COLORS = {
    "Direct Prediction": {"icl": "#FDE5C6", "reason": "#FDB462"},
    "Chain-of-Thought": {"icl": "#D3E5FF", "reason": "#7FB2FF"},
    "Self-Refinement": {"icl": "#E8DAFF", "reason": "#C39BFF"},
    "Other": {"icl": "#E0E0E0", "reason": "#B0B0B0"},
}

LINE_COLOR = "#E15759"


def compute_overall_f1(row: pd.Series) -> float:
    values = []
    for col in ("Depression F1", "Anxiety F1", "Stress F1"):
        if col in row and pd.notna(row[col]) and row[col] != "":
            values.append(float(row[col]))
    return float(np.mean(values)) if values else np.nan


def load_latency_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = ["ICL Selection", "Reasoning", "Reasoning_sd", "Depression F1", "Anxiety F1", "Stress F1"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["overall_f1"] = df.apply(compute_overall_f1, axis=1)
    return df


def determine_group(type_name: str) -> str:
    if not isinstance(type_name, str):
        return "Other"
    key = type_name.upper()
    if "SF" in key:
        return "Self-Refinement"
    if "COT" in key:
        return "Chain-of-Thought"
    if "DP" in key:
        return "Direct Prediction"
    return "Other"


def build_color_arrays(groups, metric: str):
    colors = []
    for g in groups:
        group_colors = GROUP_COLORS.get(g, GROUP_COLORS["Other"])
        colors.append(group_colors[metric])
    return colors


def legend_handles(groups_present, include_icl: bool, include_reason: bool, include_line: bool):
    handles = []
    labels = []
    for g in groups_present:
        if include_icl:
            handles.append(Patch(facecolor=GROUP_COLORS[g]["icl"], edgecolor="none"))
            labels.append(f"ICL Selection · {g}")
        if include_reason:
            handles.append(Patch(facecolor=GROUP_COLORS[g]["reason"], edgecolor="none"))
            labels.append(f"{g}")
    if include_line:
        handles.append(Line2D([0], [0], color=LINE_COLOR, marker="o", linewidth=2.0, markersize=6))
        labels.append("Overall F1")
    return handles, labels


def style_bar_container(container, type_order):
    if container is None:
        return
    for rect, type_name in zip(container.patches, type_order):
        rect.set_hatch("")
        rect.set_edgecolor("none")
        rect.set_linewidth(0)
        if rect.get_height() <= 0:
            continue
        key = type_name.upper()
        if key.startswith("ZS-"):
            rect.set_edgecolor("#6E6E6E")
            rect.set_linewidth(1.0)
        if key.startswith("PR-"):
            rect.set_hatch("////")
            rect.set_edgecolor("#FFFFFF")
            rect.set_linewidth(1.0)
        # if key.startswith("CR-"):
        #     rect.set_hatch("////")
        #     rect.set_edgecolor("#FFFFFF")
        #     rect.set_linewidth(1.0)


def plot_dataset(df: pd.DataFrame, dataset: str, remove_icl: bool, output: Path) -> None:
    subset = df[df["DATA"] == dataset].set_index("TYPE")
    if subset.empty:
        raise ValueError(f"No rows found for dataset '{dataset}'")

    icl_vals = [subset.at[t, "ICL Selection"] if t in subset.index else np.nan for t in TYPE_ORDER]
    reasoning_vals = [subset.at[t, "Reasoning"] if t in subset.index else np.nan for t in TYPE_ORDER]
    has_reasoning_sd = "Reasoning_sd" in subset.columns
    reasoning_sd_vals = [
        subset.at[t, "Reasoning_sd"] if has_reasoning_sd and t in subset.index else np.nan for t in TYPE_ORDER
    ]
    overall_f1 = [subset.at[t, "overall_f1"] if t in subset.index else np.nan for t in TYPE_ORDER]
    groups = [determine_group(t) for t in TYPE_ORDER]

    x = np.arange(len(TYPE_ORDER))
    fig, ax1 = plt.subplots(figsize=(7.2, 3.2))
    title_suffix = "Reasoning Only" if remove_icl else "Latency vs Overall F1"
    ax1.set_title(f"{dataset}: {title_suffix} (GPT-5)")

    reason_colors = build_color_arrays(groups, "reason")
    icl_colors = build_color_arrays(groups, "icl")

    reasoning_clean = np.nan_to_num(reasoning_vals, nan=0.0)
    reasoning_sd_clean = np.nan_to_num(reasoning_sd_vals, nan=0.0)
    reasoning_error_kw = {}
    if np.any(reasoning_sd_clean > 0):
        reasoning_error_kw = {
            "yerr": reasoning_sd_clean,
            "error_kw": {"capsize": 4, "elinewidth": 1, "ecolor": "#999999", "capthick": 1.2},
        }
    icl_clean = np.nan_to_num(icl_vals, nan=0.0)

    bar_width = 0.58
    if remove_icl:
        reasoning_container = ax1.bar(
            x,
            reasoning_clean,
            width=bar_width,
            color=reason_colors,
            label="Reasoning (sec)",
            zorder=2,
            **reasoning_error_kw,
        )
        latency_totals = reasoning_clean
        icl_container = None
    else:
        icl_container = ax1.bar(
            x,
            icl_clean,
            width=bar_width,
            color=icl_colors,
            label="ICL Selection (sec)",
            zorder=2,
        )
        reasoning_container = ax1.bar(
            x,
            reasoning_clean,
            bottom=icl_clean,
            width=bar_width,
            color=reason_colors,
            label="Reasoning (sec)",
            zorder=2,
            **reasoning_error_kw,
        )
        latency_totals = icl_clean + reasoning_clean

    style_bar_container(icl_container, TYPE_ORDER)
    style_bar_container(reasoning_container, TYPE_ORDER)

    ax1.set_ylabel("Latency (sec)")
    latency_ylim = (0, 70)
    ax1.set_ylim(latency_ylim)
    ax1.set_axisbelow(True)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    ax2 = ax1.twinx()
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

    valid_f1 = [v for v in overall_f1 if not np.isnan(v)]
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Overall F1")
    ax2.grid(False)

    ax1.set_xticks(x)
    xtick_labels = [TYPE_DISPLAY.get(t, t) for t in TYPE_ORDER]
    ax1.set_xticklabels(xtick_labels, rotation=0, ha="center")

    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)

    groups_present = []
    for g, r_val, i_val in zip(groups, reasoning_vals, icl_vals):
        r_valid = isinstance(r_val, (int, float)) and not np.isnan(r_val) and r_val > 0
        i_valid = isinstance(i_val, (int, float)) and not np.isnan(i_val) and i_val > 0
        if g not in groups_present and (r_valid or (not remove_icl and i_valid)):
            if g in GROUP_COLORS:
                groups_present.append(g)

    handles, labels = legend_handles(
        groups_present,
        include_icl=not remove_icl,
        include_reason=True,
        include_line=True,
    )
    legend_ncol = len(labels) if remove_icl else min(len(labels), 4)
    ax1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2 if remove_icl else -0.3),
        ncol=legend_ncol,
        frameon=False,
    )

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {output.resolve()}")


def main() -> None:
    df = load_latency_frame(DATA_FILE)
    for dataset in ["CES", "GLOBEM", "Mental IoT"]:
        stacked_output = Path(f"./plots/gpt5_latency_multi_f1_grouped_{dataset.lower().replace(' ', '')}.png")
        reasoning_only_output = Path(f"./plots/gpt5_latency_reasoning_only_grouped_{dataset.lower().replace(' ', '')}.png")
        # plot_dataset(df, dataset, remove_icl=False, output=stacked_output)
        plot_dataset(df, dataset, remove_icl=True, output=reasoning_only_output)


if __name__ == "__main__":
    main()
