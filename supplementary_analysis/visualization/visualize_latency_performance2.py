# Re-run the two-figure plotting utilities (environment is fresh).
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")

# ------------------ Demo data (EDIT with your real numbers) ------------------
demo_perf_latency = {
    "ZS-DP":  {"Example Selection": 0.0,  "Reasoning": 14.2, "Depression F1": 0.53, "Anxiety F1": 0.49, "Stress F1": 0.50},
    "CR-DP":  {"Example Selection": 0.021,  "Reasoning": 30.4, "Depression F1": 0.61, "Anxiety F1": 0.58, "Stress F1": 0.59},
    "PR-DP":  {"Example Selection": 0.0065,  "Reasoning": 28.9, "Depression F1": 0.63, "Anxiety F1": 0.60, "Stress F1": 0.61},
    "ZS-CoT": {"Example Selection": 0.0,  "Reasoning": 29.5, "Depression F1": 0.59, "Anxiety F1": 0.56, "Stress F1": 0.57},
    "CR-CoT": {"Example Selection": 0.021,  "Reasoning": 31.0, "Depression F1": 0.61, "Anxiety F1": 0.59, "Stress F1": 0.60},
    "CS-CoT": {"Example Selection": 1.11,  "Reasoning": 30.1, "Depression F1": 0.63, "Anxiety F1": 0.61, "Stress F1": 0.62},
    "PR-CoT": {"Example Selection": 0.0065,  "Reasoning": 29.0, "Depression F1": 0.65, "Anxiety F1": 0.62, "Stress F1": 0.63},
    "HB-CoT": {"Example Selection": 0.021,  "Reasoning": 28.4, "Depression F1": 0.66, "Anxiety F1": 0.63, "Stress F1": 0.64},
    "PR-SF": {"Example Selection": 0.0065,  "Reasoning": 28.4, "Depression F1": 0.66, "Anxiety F1": 0.63, "Stress F1": 0.64},
} # CES

# demo_perf_latency = {
#     "ZS-DP":  {"Example Selection": 0.0,  "Reasoning": 14.2, "Depression F1": 0.53, "Anxiety F1": 0.49, "Stress F1": 0.50},
#     "CR-DP":  {"Example Selection": 0.04995,  "Reasoning": 30.4, "Depression F1": 0.61, "Anxiety F1": 0.58, "Stress F1": 0.59},
#     "PR-DP":  {"Example Selection": 0.04499,  "Reasoning": 28.9, "Depression F1": 0.63, "Anxiety F1": 0.60, "Stress F1": 0.61},
#     "ZS-CoT": {"Example Selection": 0.0,  "Reasoning": 29.5, "Depression F1": 0.59, "Anxiety F1": 0.56, "Stress F1": 0.57},
#     "CR-CoT": {"Example Selection": 0.04995,  "Reasoning": 31.0, "Depression F1": 0.61, "Anxiety F1": 0.59, "Stress F1": 0.60},
#     "CS-CoT": {"Example Selection": 0.9532,  "Reasoning": 30.1, "Depression F1": 0.63, "Anxiety F1": 0.61, "Stress F1": 0.62},
#     "PR-CoT": {"Example Selection": 0.04499,  "Reasoning": 29.0, "Depression F1": 0.65, "Anxiety F1": 0.62, "Stress F1": 0.63},
#     "HB-CoT": {"Example Selection": 0.065,  "Reasoning": 28.4, "Depression F1": 0.66, "Anxiety F1": 0.63, "Stress F1": 0.64},
#     "PR-SF": {"Example Selection": 0.04499,  "Reasoning": 28.4, "Depression F1": 0.66, "Anxiety F1": 0.63, "Stress F1": 0.64},
# } # GLOBEM

# demo_perf_latency = {
#     "ZS-DP":  {"Example Selection": 0.0,  "Reasoning": 14.2, "Depression F1": 0.53, "Anxiety F1": 0.49, "Stress F1": 0.50},
#     "CR-DP":  {"Example Selection": 0.022,  "Reasoning": 30.4, "Depression F1": 0.61, "Anxiety F1": 0.58, "Stress F1": 0.59},
#     "PR-DP":  {"Example Selection": 0.022,  "Reasoning": 28.9, "Depression F1": 0.63, "Anxiety F1": 0.60, "Stress F1": 0.61},
#     "ZS-CoT": {"Example Selection": 0.0,  "Reasoning": 29.5, "Depression F1": 0.59, "Anxiety F1": 0.56, "Stress F1": 0.57},
#     "CR-CoT": {"Example Selection": 0.022,  "Reasoning": 31.0, "Depression F1": 0.61, "Anxiety F1": 0.59, "Stress F1": 0.60},
#     "CS-CoT": {"Example Selection": 0.05,  "Reasoning": 30.1, "Depression F1": 0.63, "Anxiety F1": 0.61, "Stress F1": 0.62},
#     "PR-CoT": {"Example Selection": 0.022,  "Reasoning": 29.0, "Depression F1": 0.65, "Anxiety F1": 0.62, "Stress F1": 0.63},
#     "HB-CoT": {"Example Selection": 0.024,  "Reasoning": 28.4, "Depression F1": 0.66, "Anxiety F1": 0.63, "Stress F1": 0.64},
#     "PR-SF": {"Example Selection": 0.022,  "Reasoning": 28.4, "Depression F1": 0.66, "Anxiety F1": 0.63, "Stress F1": 0.64},
# } # Mental IoT

def plot_performance_only(
    data: dict,
    order=None,
    title="(a) Performance metrics across strategies",
    ylabel="Metric",
    figsize=(8, 6),
    markers=None,
    linewidth=1.6,
    markersize=5.5,
    yticks=None,
    ylim=None,
    font_axes=11,
    font_title=12,
    font_tick=10,
    legend_loc="lower center",
    legend_frame=True,
):
    keys = order if order is not None else list(data.keys())
    candidate = ["Accuracy", "Macro F1", "Depression F1", "Anxiety F1", "Stress F1"]
    present = [c for c in candidate if any(c in data[k] for k in keys)]
    series = {c: [data[k].get(c, np.nan) for k in keys] for c in present}
    if markers is None:
        markers = {}
    plt.figure(figsize=figsize)
    ax = plt.gca()
    x = np.arange(len(keys))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.7, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for label in present:
        ax.plot(x, series[label], marker=markers.get(label, "o"), linestyle="-",
                linewidth=linewidth, markersize=markersize, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=font_title)
    ax.tick_params(axis="both", labelsize=font_tick)
    ax.yaxis.label.set_size(font_axes)
    ax.xaxis.label.set_size(font_axes)
    ax.legend(loc=legend_loc, frameon=legend_frame, fontsize=font_tick, bbox_to_anchor=(0.5, -0.3), ncol=3)
    out = "plots/figure_a_performance_only.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {out}")

def plot_latency_only(
    data: dict,
    order=None,
    title="(b) Real-time processing latency by strategy",
    ylabel="Latency (sec)",
    figsize=(8, 6),
    bar_width=0.65,
    yticks=None,
    ylim=None,
    stacked=True,
    font_axes=11,
    font_title=12,
    font_tick=10,
):
    keys = order if order is not None else list(data.keys())
    x = np.arange(len(keys))
    sel = [data[k].get("Example Selection", 0.0) for k in keys]
    rea = [data[k].get("Reasoning", 0.0) for k in keys]
    plt.figure(figsize=figsize)
    ax = plt.gca()
    # ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if stacked:
        ax.bar(x, sel, width=bar_width, label="Example Selection")
        ax.bar(x, rea, width=bar_width, bottom=sel, label="Reasoning")
    else:
        total = np.array(sel) + np.array(rea)
        ax.bar(x, total, width=bar_width, label="Total latency")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=font_title)
    ax.tick_params(axis="both", labelsize=font_tick)
    ax.yaxis.label.set_size(font_axes)
    ax.xaxis.label.set_size(font_axes)
    ax.legend(frameon=True, fontsize=font_tick, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    out = "plots/figure_b_latency_only.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {out}")

# Example category order resembling the reference
order = ["ZS-DP", "CR-DP", "PR-DP", "ZS-CoT", "CR-CoT", "CS-CoT", "PR-CoT", "HB-CoT"]
remapped = {
    order[0]: demo_perf_latency["ZS-DP"],
    order[1]: demo_perf_latency["CR-DP"],
    order[2]: demo_perf_latency["PR-DP"],
    order[3]: demo_perf_latency["ZS-CoT"],
    order[4]: demo_perf_latency["CR-CoT"],
    order[5]: demo_perf_latency["CS-CoT"],
    order[6]: demo_perf_latency["PR-CoT"],
    order[7]: demo_perf_latency["HB-CoT"],
}

plot_performance_only(
    remapped,
    order=order,
    title="(a) Performance metrics across strategies",
    ylabel="Metric",
    figsize=(7.2, 3.0),
    markers={"Accuracy":"o", "Depression F1":"^", "Anxiety F1":"D", "Stress F1":"s"},
    yticks=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    ylim=(.40,0.80),
    font_axes=11, font_title=12, font_tick=10,
    legend_loc="lower center",
    legend_frame=True,
)

plot_latency_only(
    remapped,
    order=order,
    title="(b) Real-time processing latency by strategy",
    ylabel="Latency (sec)",
    figsize=(7.2, 3.0),
    bar_width=0.6,
    yticks=list(range(0, 41, 5)),
    ylim=(0, 40),
    stacked=True,
    font_axes=11, font_title=12, font_tick=10,
)
