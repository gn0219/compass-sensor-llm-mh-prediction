import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")

def plot_single_line(
    xlabels,
    series_dict,
    colors=None,
    markers=None,
    title="(a) Custom line graph",
    ylabel="Metric",
    xlabel=None,
    figsize=(8, 6),
    linewidth=1.6,
    markersize=5.5,
    yticks=None,
    ylim=None,
    font_axes=11,
    font_title=12,
    font_tick=10,
    legend_loc="lower center",
    legend_frame=True,
    legend_ncol=3,
    out_path="plots/figure_custom_line.png",
):
    """
    Draw a clean multi-series line plot (same format as figure_a).

    Args:
        xlabels (list[str]): x-axis labels
        series_dict (dict): {"Label": [values...], ...}
        colors (dict): {"Label": color_code}
        markers (dict): {"Label": marker_symbol}
        title (str): figure title
        ylabel (str): y-axis label
        xlabel (str): optional x-axis label
        figsize (tuple): figure size
        linewidth (float): line width
        markersize (float): marker size
        yticks (list[float]): custom y ticks
        ylim (tuple): y-axis limits
        legend_loc (str): legend position
        legend_frame (bool): show frame
        legend_ncol (int): legend column count
        out_path (str): file path to save figure
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    x = np.arange(len(xlabels))

    # 기본 스타일
    ax.grid(True, axis="y", linestyle="-", linewidth=0.7, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # 선 플롯
    for label, values in series_dict.items():
        ax.plot(
            x,
            values,
            label=label,
            color=colors.get(label) if colors else None,
            marker=markers.get(label, "o") if markers else "o",
            linestyle="-",
            linewidth=linewidth,
            markersize=markersize,
        )

    # 축 설정
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=font_title)
    ax.tick_params(axis="both", labelsize=font_tick)
    ax.yaxis.label.set_size(font_axes)
    ax.xaxis.label.set_size(font_axes)

    # 범례
    ax.legend(
        loc=legend_loc,
        frameon=legend_frame,
        fontsize=font_tick,
        bbox_to_anchor=(0.5, -0.3),
        ncol=legend_ncol,
    )

    # 저장
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    # plt.show()


xlabels = ["GPT-5", "Claude 4.5 Sonnet", "Gemini 2.5 Pro", "gpt-oss-20b", "LLAMA-3", "CS-CoT", "PR-CoT", "HB-CoT"]

series_dict = {
    "CES": [0.53, 0.61, 0.63, 0.59, 0.61, 0.63, 0.65, 0.66],
    "GLOBEM": [0.49, 0.58, 0.60, 0.56, 0.59, 0.61, 0.62, 0.63],
    "Mental-IoT": [0.50, 0.59, 0.61, 0.57, 0.60, 0.62, 0.63, 0.64],
}

colors = {
    "CES": "#E74C3C",  # 빨강
    "GLOBEM": "#3498DB",     # 파랑
    "Mental-IoT": "#F1C40F",      # 노랑
}

markers = {
    "CES": "^",
    "GLOBEM": "D",
    "Mental-IoT": "s",
}

plot_single_line(
    xlabels=xlabels,
    series_dict=series_dict,
    colors=colors,
    markers=markers,
    title="(a) Performance metrics across strategies",
    ylabel="Metric",
    ylim=(0.4, 0.8),
    yticks=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    legend_loc="lower center",
    legend_frame=True,
)
