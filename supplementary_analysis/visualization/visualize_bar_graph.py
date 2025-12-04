import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")

def plot_single_bar(
    xlabels,
    values,
    colors=None,
    title="(c) Custom bar graph",
    ylabel="Value",
    xlabel=None,
    figsize=(8, 6.0),
    bar_width=0.6,
    yticks=None,
    ylim=None,
    font_axes=11,
    font_title=12,
    font_tick=10,
    legend_label=None,
    out_path="plots/figure_c_single_bar.png",
):
    """
    Draw a simple bar graph with same style as previous plots.

    Args:
        xlabels (list[str]): labels for x-axis
        values (list[float]): bar heights
        colors (list[str] or single str): custom color(s)
        title (str): plot title
        ylabel (str): y-axis label
        xlabel (str): x-axis label (optional)
        figsize (tuple): figure size
        bar_width (float): width of bars
        yticks (list[float]): optional y ticks
        ylim (tuple): optional y-axis limits
        legend_label (str): label for legend (optional)
        out_path (str): save path
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 기본 스타일
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # 데이터
    x = np.arange(len(xlabels))
    ax.bar(
        x, values, width=bar_width,
        color=colors, label=legend_label
    )

    # 축 설정
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=font_axes)
    ax.set_ylabel(ylabel, fontsize=font_axes)
    ax.set_title(title, fontsize=font_title)
    ax.tick_params(axis="both", labelsize=font_tick)

    # 옵션 처리
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # 범례
    if legend_label:
        ax.legend(
            frameon=True,
            fontsize=font_tick,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=1
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    # plt.show()


xlabels = ["ZS-DP", "CR-DP", "PR-DP", "ZS-CoT", "CR-CoT", "CS-CoT", "PR-CoT", "HB-CoT"]
values = [14.2, 30.4, 28.9, 29.5, 31.0, 30.1, 29.0, 28.4]
colors = sns.color_palette("Set2", n_colors=len(xlabels))  # or ["#5DADE2", "#EC7063", ...]

plot_single_bar(
    xlabels=xlabels,
    values=values,
    colors=colors,
    title="Performance by LLM Model",
    ylabel="Overall F1 Score",
    xlabel="Model",
    ylim=(0, 40),
    # legend_label="",
)

