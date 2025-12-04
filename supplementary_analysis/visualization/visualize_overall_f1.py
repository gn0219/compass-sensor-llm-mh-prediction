import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper")

SOURCE_FILES = [
    ("./data/Table_COMPass - CES Final Result.csv", "CES"),
    ("./data/Table_COMPass - GLOBEM Final Result.csv", "GLOBEM"),
    ("./data/Table_COMPass - Mental IoT Final Result.csv", "Mental IoT"),
]

RAW_MODEL_ORDER = [
    "GPT-5",
    "CLAUDE-4.5",
    "GEMINI-2.5-PRO",
    "gpt-oss-20b",
    "LLAMA-3.1-8B",
    "MISTRAL-7b",
]

MODEL_DISPLAY = {
    "GPT-5": "GPT-5",
    "CLAUDE-4.5": "CLAUDE 4.5 Sonnet",
    "GEMINI-2.5-PRO": "Gemini 2.5 Pro",
    "gpt-oss-20b": "gpt-oss-20b",
    "LLAMA-3.1-8B": "Llama3.1-8B",
    "MISTRAL-7b": "Mistral-7B",
    # Resiliency against alternative casing
    "MISTRAL-7B": "Mistral-7B",
    "LLAMA-3.1-8b": "Llama3.1-8B",
    "CLAUDE-3.5": "CLAUDE",
    "CLAUDE3.5": "CLAUDE",
}

DATASET_ORDER = ["CES", "GLOBEM", "Mental IoT"]
DATASET_COLORS = {
    "CES": "#E74C3C",        # red
    "GLOBEM": "#3498DB",     # blue
    "Mental IoT": "#F1C40F", # gold
}
DATASET_MARKERS = {
    "CES": "^",
    "GLOBEM": "D",
    "Mental IoT": "s",
}


def compute_overall_f1():
    records = []
    for csv_path, dataset_label in SOURCE_FILES:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"Model": "model"})
        if "model" not in df.columns:
            continue

        f1_columns = []
        for label in ["Macro F1 (Depression)", "Macro F1 (Anxiety)"]:
            if label in df.columns:
                f1_columns.append(label)
        include_stress = dataset_label != "GLOBEM" and "Macro F1 (Stress)" in df.columns
        if include_stress:
            f1_columns.append("Macro F1 (Stress)")

        for _, row in df.iterrows():
            model = row.get("model", None)
            if not isinstance(model, str) or not model.strip():
                continue

            f1_values = []
            for col in f1_columns:
                value = row.get(col, np.nan)
                if pd.notna(value):
                    f1_values.append(value)

            if not f1_values:
                continue

            overall = float(np.mean(f1_values))
            records.append(
                {
                    "dataset": dataset_label,
                    "model_raw": model.strip(),
                    "overall_f1": overall,
                }
            )

    metrics = (
        pd.DataFrame(records)
        .groupby(["dataset", "model_raw"], as_index=False)["overall_f1"]
        .mean()
    )
    metrics["model_display"] = metrics["model_raw"].map(
        lambda m: MODEL_DISPLAY.get(m, m)
    )
    return metrics


def main():
    metrics = compute_overall_f1()

    x_labels = [
        MODEL_DISPLAY.get(raw, raw) for raw in RAW_MODEL_ORDER
    ]
    x = np.arange(len(RAW_MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.grid(True, axis="y", linestyle="-", linewidth=0.7, alpha=0.35, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for dataset in DATASET_ORDER:
        subset = metrics[metrics["dataset"] == dataset]
        if subset.empty:
            continue
        y_values = []
        for model in RAW_MODEL_ORDER:
            match = subset[subset["model_raw"] == model]
            y_values.append(match["overall_f1"].iloc[0] if not match.empty else np.nan)

        ax.plot(
            x,
            y_values,
            label=dataset,
            color=DATASET_COLORS.get(dataset),
            marker=DATASET_MARKERS.get(dataset, "o"),
            linestyle="-",
            linewidth=2.0,
            markersize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.set_xlim(-0.5, len(x) - 0.5)

    y_min = np.nanmin(metrics["overall_f1"])
    y_max = np.nanmax(metrics["overall_f1"])
    margin = max((y_max - y_min) * 0.1, 0.02)
    ax.set_ylim(0, 0.75)

    ax.set_ylabel("Overall Macro F1")
    ax.set_title("Overall Macro F1 by Model (Averaged across ICL · Reasoning)")

    ax.legend(
        loc="upper center",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )

    plt.tight_layout()
    output_path = Path("./plots/overall_f1_by_model.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
