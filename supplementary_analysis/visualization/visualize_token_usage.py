# Re-render "Completion Tokens by Model" plot
# Averaged per (model, ICL, Reasoning)
# ✅ legend 이름·순서 모두 사용자 지정 가능

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pathlib import Path
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")

# ---------- Load ----------
paths = [
    "./data/Table_COMPass - CES Final Result.csv",
    "./data/Table_COMPass - GLOBEM Final Result.csv",
    "./data/Table_COMPass - Mental IoT Final Result.csv",
]
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

df = df.rename(columns={
    "Dataset":"dataset",
    "Model":"model",
    "ICL Strategy":"icl",
    "Reasoning":"reasoning",
    "Shot":"shot"
})

chosen_col = "completion_tokens"
print(f"[INFO] Completion token source column: {chosen_col}")

# ---------- Normalize ----------
def norm(s):
    if pd.isna(s): return s
    return str(s).strip()

def norm_key(s):
    if pd.isna(s): return s
    return str(s).strip().lower().replace(" ", "")

df["model"] = df["model"].map(norm)
df["icl"] = df["icl"].map(norm)
df["reasoning"] = df["reasoning"].map(norm)
df["icl_key"] = df["icl"].map(norm_key)
df["reasoning_key"] = df["reasoning"].map(norm_key)

# ---------- Aggregate ----------
agg = (
    df.groupby(["model", "icl_key", "reasoning_key"], dropna=False)[chosen_col]
      .mean()
      .reset_index()
)

icl_label_map = (
    df.dropna(subset=["icl_key", "icl"])
      .drop_duplicates(subset=["icl_key"])
      .set_index("icl_key")["icl"]
      .to_dict()
)
reason_label_map = (
    df.dropna(subset=["reasoning_key", "reasoning"])
      .drop_duplicates(subset=["reasoning_key"])
      .set_index("reasoning_key")["reasoning"]
      .to_dict()
)

agg["icl_label"] = agg["icl_key"].map(lambda k: icl_label_map.get(k, k))
agg["reasoning_label"] = agg["reasoning_key"].map(lambda k: reason_label_map.get(k, k))
agg["combo_key"] = list(zip(agg["icl_key"], agg["reasoning_key"]))

# ---------- ✅ User-custom legend names ----------
combo_alias = {
    ("none","direct"):        "Zero-Shot · DP",
    ("none","cot"):           "Zero-Shot · CoT",
    ("crossrandom","cot"):    "Cross-Random · CoT",
    ("crossretrieval","cot"): "Cross-Retrieval · CoT",
    ("personalrecent","cot"): "Personal-Recent · CoT",
    ("hybridblend","cot"):    "Hybrid · CoT",
    # ("hybridblend","self-feedback"): "Hybrid · Self-Feedback"
}

# ---------- ✅ User-custom legend order ----------
# 지정된 순서로 legend·color·marker 정렬됨
combo_order = [
    ("none","direct"),
    ("none","cot"),
    ("crossrandom","cot"),
    ("crossretrieval","cot"),
    ("personalrecent","cot"),
    ("hybridblend","cot"),
    # ("hybridblend","self-feedback"),
]

agg["combo_label"] = agg.apply(
    lambda r: combo_alias.get((r["icl_key"], r["reasoning_key"]),
                              f"{r['icl_label']} · {r['reasoning_label']}"),
    axis=1
)

# ---------- Aliases ----------
model_alias = {
    "LLAMA-3.1-8B": "Llama-3.1-8b",
    "GEMINI-2.5-PRO": "Gemini 2.5 Pro",
    "CLAUDE-4.5": "Claude 4.5 Sonnet",
    "MISTRAL-7B": "Mistral 7B",
    "MISTRAL-7b": "Mistral 7B",
    "GPT-oss-20b": "gpt-oss-20b"
}
agg["model_display"] = agg["model"].map(lambda m: model_alias.get(m, m))

# ---------- Colors & markers ----------
colors_cycle = [
    "#00CD6C", "#A1B1BA", "#C58AF9", "#FFE51E", "#F538A0",
    "#00BFFF", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
]
markers_cycle = ["o","^","s","D","P","*","X","v","<",">"]

# ✅ robust: numpy unique() 쓰지 말고, 파이썬 리스트/세트로 처리 (순서 보존)
combo_keys_in_data = list(dict.fromkeys(agg["combo_key"].tolist()))  # 데이터에 실제 존재하는 (icl, reasoning) 튜플들
ordered_combos = [k for k in combo_order if k in combo_keys_in_data]
unordered_combos = [k for k in combo_keys_in_data if k not in set(ordered_combos)]
final_combos = ordered_combos + unordered_combos

color_map = {k: colors_cycle[i % len(colors_cycle)] for i, k in enumerate(final_combos)}
marker_map = {k: markers_cycle[i % len(markers_cycle)] for i, k in enumerate(final_combos)}

# ---------- Plot ----------
y_tick_mode = "0.5k"
ylim = (0, 4000)
preferred_model_order = [
    "GPT-5", "CLAUDE-4.5", "GEMINI-2.5-PRO",
    "GPT-oss-20b", "Llama-3.1-8B", "Mistral-7B"
]
avail_models = sorted(agg["model_display"].dropna().unique().tolist())
order_mapped = [model_alias.get(m, m) for m in preferred_model_order]
order = [m for m in order_mapped if m in avail_models] + \
        [m for m in avail_models if m not in order_mapped]
x_pos = {m:i for i,m in enumerate(order)}

fig, ax = plt.subplots(figsize=(7,3.5))

for combo in final_combos:
    sub = agg[agg["combo_key"] == combo]
    sub = sub[sub["model_display"].isin(order)]
    if sub.empty:
        continue
    xs = np.array([x_pos[m] for m in sub["model_display"]], dtype=float)
    ys = sub[chosen_col].values
    ax.scatter(
        xs, ys,
        c=[color_map.get(combo, "gray")],
        marker=marker_map.get(combo, "o"),
        s=120, alpha=0.9, edgecolors="none",
        label=combo_alias.get(combo, f"{sub['icl_label'].iloc[0]} · {sub['reasoning_label'].iloc[0]}")
    )

ax.grid(True, axis="y", linestyle="-", linewidth=0.7, alpha=0.35, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=0, ha="center")
ax.set_xlim(-0.5, len(order)-0.5)
for xi in range(len(order)):
    ax.axvline(x=xi, color="#eaeaea", lw=0.6, zorder=0)

if y_tick_mode == "0.5k":
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v/1000:.1f}k" if v>=1000 else f"{int(v)}"))
if isinstance(ylim, tuple) and len(ylim) == 2:
    ax.set_ylim(ylim)

ax.set_ylabel("Completion tokens per sample")
ax.set_title("Completion Tokens by Model", fontweight='bold')

ax.legend(
    title="Strategy · Reasoning",
    # ncol=3,
    ncol=1,
    # bbox_to_anchor=(0.5, -0.12),
    # loc="upper center",
    loc="upper right",
    # frameon=False,
    fontsize=9
)

plt.tight_layout()
output_path = Path("./plots/completion_tokens_by_model_avg_icl_reasoning_custom_order.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"[INFO] Saved to: {output_path}")
