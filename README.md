# 🧭 COMPass: Context-Oriented Mental Health Modeling with LLMs

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](.)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-GLOBEM%20|%20CES%20|%20Mental--IoT-purple)](.)

**Context makes or breaks LLM-based mental health sensing.**

</div>

---

## 💡 Overview

**COMPass** is a study designed for **systematic** evaluation (with reproducibility in mind) of how *context* governs LLM behavior in passive-sensing mental health prediction. Rather than a single model, we treat context as a three-axis design space under a shared **clinical instruction**—functional-impairment framing, domain cues, and a fixed output schema.

- **📊 Sensor-to-Text Representation** — how multiday signals are expressed as textual evidence  
- **🎯 In-Context Learning (ICL) Strategy** — how demonstrations are sourced and selected (zero-shot / cross-user-random / cross-user-retrieval / personal-recent / hybrid)
- **🧠 Reasoning Method** — how the model reasons (direct prediction / chain-of-thought / self-refinement)

We implement a consistent pipeline of **Preparation → Context Engineering → Evaluation** across GLOBEM, CES, and Mental-IoT and multiple LLM back-ends, reporting performance (mainly Macro-F1) alongside efficiency indicators (e.g., token usage and latency). The goal is **practical design guidance**: when representation, exemplar policy, and reasoning align to improve both performance and efficiency, when they conflict, and a sensible default (e.g., *personal-recent + CoT*) for real-world, personalized sensing systems.


## 📑 Table of Contents

- [About](#-about)
- [Results (RQ1–RQ3)](#-results-mapped-to-rq1rq3)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Supplementary Materials](#-supplementary-materials)
- [Dataset Citations](#dataset-citations)

---

## 📖 About

### Motivation
LLMs are promising for behavioral and mental health modeling, but their effectiveness depends critically on **how context is constructed**. COMPass offers a systematic investigation of context design for sensor-based prediction.

### Research Questions
1. **RQ1**: How can we **systematically design an end-to-end LLM pipeline** that predicts users' mental health states from multimodal sensing data?
2. **RQ2**: How much can **context-aware LLMs** improve mental-health-state prediction from multimodal sensing data compared to **traditional ML models** and **LLMs without contextual enrichment**?
3. **RQ3**: What is the **optimal pipeline configuration** considering both **prediction accuracy** and **computational efficiency**?

### Datasets
<div align="center">
  <img src="assets/dataset.png" width="100%" alt="Dataset overview">
  <p><i>Figure: Dataset overview</i></p>
</div>

### Context Dimensions Explored

#### 1) Sensor-to-Text Representation
We compare representation formats on GLOBEM:
- [**Method A**](https://dl.acm.org/doi/abs/10.1145/3659604): markdown table with daily aggregates
- [**Method B**](https://arxiv.org/abs/2401.06866): statistical summaries (e.g., mean, std, min, max)
- **Our Baseline**: integrates statistical, structural, and temporal features with natural-language descriptions (inspired by [SensorLM](https://arxiv.org/abs/2506.09108))

#### 2) In-Context Learning Strategies
<div align="center">
  <img src="assets/reasoning.png" width="900" alt="ICL strategies comparison">
  <p><i>Figure: ICL strategies and their impact at a glance</i></p>
</div>

- **Zero-shot** (no demonstrations)
- **Cross-Random** (random cross-user examples)
- **Cross-Retrieval** (DTW-based retrieval from other users)
- **Personal-Recent** (recent examples from the same user)
- **Hybrid** (combination of cross-user and personal)

#### 3) Reasoning Methods
- **Direct Prediction (DP)** — single-step classification  
- **Chain-of-Thought (CoT)** — structured, step-by-step reasoning  
- **Self-Refinement (SR)** — critique-and-revise iteration

---

## 📊 Results (mapped to RQ1–RQ3)

### RQ1 — Pipeline Overview
Our pipeline is organized into three stages:

1. **Preparation** — LLM model selection and data preprocessing  
2. **Context Engineering** — sensor data representation (fixed) + prompt construction (fixed) + exemplar selection (Zero-shot / Cross-Random / Cross-Retrieval / Personal-Recent / Hybrid) + reasoning design (DP / CoT / SR)
3. **Evaluation** — consistent metrics and efficiency reporting across datasets and models

This structure encourages reproducible comparisons and isolates the effects of example selection and reasoning.

### RQ2 — Predictive Performance

#### GLOBEM
<div align="center">
  <img src="assets/globem_icl.png" width="900" alt="GLOBEM results">
  <p><i>Figure: ICL strategies on GLOBEM</i></p>
</div>

**Observations (concise)**
- **Personal-Recent** and **Hybrid** consistently outperform cross-user baselines.  
- **CoT** helps **when context is reliable**; with poorly matched cross-user examples it can be neutral or even harmful.  
- Well-matched cross-user retrieval can be competitive when it captures the target user’s **behavioral regime**.

#### CES (College Students)
<div align="center">
  <img src="assets/ces_icl.png" width="900" alt="CES results">
  <p><i>Figure: ICL strategies on CES</i></p>
</div>

**Observations (concise)**
- Personalization yields the most stable gains.  
<!-- - **CoT** improves interpretability and stability; **SR** can add robustness on harder cases. -->

#### Mental-IoT (General Population)
<div align="center">
  <img src="assets/miot_icl.png" width="900" alt="Mental-IoT results">
  <p><i>Figure: ICL strategies on Mental-IoT</i></p>
</div>

**Observations (concise)**
- Harder overall due to sparsity/short windows; benefits vary by target.  
<!-- - Depending on the setting, **CoT** or **SR** can be preferable; personalization remains helpful when available. -->

#### Reasoning Comparison (beyond a Macro-F1 table)

<div align="center">
  <img src="assets/reasoning.png" width="900" alt="Reasoning comparison">
  <p><i>Figure: Reasoning variants (DP / CoT / SR)</i></p>
</div>

**Key Observations**
- **Dataset/setting dependence**: In **zero-shot settings, DP outperforms CoT on CES/GLOBEM**, while **CoT outperforms DP on Mental-IoT**—multi-step reasoning without demonstrations is dataset-dependent, not uniformly helpful.
- **Reasoning × context coupling**: With **Personal-Recent** exemplars, **DP already matches CoT** (limited sensitivity to reasoning); with **Cross-Random**, structured reasoning can **underperform zero-shot**. Well-aligned exemplars are the precondition for reasoning gains.
- **Few-shot nuance**: Under **four-shot Personal-Recent**, no single reasoning strategy dominates across datasets; **on Mental-IoT, Self-Refinement is consistently best**, but this advantage is **not universal**.
- **Heterogeneity matters**: Variation within and across users makes models fragile to poorly matched context; **exemplar source and recency—rather than quantity—are the dominant drivers** of gains.
- **Cross-user retrieval**: Similarity-based **Cross-Retrieval showed limited benefit vs. zero-shot** in our settings; **Hybrid/Personal-Recent** were more reliable.

### RQ3 — Optimal Pipeline Configuration (Performance × Efficiency)

<div align="center">
  <img src="assets/performance_efficiency.png" width="900" alt="Performance and efficiency">
  <p><i>Figure: Performance–efficiency landscape under context and reasoning choices</i></p>
</div>

**Token Cost Drivers (cost/usage)**
- In our runs, completion tokens varied largely by **model family** and **reasoning depth**, with prompts held constant across models; output tokens are the key cost driver and full distributions are available in the repo.
- Within a given model, moving from **Direct Prediction (DP)** to **Chain-of-Thought (CoT)** typically increases completion length, while moving from **zero-shot** to **few-shot** tends to contribute a smaller increment; see tables/plots for per-setting details.

**Latency × Performance (how they co-move)**
- **Context design materially affects both metrics**: in several cases, **higher Macro-F1 coincided with equal or lower latency**—i.e., not a strict trade-off.  
  - **CES:** **Personal-Recent** outperformed **cross-user random** on **both** Macro-F1 and latency; **Hybrid** ranked second.  
  - **GLOBEM:** With **Personal-Recent** demonstrations, Macro-F1 was **similar across reasoning methods** (small spread). **Personal-Recent + Direct Prediction** performed **on par** with higher-latency reasoning.  
  - **Mental-IoT:** In zero-shot, **CoT** was strong at moderate latency; with **Personal-Recent** demonstrations and **Self-Refinement (SR)**, top Macro-F1 was reached with a larger latency budget.

**Design Rules (practical guide)**
- **No strict trade-off:** With well-chosen demonstrations and a suitable reasoning mode, accuracy **and** latency can improve together.  
- **Context > Size:** Prioritize **exemplar relevance and recency**; gains are driven more by where examples come from than by simply adding more.  
- **A reasonable starting point:** **Personal-Recent** exemplars plus a lightweight reasoning cue (e.g., **CoT**) provide a balanced baseline; when latency is critical, **Personal-Recent + Direct Prediction** can be competitive.

> **Note**: The README summarizes key trends. See `experiment_summary.csv` and result JSON files for detailed accuracy/latency/token/cost data.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ (Implemented with Python 3.12.3)
- API keys for at least one LLM provider:
  - OpenAI (GPT models)
  - Anthropic (Claude) via OpenRouter *(optional)*
  - Google (Gemini) *(optional)*
  - Ollama for local models *(optional)*

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/compass-sensor-llm-mh-prediction.git
cd compass-sensor-llm-mh-prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up API keys**

Create `src/api_keys.json` with your API credentials:

```json
{
  "openai_api_key": "sk-...",
  "openrouter_api_key": "sk-or-v1-..."
}
```

> **Note**: Only include keys for providers you intend to use. The system will automatically detect available providers.

4. **Configure dataset paths**

Edit `src/config.py` to point to your dataset locations:

```python
GLOBEM_BASE_PATH = '../dataset/Globem'
CES_BASE_PATH = '../dataset/CES'
# ... set DATASET_TYPE to 'globem', 'ces', or 'mentaliot'
```

---

## 💻 Usage

### Quick Start

**Step 1: Test with a single prediction**

Run a single prediction to verify your setup (default: 4-shot Hybrid ICL + CoT):

```bash
python run_evaluation.py --mode single --verbose
```

**Step 2: Generate and save prompts for reproducible experiments**

Since we test multiple LLM models (GPT-4o, Claude, Gemini, etc.) on the **same prompts** to ensure fair comparison, we first generate and save prompts:

```bash
# Generate prompts for full test set and save them
python run_evaluation.py --mode batch --seed 42 \
  --strategy personal_recent --n_shot 4 --reasoning cot \
  --save-prompts
```

This creates a folder in `saved_prompts/` with the experiment configuration (e.g., `ces_compass_4shot_personalrecent_cot_seed42/`).

> **💡 For quick testing**: Add `--n_samples 5` to test with a smaller subset before running the full evaluation. - only 5 sample prediction

**Step 3: Run experiments with different models using saved prompts**

Now test different models on the **exact same prompts**:

```bash
# Test with GPT-4o (with checkpointing for long runs)
python run_evaluation.py --mode batch --load-prompts ces_compass_4shot_personalrecent_cot_seed42 \
  --model gpt-5 --checkpoint-every 10

# Test with Claude Sonnet 4.5
python run_evaluation.py --mode batch --load-prompts ces_compass_4shot_personalrecent_cot_seed42 \
  --model claude-4.5-sonnet --checkpoint-every 10

# Test with Gemini 2.5 Pro
python run_evaluation.py --mode batch --load-prompts ces_compass_4shot_personalrecent_cot_seed42 \
  --model gemini-2.5-pro --checkpoint-every 10
```

This workflow ensures that **all model comparisons use identical inputs**, eliminating variability from prompt generation and enabling true apples-to-apples comparison.

**Checkpoint Files Structure**:
```
results/
├── ces_compass_4shot_personalrecent_cot_seed42_gpt_5_cot_42_20250111_143022_checkpoint_10.json
├── ces_compass_4shot_personalrecent_cot_seed42_gpt_5_cot_42_20250111_143022_checkpoint_20.json
├── ces_compass_4shot_personalrecent_cot_seed42_gpt_5_cot_42_20250111_143022_checkpoint_30.json
└── ... (saved every 10 samples)
```

Each checkpoint contains:
- All predictions up to that point
- Metadata (timestamps, model config, etc.)
- Progress information for resumption

**Resume from checkpoint** if interrupted:
```bash
python run_evaluation.py --mode batch \
  --resume-from results/ces_compass_4shot_personalrecent_cot_seed42_gpt_5_cot_42_20250111_143022_checkpoint_30.json
```

> **💡 Tips**: 
> - Use `--checkpoint-every 10` for experiments with large samples to avoid losing progress
> - Use `--save-prompts-only` to generate prompts without calling any LLM first
> - Checkpoints are automatically cleaned up (keeps only the most recent one) to save disk space

### Configuration Options

#### ICL Strategies

```bash
# Zero-shot (no examples)
python run_evaluation.py --strategy none --n_shot 0

# Cross-random (random examples from other users)
python run_evaluation.py --strategy cross_random --n_shot 4

# Cross-retrieval (DTW-based retrieval from other users)
python run_evaluation.py --strategy cross_retrieval --n_shot 4

# Personal-recent (recent examples from same user)
python run_evaluation.py --strategy personal_recent --n_shot 4

# Hybrid (mix of cross-random + personal)
python run_evaluation.py --strategy hybrid_blend --n_shot 4
```

**ICL Selection Details**:

- For **Cross-Retrieval** and **Hybrid**, we use TimeRAG-accelerated DTW for similarity-based retrieval
- For **Hybrid**, we combine: \( k/2 \) cross-user examples + \( k/2 \) personal-recent examples
- See [Cross-Retrieval Method](#-cross-retrieval-method-timerag) section below for algorithm details

#### Reasoning Methods

```bash
# Direct prediction (fastest)
python run_evaluation.py --reasoning direct

# Chain-of-Thought (recommended)
python run_evaluation.py --reasoning cot

# Self-refinement (iterative improvement)
python run_evaluation.py --reasoning self_feedback
```

#### Model Selection

```bash
# GPT-4o (default, best performance)
python run_evaluation.py --model gpt-5

# Claude Sonnet 4.5
python run_evaluation.py --model claude-4.5-sonnet

# Gemini 2.5 Pro
python run_evaluation.py --model gemini-2.5-pro

# Open-source 20B model
python run_evaluation.py --model gpt-oss-20b
```

#### Additional Options

**Run end-to-end without saving prompts** (not recommended for multi-model comparison):

```bash
# Full test set
python run_evaluation.py --mode batch --seed 42

# Or with a smaller subset for testing
python run_evaluation.py --mode batch --n_samples 30 --seed 42
```

### Configuration Files

The system uses YAML and JSON configurations in the `config/` directory:

- **`prompt_configs.yaml`**: Prompt templates and experimental presets
- **`globem_use_cols.json`**: Feature selection for GLOBEM dataset
- **`ces_use_cols.json`**: Feature selection for CES dataset  
- **`mentaliot_use_cols.json`**: Feature selection for Mental-IoT dataset

You can modify `src/config.py` to adjust:
- Dataset selection (`DATASET_TYPE`)
- Feature aggregation windows (`AGGREGATION_WINDOW_DAYS`)
- ICL parameters (`DEFAULT_N_SHOT`, `MIN_HISTORICAL_LABELS`)
- Model settings (`DEFAULT_MODEL`, `DEFAULT_TEMPERATURE`)

---

## 📁 Repository Structure

```
compass-sensor-llm-mh-prediction/
├── config/                          # Configuration files
│   ├── prompt_configs.yaml         # Prompt templates & presets
│   ├── globem_use_cols.json        # GLOBEM feature configuration
│   ├── ces_use_cols.json           # CES feature configuration
│   └── mentaliot_use_cols.json     # Mental-IoT feature configuration
│
├── src/                             # Source code
│   ├── config.py                   # Central configuration constants
│   ├── sensor_transformation.py    # Data loading & preprocessing
│   ├── data_utils.py               # Test set sampling utilities
│   ├── prompt_manager.py           # Prompt construction from YAML
│   ├── example_selection.py        # ICL example selection strategies
│   ├── timerag_retrieval.py        # TimeRAG-accelerated DTW retrieval
│   ├── reasoning.py                # LLM reasoning methods (CoT, SR, etc.)
│   ├── llm_client.py               # Multi-provider LLM API wrapper
│   ├── evaluation_runner.py        # Batch evaluation orchestration
│   ├── performance.py              # Metrics calculation & reporting
│   └── api_keys.json               # API credentials (not in git)
│
├── assets/                          # Figures and visualizations
│
├── results/                         # Evaluation outputs
│   └── [experiment_results].json   # Per-experiment result files
│
├── saved_prompts/                   # Cached prompts (for reuse)
│   └── [experiment_name]/          # Organized by configuration
│
├── example_prompt_and_result/       # Example predictions (see below)
│
├── run_evaluation.py                # Main entry point
├── requirements.txt                 # Python dependencies
├── experiment_summary.csv           # Aggregated results across experiments
└── README.md                        # This file

dataset/
├── CES/                             # Download from link below
├── Globem/                          # Download from link below
└── MentalIot/                       # Download from link below
```

### 📂 Example Results

Due to dataset End-User License Agreements (EULAs), we cannot publicly share all per-sample prediction results. However, we provide representative examples in [`example_prompt_and_result/`](./example_prompt_and_result):

- **Prompts**: Complete prompt structure including ICL examples and reasoning instructions
- **Model Outputs**: Raw predictions and reasoning traces
- **Metadata**: Sample information and ground truth labels

**Available Examples**:
- Mental-IoT dataset with 4-shot Personal-Recent + Self-Refinement strategy
- See [`example_prompt_and_result/mentaliot_compass_4shot_personalrecent_selffeedback_seed42_prompt/`](./example_prompt_and_result/mentaliot_compass_4shot_personalrecent_selffeedback_seed42_prompt/) for full prompt files

> **Note**: For full experimental results on your own data, please follow the [Usage](#-usage) instructions to run evaluations with your dataset access.

---

## 📚 Supplementary Materials

### 🎨 Prompt Visualizations

#### Full Prompt Structure

Below is a complete example of how prompts are constructed for mental health prediction tasks, showing the integration of system instructions, ICL examples, and reasoning guidance:

<div align="center">
  <img src="assets/full_prompt_example.png" width="900" alt="Full Prompt Example">
  <p><i>Figure: Complete prompt structure showing system instructions, feature descriptions, ICL examples, and task specification</i></p>
</div>

#### Reasoning Strategy - Prompt

Different reasoning methods guide the LLM's cognitive process in distinct ways:

**Direct Prediction (DP)**

<div align="center">
  <img src="assets/prompt_reasoning_DP.png" width="900" alt="Direct Prediction Prompt">
  <p><i>Figure: Direct prediction prompt - single-step classification without intermediate reasoning</i></p>
</div>

**Chain-of-Thought (CoT)**

<div align="center">
  <img src="assets/prompt_reasoning_CoT.png" width="900" alt="Chain-of-Thought Prompt">
  <p><i>Figure: Chain-of-Thought prompt - structured step-by-step reasoning process</i></p>
</div>

**Self-Refinement (SR)**

<div align="center">
  
- **Initial Prediction**:
<img src="assets/prompt_reasoning_SR_initial.png" width="900" alt="Self-Refinement Initial">
<p><i>Figure: Self-refinement initial prediction phase</i></p>

- **Critique & Refinement**:
<img src="assets/prompt_reasoning_SR_refined.png" width="900" alt="Self-Refinement Refined">
<p><i>Figure: Self-refinement critique and revision phase</i></p>

</div>

---

### 🔍 Cross-Retrieval Method (TimeRAG)

For Cross-Retrieval and Hybrid-Blend ICL strategies, we implement an efficient retrieval system based on **TimeRAG** (Time-series Retrieval Augmented Generation). This approach combines:

Our retrieval pipeline consists of three main steps:

**Step 1: Clustering** — Reduce training set from N samples to M=300 representatives using K-Means clustering

**Step 2: DTW Ranking** — Compute DTW distance between target and each representative, retrieve top-(k×α) candidates

**Step 3: Label Stratification** — Balance selected examples across outcome classes (anxiety × depression)

```
Input: Training data D, target series T, k examples
Output: k balanced ICL examples

1. Cluster D into M=300 groups, select centroid-nearest from each
2. Compute DTW(T, representative) for all M representatives  
3. Retrieve top-(k×2) by DTW distance
4. Select k examples with balanced labels (stratified sampling)
```

---

### Quick Links to Visualizations

<div align="center">

| GLOBEM ICL | CES ICL | Mental-IoT ICL |
|:----------:|:-------:|:--------------:|
| <img src="assets/globem_icl.png" width="250"> | <img src="assets/ces_icl.png" width="250"> | <img src="assets/miot_icl.png" width="250"> |

</div>

---


### Dataset Citations

- **GLOBEM**: [Paper](https://dl.acm.org/doi/abs/10.1145/3569485), [Dataset](https://the-globem.github.io/)
- **CES**: [Paper](https://dl.acm.org/doi/10.1145/3643501), [Dataset](https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset)
- **Mental-IoT**: [Paper](https://dl.acm.org/doi/abs/10.1145/3749485), [Dataset](https://github.com/Kaist-ICLab/multimodal-mh-detection)


---

<div align="center">

**🧭 COMPass: Because context is everything.**

</div>
