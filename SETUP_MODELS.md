# Model Setup Guide

This guide explains how to set up LLM models for your mental health prediction experiments.

## 📊 Supported Models

Just use the model name - provider is detected automatically!

### ☁️ Cloud-Based Models (API Required)
- `gpt-5-nano` (OpenAI)
- `claude-3-5-sonnet` (Anthropic)
- `gemini-2.5-pro` (Google)

### 🏠 On-Device Models (via Ollama)
- `llama-3.1-8b` (LLaMA 3.1 8B)
- `mistral-7b` (Mistral 7B)
- `alpaca-7b` (Alpaca 7B)

**Note**: Ollama uses **4-bit quantization** for practical deployment on consumer hardware. This represents realistic on-device deployment scenarios.

---

## 🔑 Step 1: Set Up API Keys (for Cloud Models)

### Create `api_keys.json`

Copy the template:

```bash
cp api_keys.json.template api_keys.json
```

Edit `api_keys.json` with your actual API keys:

```json
{
  "openai": "sk-YOUR-ACTUAL-OPENAI-KEY",
  "anthropic": "sk-ant-YOUR-ACTUAL-ANTHROPIC-KEY",
  "google": "YOUR-ACTUAL-GOOGLE-KEY",
  "ollama": {
    "base_url": "http://localhost:11434"
  }
}
```

**⚠️ Important**: The file is already in `.gitignore` to protect your keys!

### Get API Keys

**OpenAI (GPT models)**:
- Visit: https://platform.openai.com/api-keys
- Create new secret key
- Copy the key (starts with `sk-`)

**Anthropic (Claude models)**:
- Visit: https://console.anthropic.com/settings/keys
- Create API key
- Copy the key (starts with `sk-ant-`)

**Google (Gemini models)**:
- Visit: https://makersuite.google.com/app/apikey
- Create API key
- Copy the key

---

## 🏠 Step 2: Install Ollama (for On-Device Models)

### Why Ollama?

✅ **On-Device/Local** - Runs on your machine, not in the cloud  
✅ **No API Keys** - No cost, no API limits  
✅ **Privacy** - Data never leaves your machine  
✅ **4-bit Quantization** - Optimized for practical deployment  
✅ **Perfect for Research** - Compare cloud vs on-device capabilities  

### Install Ollama

**Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS**:
```bash
brew install ollama
```

**Windows**:
Download from https://ollama.com/download

### Start Ollama Service

```bash
# Start Ollama (runs in background)
ollama serve
```

Keep this running in a separate terminal!

### Download Models

```bash
# Download models (use Ollama's naming format for download)
ollama pull llama3.1:8b      # LLaMA 3.1 8B (~4.7GB)
ollama pull mistral:7b       # Mistral 7B (~4.1GB)
ollama pull alpaca:7b        # Alpaca 7B
```

**Model Sizes**:
- 7B-8B models: ~4-5 GB
- 13B models: ~7-8 GB

### Verify Installation

```bash
# List downloaded models
ollama list

# Test a model
ollama run llama3.1:8b "Hello, how are you?"
```

---

## 📦 Step 3: Install Python Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually:
pip install openai              # OpenAI (required if using GPT)
pip install anthropic           # Anthropic (optional, for Claude)
pip install google-generativeai # Google (optional, for Gemini)
pip install ollama              # Ollama (optional, for on-device)
```

---

## 🚀 Step 4: Run Experiments

### Cloud Models

```bash
# OpenAI GPT-5-nano
python src/run_evaluation.py --mode batch --model gpt-5-nano --n_samples 50

# Anthropic Claude-3.5-Sonnet
python src/run_evaluation.py --mode batch --model claude-3-5-sonnet --n_samples 50

# Google Gemini-2.5-Pro
python src/run_evaluation.py --mode batch --model gemini-2.5-pro --n_samples 50
```

### On-Device Models (Ollama)

```bash
# Make sure Ollama is running: ollama serve

# LLaMA 3.1 8B
python src/run_evaluation.py --mode batch --model llama-3.1-8b --n_samples 50

# Mistral 7B
python src/run_evaluation.py --mode batch --model mistral-7b --n_samples 50

# Alpaca 7B
python src/run_evaluation.py --mode batch --model alpaca-7b --n_samples 50
```

---

## 🔬 Research Workflow: Cloud vs On-Device Comparison

### Step 1: Save Prompts with Reference Model

```bash
python src/run_evaluation.py --mode batch \
  --model gpt-5-nano --save-prompts --seed 42 --n_samples 100
```

This saves all prompts to `saved_prompts/globem_structured_5shot_hybrid_cot_42/`

### Step 2: Test Cloud Models (Same Prompts)

```bash
python src/run_evaluation.py --mode batch \
  --load-prompts globem_structured_5shot_hybrid_cot_42 \
  --model claude-3-5-sonnet

python src/run_evaluation.py --mode batch \
  --load-prompts globem_structured_5shot_hybrid_cot_42 \
  --model gemini-2.5-pro
```

### Step 3: Test On-Device Models (Same Prompts)

```bash
python src/run_evaluation.py --mode batch \
  --load-prompts globem_structured_5shot_hybrid_cot_42 \
  --model llama-3.1-8b

python src/run_evaluation.py --mode batch \
  --load-prompts globem_structured_5shot_hybrid_cot_42 \
  --model mistral-7b
```

**Result**: Fair comparison across all models with identical prompts!

---

## 🔧 Troubleshooting

### Ollama Connection Error

```bash
# Make sure Ollama is running
ollama serve

# Check if it's running
curl http://localhost:11434
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the model if missing
ollama pull llama3.1:8b
```

### API Key Errors

- Check `api_keys.json` exists and has valid keys
- Make sure keys are not expired
- Verify you have credits/quota remaining

### Out of Memory (Ollama)

```bash
# Use smaller models
ollama pull llama2:7b  # Instead of larger models

# Or quantized versions (already 4-bit by default in Ollama)
```

---

## 📝 For Your Research Paper

### Method Section - Ollama Deployment

```
We evaluate both cloud-based APIs (GPT-4o, Claude-3.5-Sonnet, Gemini-2.5-Pro) 
and on-device deployment via Ollama (LLaMA-3.1-8B, Mistral-7B). Ollama employs 
4-bit quantization to enable efficient inference on consumer hardware, 
representing realistic on-device deployment scenarios. While quantization may 
introduce slight accuracy degradation compared to cloud-based full-precision 
models, this reflects the inherent trade-off between accuracy and practical 
deployment constraints (compute, memory, privacy, cost).
```

### Cloud vs On-Device Trade-offs

**Cloud-Based Models**:
- ✅ Higher accuracy (larger models, full precision)
- ✅ No local compute requirements
- ❌ API costs per request
- ❌ Privacy concerns (data sent to cloud)
- ❌ Requires internet connection

**On-Device Models (Ollama)**:
- ✅ Zero API cost
- ✅ Privacy-preserving (data stays local)
- ✅ Works offline
- ✅ Lower latency (no network overhead)
- ❌ Requires local GPU/CPU
- ❌ Quantization may reduce accuracy

---

## 🎯 Quick Start

```bash
# 1. Set up API keys
cp api_keys.json.template api_keys.json
# Edit api_keys.json with your keys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # Keep running in separate terminal
ollama pull llama3.1:8b

# 4. Test cloud model
python src/run_evaluation.py --mode single --model gpt-5-nano

# 5. Test on-device model
python src/run_evaluation.py --mode single --model llama-3.1-8b
```

You're ready to compare cloud vs on-device models! 🚀