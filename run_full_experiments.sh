#!/bin/bash

################################################################################
# Full Experimental Pipeline
# 
# Runs all experimental combinations systematically:
# - Dataset: GLOBEM
# - Sensor strategy: COMPASS
# - ICL strategies: zeroshot, cross_random, cross_retrieval, personal_recent, hybrid_blend
# - Reasoning: direct, cot, self_feedback (with constraints)
# - Models: Cloud (3) + Open-source (4)
#
# Total combinations: 49
# - Zeroshot + direct + 7 models = 7
# - 5 ICL + cot + 7 models = 35
# - cross_random + self_feedback + 7 models = 7
################################################################################

set -e
set -u

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_experiment() { echo -e "${MAGENTA}[EXPERIMENT]${NC} $1"; }

# ============================================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================================

# Dataset and sensor configuration
DATASET="globem"
SENSOR_STRATEGY="compass"

# ICL strategies
declare -a ICL_STRATEGIES=(
    "zeroshot"
    "cross_random"
    "cross_retrieval"
    "personal_recent"
    "hybrid_blend"
)

# Reasoning strategies (with constraints)
declare -a REASONING_STRATEGIES=(
    "direct"          # Only with zeroshot
    "cot"             # With all ICL strategies
    "self_feedback"   # Only with cross_random
)

# Models
declare -a CLOUD_MODELS=(
    "gpt-5|GPT-4o"
    "gemini-2.5-pro|Gemini-2.0-Pro"
    "claude-4.5-sonnet|Claude-3.5-Sonnet"
)

declare -a OPEN_MODELS=(
    "llama-3.2-3b|Llama-3.2-3B"
    "mistral-7b|Mistral-7B"
    "llama-3.1-8b|Llama-3.1-8B"
    "gemma2-9b|Gemma2-9B"
)

# Combine all models
ALL_MODELS=("${CLOUD_MODELS[@]}" "${OPEN_MODELS[@]}")
# ALL_MODELS=("${OPEN_MODELS[@]}")

# Common parameters
N_SAMPLES=100
N_SHOT=4  # For non-zeroshot strategies
SEED=42
LLM_SEED=42
CHECKPOINT_EVERY=10

# Python command
PYTHON_CMD="python"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Generate experiment name
get_experiment_name() {
    local icl=$1
    local reasoning=$2
    local n_shot=$3
    
    if [ "$icl" = "zeroshot" ]; then
        echo "${DATASET}_${SENSOR_STRATEGY}_0shot_${reasoning}_seed${SEED}"
    else
        echo "${DATASET}_${SENSOR_STRATEGY}_${n_shot}shot_${icl}_${reasoning}_seed${SEED}"
    fi
}

# Check if prompts exist
prompts_exist() {
    local exp_name=$1
    [ -d "saved_prompts/${exp_name}" ]
}

# Generate prompts if needed
ensure_prompts() {
    local icl=$1
    local exp_name=$2
    
    # Zeroshot doesn't need prompts
    if [ "$icl" = "zeroshot" ]; then
        return 0
    fi
    
    if prompts_exist "$exp_name"; then
        log_info "Prompts already exist: $exp_name"
        return 0
    fi
    
    log_info "Generating prompts: $exp_name"
    
    local dtw_flag=""
    if [ "$icl" = "cross_retrieval" ] || [ "$icl" = "hybrid_blend" ]; then
        dtw_flag="--use_dtw"
    fi
    
    ${PYTHON_CMD} run_evaluation.py \
        --mode save-prompts-only \
        --n_shot ${N_SHOT} \
        --strategy ${icl} \
        ${dtw_flag} \
        --reasoning cot \
        --seed ${SEED} \
        --save_prompts \
        || { log_error "Failed to generate prompts for $exp_name"; return 1; }
    
    log_success "Prompts generated: $exp_name"
    return 0
}

# Run single experiment
run_experiment() {
    local model_name=$1
    local model_display=$2
    local icl=$3
    local reasoning=$4
    local exp_name=$5
    local exp_num=$6
    local total_exps=$7
    
    log_experiment "[$exp_num/$total_exps] ${model_display} | ${icl} | ${reasoning}"
    
    local start_time=$(date +%s)
    
    # Build command
    local cmd="${PYTHON_CMD} run_evaluation.py"
    
    if [ "$icl" = "zeroshot" ]; then
        # Zeroshot: direct inference
        cmd="${cmd} --mode batch"
        cmd="${cmd} --n_samples ${N_SAMPLES}"
        cmd="${cmd} --n_shot 0"
    else
        # Few-shot: load prompts
        cmd="${cmd} --mode load-prompts"
        cmd="${cmd} --load_prompts ${exp_name}"
    fi
    
    cmd="${cmd} --model ${model_name}"
    cmd="${cmd} --reasoning ${reasoning}"
    cmd="${cmd} --llm_seed ${LLM_SEED}"
    cmd="${cmd} --checkpoint_every ${CHECKPOINT_EVERY}"
    cmd="${cmd} --verbose"
    
    # Execute
    if eval ${cmd}; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "Completed in ${duration}s"
        echo "${exp_num},${model_name},${icl},${reasoning},SUCCESS,${duration}" >> "${PROGRESS_FILE}"
        return 0
    else
        log_error "Failed"
        echo "${exp_num},${model_name},${icl},${reasoning},FAILED,0" >> "${PROGRESS_FILE}"
        return 1
    fi
}

# ============================================================================
# EXPERIMENTAL COMBINATIONS GENERATOR
# ============================================================================

generate_experiments() {
    local experiments=()
    local exp_num=1
    
    # Group 1: Zeroshot + direct + all models (7 experiments)
    log_info "Planning Group 1: Zeroshot + direct (7 experiments)"
    for model_info in "${ALL_MODELS[@]}"; do
        IFS='|' read -r model_name model_display <<< "${model_info}"
        experiments+=("${exp_num}|${model_name}|${model_display}|zeroshot|direct")
        ((exp_num++))
    done
    
    # Group 2: 5 ICL strategies + cot + all models (35 experiments)
    log_info "Planning Group 2: 5 ICL + cot (35 experiments)"
    for icl in "${ICL_STRATEGIES[@]}"; do
        for model_info in "${ALL_MODELS[@]}"; do
            IFS='|' read -r model_name model_display <<< "${model_info}"
            experiments+=("${exp_num}|${model_name}|${model_display}|${icl}|cot")
            ((exp_num++))
        done
    done
    
    # Group 3: cross_random + self_feedback + all models (7 experiments)
    log_info "Planning Group 3: cross_random + self_feedback (7 experiments)"
    for model_info in "${ALL_MODELS[@]}"; do
        IFS='|' read -r model_name model_display <<< "${model_info}"
        experiments+=("${exp_num}|${model_name}|${model_display}|cross_random|self_feedback")
        ((exp_num++))
    done
    
    echo "${experiments[@]}"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    local start_time=$(date +%s)
    
    # Setup
    mkdir -p results logs
    PROGRESS_FILE="logs/progress_$(date +%Y%m%d_%H%M%S).csv"
    SUMMARY_FILE="logs/summary_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "exp_num,model,icl_strategy,reasoning,status,duration_sec" > "${PROGRESS_FILE}"
    
    log_info "========================================"
    log_info "  FULL EXPERIMENTAL PIPELINE"
    log_info "========================================"
    log_info "Dataset: ${DATASET}"
    log_info "Sensor: ${SENSOR_STRATEGY}"
    log_info "Samples: ${N_SAMPLES}"
    log_info "Seed: ${SEED}"
    log_info "Progress log: ${PROGRESS_FILE}"
    log_info "========================================"
    echo
    
    # Generate experiment list
    log_info "Generating experimental combinations..."
    readarray -t EXPERIMENTS < <(generate_experiments | tr ' ' '\n')
    local total_experiments=${#EXPERIMENTS[@]}
    
    log_info "Total experiments planned: ${total_experiments}"
    log_info "  - Zeroshot + direct: 7"
    log_info "  - All ICL + cot: 35"
    log_info "  - cross_random + self_feedback: 7"
    echo
    
    # Confirmation
    read -p "Do you want to proceed with all ${total_experiments} experiments? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warning "Aborted by user"
        exit 0
    fi
    echo
    
    # Generate required prompts
    log_info "========================================"
    log_info "  STEP 1: PROMPT GENERATION"
    log_info "========================================"
    echo
    
    declare -A prompt_sets_needed
    for exp_info in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_num model_name model_display icl reasoning <<< "${exp_info}"
        if [ "$icl" != "zeroshot" ]; then
            local exp_name=$(get_experiment_name "$icl" "$reasoning" "$N_SHOT")
            prompt_sets_needed["${exp_name}|${icl}"]=1
        fi
    done
    
    log_info "Unique prompt sets needed: ${#prompt_sets_needed[@]}"
    
    for key in "${!prompt_sets_needed[@]}"; do
        IFS='|' read -r exp_name icl <<< "${key}"
        ensure_prompts "$icl" "$exp_name" || {
            log_error "Failed to generate prompts for $exp_name"
            exit 1
        }
    done
    
    log_success "All prompts ready!"
    echo
    
    # Run experiments
    log_info "========================================"
    log_info "  STEP 2: RUN EXPERIMENTS"
    log_info "========================================"
    echo
    
    local success_count=0
    local fail_count=0
    declare -a failed_experiments
    
    for exp_info in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_num model_name model_display icl reasoning <<< "${exp_info}"
        
        local exp_name=$(get_experiment_name "$icl" "$reasoning" "$N_SHOT")
        
        if run_experiment "$model_name" "$model_display" "$icl" "$reasoning" "$exp_name" "$exp_num" "$total_experiments"; then
            ((success_count++))
        else
            ((fail_count++))
            failed_experiments+=("$exp_num: $model_display | $icl | $reasoning")
        fi
        
        echo
        
        # Small delay between experiments
        if [ $exp_num -lt $total_experiments ]; then
            log_info "Waiting 3 seconds before next experiment..."
            sleep 3
        fi
    done
    
    # Summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local total_hours=$((total_duration / 3600))
    local total_minutes=$(((total_duration % 3600) / 60))
    local total_seconds=$((total_duration % 60))
    
    log_info "========================================"
    log_info "  FINAL SUMMARY"
    log_info "========================================"
    log_info "Total experiments: ${total_experiments}"
    log_info "Successful: ${success_count}"
    log_info "Failed: ${fail_count}"
    log_info "Total time: ${total_hours}h ${total_minutes}m ${total_seconds}s"
    log_info "Progress log: ${PROGRESS_FILE}"
    echo
    
    # Save summary
    {
        echo "Full Experimental Pipeline Summary"
        echo "=================================="
        echo "Date: $(date)"
        echo "Total experiments: ${total_experiments}"
        echo "Successful: ${success_count}"
        echo "Failed: ${fail_count}"
        echo "Total time: ${total_hours}h ${total_minutes}m ${total_seconds}s"
        echo ""
        echo "Configuration:"
        echo "  Dataset: ${DATASET}"
        echo "  Sensor: ${SENSOR_STRATEGY}"
        echo "  Samples: ${N_SAMPLES}"
        echo "  Seed: ${SEED}"
        echo ""
        if [ ${fail_count} -gt 0 ]; then
            echo "Failed experiments:"
            for failed in "${failed_experiments[@]}"; do
                echo "  - ${failed}"
            done
        fi
    } > "${SUMMARY_FILE}"
    
    if [ ${fail_count} -gt 0 ]; then
        log_warning "Some experiments failed:"
        for failed in "${failed_experiments[@]}"; do
            log_warning "  - ${failed}"
        done
    fi
    
    log_info "Summary saved to: ${SUMMARY_FILE}"
    
    if [ ${success_count} -eq ${total_experiments} ]; then
        log_success "All experiments completed successfully! 🎉"
        exit 0
    else
        log_warning "Pipeline completed with ${fail_count} failures"
        exit 1
    fi
}

# Run main
main "$@"

