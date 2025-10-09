"""
Mental Health Prediction Evaluation Runner - Main entry point for LLM-based mental health prediction.

Usage:
  python run_evaluation.py --mode single
  python run_evaluation.py --mode batch --n_samples 5 --seed 42 --stratified --stratify_by phq4_anxiety_EMA --save-prompts
  python run_evaluation.py --mode batch --load-prompts EXP_NAME --model MODEL_NAME
"""

import argparse
import json
import os
from datetime import datetime

from src.sensor_transformation import load_globem_data, get_data_statistics, print_data_statistics
from src.reasoning import LLMReasoner
from src.prompt_manager import PromptManager
from src.prompt_utils import (
    NumpyEncoder,
    save_prompts_to_disk,
    load_prompts_from_disk,
    build_experiment_prefix,
)
from src.evaluation_runner import run_single_prediction, run_batch_evaluation, run_batch_with_loaded_prompts
from src.performance import export_comprehensive_report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Sensor LLM-based Mental Health Prediction Evaluation')
    
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'])
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--source', type=str, default='hybrid',
                       choices=['personalized', 'generalized', 'hybrid'])
    parser.add_argument('--reasoning', type=str, default='cot',
                       choices=['direct', 'cot', 'tot', 'sc'])
    parser.add_argument('--model', type=str, default='gpt-5-nano',
                       choices=['gpt-5-nano', 'claude-4.0-sonnet', 'gemini-2.5-pro', 'llama-3.1-8b', 'mistral-7b', 'alpaca-7b'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--stratify_by', type=str, default='phq4_anxiety_EMA',
                       choices=['phq4_anxiety_EMA', 'phq4_depression_EMA'])
    parser.add_argument('--show_stats', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--save-prompts', action='store_true')
    parser.add_argument('--load-prompts', type=str, default=None)
    parser.add_argument('--prompts-dir', type=str, default='./saved_prompts')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Print config
    print("\n" + "="*60)
    print("🚀 LLM MENTAL HEALTH PREDICTION EVALUATION")
    print("="*60)
    print(f"  Mode: {args.mode} | Model: {args.model}")
    if not args.load_prompts:
        print(f"  ICL: {args.source} | N-Shot: {args.n_shot} | Reasoning: {args.reasoning}")
    if args.seed:
        print(f"  Random Seed: {args.seed}")
    if args.load_prompts:
        print(f"  Loading Prompts: {args.load_prompts}")
    print("="*60)
    
    # Optimization: Skip data/prompt loading when using --load-prompts
    if args.load_prompts:
        print("\n🔄 Model Comparison Mode")
        print("  ⚡ Skipping data loading (not needed)")
        print("  ⚡ Skipping prompt manager (prompts pre-generated)")
        feat_df, lab_df, cols, prompt_manager = None, None, None, None
    else:
        print("\n📂 Loading GLOBEM dataset...")
        feat_df, lab_df, cols = load_globem_data()
        print(f"  Features: {feat_df.shape[0]} rows | Labels: {lab_df.shape[0]} rows")
        
        ### 제거 후보
        if args.show_stats:
            stats = get_data_statistics(lab_df, cols)
            print_data_statistics(stats)
        ###
        
        print("\n🎨 Initializing Prompt Manager...")
        prompt_manager = PromptManager()
        print("  ✓ YAML templates loaded")
    
    # Initialize LLM reasoner
    print(f"\n🤖 Initializing LLM Reasoner ({args.model})...")
    reasoner = LLMReasoner(model=args.model)
    print("  ✓ Ready")
    
    model_name = args.model.replace('/', '_').replace('-', '_').replace('.', '_')
    # Common prefix per spec
    exp_prefix = build_experiment_prefix(args.n_shot, args.source, seed=args.seed)
    # Run evaluation
    if args.mode == 'single':
        result = run_single_prediction(
            prompt_manager, reasoner, feat_df, lab_df, cols, n_shot=args.n_shot,
            source=args.source, reasoning_method=args.reasoning, random_state=args.seed,
            llm_seed=args.llm_seed, verbose=args.verbose
        )
        
        if result:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # dataset_sensor_nshot_source_seed_Model_reasoning_llmseed_timestamp.json
            filename = f"{exp_prefix}_{model_name}_{args.reasoning}_{args.llm_seed}_{timestamp}.json"
            result_file = os.path.join(args.output_dir, filename)
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n✅ Result saved to: {result_file}")
    
    elif args.mode == 'batch':
        if args.load_prompts:
            print(f"\n🔄 Loading prompts from: {args.load_prompts}")
            prompts, labels, metadata = load_prompts_from_disk(args.load_prompts)
            
            # labels contains the sample metadata (user_id, dates, true labels)
            result = run_batch_with_loaded_prompts(
                reasoner, prompts, labels, reasoning_method=args.reasoning,
                llm_seed=args.llm_seed, verbose=args.verbose
            )
            # Use provided folder (should be prefix) as exp_prefix
            exp_prefix = os.path.basename(args.load_prompts)
        else:
            result = run_batch_evaluation(
                prompt_manager, reasoner, feat_df, lab_df, cols, n_samples=args.n_samples,
                n_shot=args.n_shot, source=args.source, reasoning_method=args.reasoning,
                random_state=args.seed, llm_seed=args.llm_seed, use_stratified=args.stratified,
                stratify_by=args.stratify_by, collect_prompts=args.save_prompts, verbose=args.verbose
            )
            exp_prefix = build_experiment_prefix(args.n_shot, args.source, seed=args.seed)
            
        if result:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{exp_prefix}_{model_name}_{args.reasoning}_{args.llm_seed}_{timestamp}"
            base_filepath = os.path.join(args.output_dir, base)
            export_comprehensive_report(result, base_filepath)
            print(f"\n💾 Results saved with model comparison name: {exp_prefix}")
            if args.save_prompts and 'prompts' in result and 'metadata' in result:
                # Save folder is just prefix (no model/reasoning/llm_seed/timestamp)
                save_prompts_to_disk(result['prompts'], result['metadata'], exp_prefix, args.seed, args.prompts_dir)
    
    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
