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

from src.sensor_transformation import (
    load_globem_data, 
    get_data_statistics, 
    print_data_statistics,
    filter_testset_by_historical_labels
)
from src.reasoning import LLMReasoner
from src.prompt_manager import PromptManager
from src.prompt_utils import (
    NumpyEncoder,
    save_prompts_to_disk,
    load_prompts_from_disk,
    build_experiment_prefix,
)
from src.evaluation_runner import run_single_prediction, run_batch_evaluation, run_batch_with_loaded_prompts, run_batch_prompts_only
from src.performance import export_comprehensive_report
from src import config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sensor LLM-based Mental Health Prediction Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core experimental parameters
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                       help='Evaluation mode')
    parser.add_argument('--n_samples', type=int, default=config.DEFAULT_BATCH_SIZE,
                       help='Number of test samples (batch mode only)')
    parser.add_argument('--n_shot', type=int, default=config.DEFAULT_N_SHOT,
                       help='Number of ICL examples')
    parser.add_argument('--source', type=str, default=config.DEFAULT_ICL_SOURCE,
                       choices=['personalized', 'generalized', 'hybrid'],
                       help='ICL source strategy')
    parser.add_argument('--selection', type=str, default=config.DEFAULT_SELECTION_METHOD,
                       choices=['random', 'similarity', 'temporal', 'diversity'],
                       help='ICL selection method')
    parser.add_argument('--beta', type=float, default=0.0,
                       help='Label balance penalty for diversity selection (0.0=no penalty, 0.1-0.3=recommended)')
    parser.add_argument('--reasoning', type=str, default=config.DEFAULT_REASONING_METHOD,
                       choices=['direct', 'cot', 'tot', 'sc'],
                       help='LLM reasoning method')
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL,
                       choices=config.SUPPORTED_MODELS,
                       help='LLM model to use')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--llm_seed', type=int, default=None,
                       help='LLM sampling seed')
    
    # I/O and utilities
    parser.add_argument('--output_dir', type=str, default=config.DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--save-prompts', action='store_true',
                       help='Save prompts to disk for later reuse')
    parser.add_argument('--save-prompts-only', action='store_true',
                       help='Generate and save prompts only (skip LLM calls) - useful for large batches')
    parser.add_argument('--load-prompts', type=str, default=None, metavar='EXP_NAME',
                       help='Load pre-saved prompts from experiment name')
    parser.add_argument('--prompts-dir', type=str, default='./saved_prompts',
                       help='Directory for saved prompts')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Save checkpoint every N samples (0 to disable)')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.selection == 'diversity' and args.beta < 0:
        raise ValueError(f"Beta must be >= 0 for diversity selection, got {args.beta}")
    
    if args.save_prompts_only:
        if args.mode == 'single':
            raise ValueError("--save-prompts-only is only available for batch mode")
        if args.load_prompts:
            raise ValueError("Cannot use --save-prompts-only with --load-prompts")
        # Automatically enable save-prompts when save-prompts-only is set
        args.save_prompts = True
    
    # Print configuration
    print("\n" + "="*60)
    print("🚀 LLM MENTAL HEALTH PREDICTION EVALUATION")
    print("="*60)
    print(f"  Mode: {args.mode} | Model: {args.model}")
    if not args.load_prompts:
        print(f"  ICL: {args.source} | Selection: {args.selection} | N-Shot: {args.n_shot}")
        print(f"  Reasoning: {args.reasoning}")
        if args.mode == 'batch':
            print(f"  Samples: {args.n_samples} | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print(f"\n  📊 Data Config (see src/config.py to change):")
        print(f"     Window: {config.AGGREGATION_WINDOW_DAYS} days | Mode: {config.DEFAULT_AGGREGATION_MODE}")
        print(f"     Adaptive: {config.USE_ADAPTIVE_WINDOW} | Immediate: {config.USE_IMMEDIATE_WINDOW}")
        print(f"     Test Filter: {config.FILTER_TESTSET_BY_HISTORY} (min {config.MIN_HISTORICAL_LABELS} labels)")
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
        
        # Filter test set by historical labels (for fair ICL comparison)
        if config.FILTER_TESTSET_BY_HISTORY:
            lab_df = filter_testset_by_historical_labels(
                lab_df, cols, min_historical=config.MIN_HISTORICAL_LABELS
            )
        else:
            print(f"\n⚠️  Warning: Test set filtering disabled - personalized ICL may fail")
        
        print("\n🎨 Initializing Prompt Manager...")
        prompt_manager = PromptManager()
        print("  ✓ YAML templates loaded")
    
    # Initialize LLM reasoner (skip if only saving prompts)
    if args.save_prompts_only:
        print(f"\n⚡ Prompt-only mode: Skipping LLM initialization")
        reasoner = None
    else:
        print(f"\n🤖 Initializing LLM Reasoner ({args.model})...")
        reasoner = LLMReasoner(model=args.model)
        print("  ✓ Ready")
    
    model_name = args.model.replace('/', '_').replace('-', '_').replace('.', '_')
    # Common prefix per spec - include selection and beta
    # Always pass beta for diversity, even if 0.0 (to distinguish diversity00, diversity01, etc.)
    exp_prefix = build_experiment_prefix(args.n_shot, args.source, selection=args.selection, 
                                        beta=args.beta if args.selection == 'diversity' else None, 
                                        seed=args.seed)
    # Run evaluation
    if args.mode == 'single':
        result = run_single_prediction(
            prompt_manager, reasoner, feat_df, lab_df, cols, 
            n_shot=args.n_shot, source=args.source, selection=args.selection, 
            reasoning_method=args.reasoning, random_state=args.seed, 
            llm_seed=args.llm_seed, beta=args.beta, verbose=args.verbose
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
            
            # Use provided folder (should be prefix) as exp_prefix
            exp_prefix = os.path.basename(args.load_prompts)
            
            # Setup checkpoint path
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{exp_prefix}_{model_name}_{args.reasoning}_{args.llm_seed}_{timestamp}"
            checkpoint_base = os.path.join(args.output_dir, base)
            
            # labels contains the sample metadata (user_id, dates, true labels)
            result = run_batch_with_loaded_prompts(
                reasoner, prompts, labels, reasoning_method=args.reasoning,
                llm_seed=args.llm_seed, 
                checkpoint_path=checkpoint_base if args.checkpoint_every > 0 else None,
                checkpoint_every=args.checkpoint_every,
                resume_from=args.resume_from,
                verbose=args.verbose
            )
        elif args.save_prompts_only:
            # Generate and save prompts only (no LLM calls)
            result = run_batch_prompts_only(
                prompt_manager, feat_df, lab_df, cols,
                n_samples=args.n_samples, n_shot=args.n_shot,
                source=args.source, selection=args.selection,
                reasoning_method=args.reasoning, random_state=args.seed,
                beta=args.beta, verbose=args.verbose
            )
            exp_prefix = build_experiment_prefix(args.n_shot, args.source, selection=args.selection,
                                                beta=args.beta if args.selection == 'diversity' else None,
                                                seed=args.seed)
        else:
            result = run_batch_evaluation(
                prompt_manager, reasoner, feat_df, lab_df, cols, 
                n_samples=args.n_samples, n_shot=args.n_shot, 
                source=args.source, selection=args.selection,
                reasoning_method=args.reasoning, random_state=args.seed, 
                llm_seed=args.llm_seed, beta=args.beta,
                collect_prompts=args.save_prompts, verbose=args.verbose
            )
            exp_prefix = build_experiment_prefix(args.n_shot, args.source, selection=args.selection,
                                                beta=args.beta if args.selection == 'diversity' else None,
                                                seed=args.seed)
            
        if result:
            if args.save_prompts_only:
                # Only save prompts, skip result export
                if 'prompts' in result and 'metadata' in result:
                    save_prompts_to_disk(result['prompts'], result['metadata'], exp_prefix, args.seed, args.prompts_dir)
                    print(f"\n💾 Prompts saved: {exp_prefix} ({result['n_samples']} samples)")
            else:
                # Normal batch mode: export results
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
