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

from src.data_utils import load_dataset_testset
from src.reasoning import LLMReasoner
from src.prompt_manager import PromptManager
from src.prompt_utils import (
    NumpyEncoder,
    save_prompts_to_disk,
    load_prompts_from_disk,
    build_experiment_prefix,
)
from src.evaluation_runner import run_batch_evaluation, run_batch_with_loaded_prompts, run_batch_prompts_only
from src.performance import export_comprehensive_report
from src import config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sensor LLM-based Mental Health Prediction Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core experimental parameters
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of test samples for evaluation (default: use all samples in testset)')
    parser.add_argument('--n_shot', type=int, default=config.DEFAULT_N_SHOT,
                       help='Number of ICL examples')
    parser.add_argument('--strategy', type=str, default='cross_random',
                       choices=['cross_random', 'cross_retrieval', 'personal_recent', 'hybrid', 'none'],
                       help='ICL strategy: cross_random (random from others), cross_retrieval (DTW from others), personal_recent (recent from self), hybrid (mix), none (zero-shot)')
    parser.add_argument('--use-dtw', action='store_true',
                       help='For hybrid: use DTW for cross-user part (default: random)')
    parser.add_argument('--reasoning', type=str, default=config.DEFAULT_REASONING_METHOD,
                       choices=['direct', 'cot', 'self_feedback'],
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
    if args.n_shot > 0 and args.strategy == 'none':
        print("Warning: n_shot > 0 but strategy is 'none'. Will use zero-shot.")
    
    if args.save_prompts_only:
        if args.load_prompts:
            raise ValueError("Cannot use --save-prompts-only with --load-prompts")
        # Automatically enable save-prompts when save-prompts-only is set
        args.save_prompts = True
    
    # Print configuration
    print("\n" + "="*60)
    print("LLM MENTAL HEALTH PREDICTION EVALUATION")
    print("="*60)
    print(f"  Model: {args.model}")
    if not args.load_prompts:
        print(f"  ICL Strategy: {args.strategy} | N-Shot: {args.n_shot}")
        print(f"  Reasoning: {args.reasoning}")
        print(f"  Samples: {args.n_samples} | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print(f"\n  Dataset: {config.DATASET_TYPE.upper()}")
        print(f"     Features: {config.CURRENT_DATASET_CONFIG['name']}")
        print(f"     Similarity: {config.CURRENT_DATASET_CONFIG['similarity_method']}")
    if args.seed:
        print(f"  Random Seed: {args.seed}")
    if args.load_prompts:
        print(f"  Loading Prompts: {args.load_prompts}")
    print("="*60)
    
    # Optimization: Skip data/prompt loading when using --load-prompts
    if args.load_prompts:
        print("\n[Model Comparison Mode]")
        print("  Skipping data loading (not needed)")
        print("  Skipping prompt manager (prompts pre-generated)")
        feat_df, lab_df, cols, prompt_manager = None, None, None, None
        initial_timings = {'loading': 0.0, 'test_sampling': 0.0}
    else:
        import time
        initial_timings = {}
        
        # Universal dataset loading
        print(f"\n[Loading {config.DATASET_TYPE.upper()} dataset...]")
        t0_total = time.time()
        
        feat_df, lab_df, test_df, train_df, cols = load_dataset_testset(config.DATASET_TYPE)
        
        # If n_samples not specified, use all samples in testset
        if args.n_samples is None:
            args.n_samples = len(test_df)
            print(f"  Using all {args.n_samples} samples from testset")
        
        total_time = time.time() - t0_total
        initial_timings['loading'] = total_time * 0.3
        initial_timings['test_sampling'] = total_time * 0.7
        
        print(f"  ✓ Features: {feat_df.shape[0]} rows")
        print(f"  ✓ Test set: {test_df.shape[0]} samples")
        print(f"  ✓ Train set: {train_df.shape[0]} samples (for ICL)")
        print(f"  ✓ Labels: {cols['labels']}")
        
        # Store test/train/full dataframes in config for ICL selection
        # full_lab_df (train+test) needed for personal_recent strategy
        if config.DATASET_TYPE == 'globem':
            config.GLOBEM_TEST_DF = test_df
            config.GLOBEM_TRAIN_DF = train_df
            config.GLOBEM_FULL_LAB_DF = lab_df  # For personal_recent ICL
        elif config.DATASET_TYPE == 'ces':
            config.CES_TEST_DF = test_df
            config.CES_TRAIN_DF = train_df
            config.CES_FULL_LAB_DF = lab_df  # For personal_recent ICL
        elif config.DATASET_TYPE == 'mentaliot':
            config.MENTALIOT_TEST_DF = test_df
            config.MENTALIOT_TRAIN_DF = train_df
            config.MENTALIOT_FULL_LAB_DF = lab_df  # For personal_recent ICL
        
        print("\n[Initializing Prompt Manager...]")
        prompt_manager = PromptManager()
        print("  YAML templates loaded")
    
    # Initialize LLM reasoner (skip if only saving prompts)
    if args.save_prompts_only:
        print(f"\n[Prompt-only mode: Skipping LLM initialization]")
        reasoner = None
    else:
        print(f"\n[Initializing LLM Reasoner ({args.model})...]")
        reasoner = LLMReasoner(model=args.model)
        print("  Ready")
    
    model_name = args.model.replace('/', '_').replace('-', '_').replace('.', '_')
    # Common prefix per spec - include selection and beta
    # Build experiment prefix with new strategy naming
    exp_prefix = build_experiment_prefix(args.n_shot, args.strategy, 
                                        reasoning=args.reasoning, 
                                        seed=args.seed)
    # Run evaluation (batch mode)
    if args.load_prompts:
        print(f"\n[Loading prompts from: {args.load_prompts}]")
        prompts, labels, metadata = load_prompts_from_disk(args.load_prompts)
        
        # If n_samples not specified, use the number of prompts loaded
        if args.n_samples is None:
            args.n_samples = metadata.get('num_samples', len(prompts))
            print(f"  Using all {args.n_samples} samples from saved prompts")
        
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
            prompt_manager, feat_df, test_df, cols,
            n_samples=args.n_samples, n_shot=args.n_shot,
            strategy=args.strategy, use_dtw=args.use_dtw,
            reasoning_method=args.reasoning, random_state=args.seed,
            verbose=args.verbose,
            initial_timings=initial_timings,
            dataset=config.DATASET_TYPE
        )
        exp_prefix = build_experiment_prefix(args.n_shot, args.strategy,
                                            reasoning=args.reasoning,
                                            seed=args.seed)
    else:
        result = run_batch_evaluation(
            prompt_manager, reasoner, feat_df, test_df, cols, 
            n_samples=args.n_samples, n_shot=args.n_shot, 
            strategy=args.strategy, use_dtw=args.use_dtw,
            reasoning_method=args.reasoning, random_state=args.seed, 
            llm_seed=args.llm_seed,
            collect_prompts=args.save_prompts, verbose=args.verbose,
            initial_timings=initial_timings,
            dataset=config.DATASET_TYPE
        )
        exp_prefix = build_experiment_prefix(args.n_shot, args.strategy,
                                            reasoning=args.reasoning,
                                            seed=args.seed)
        
    if result:
        if args.save_prompts_only:
            # Only save prompts, skip result export
            if 'prompts' in result and 'metadata' in result:
                step_timings = result.get('step_timings', None)
                save_prompts_to_disk(result['prompts'], result['metadata'], exp_prefix, args.seed, 
                                    args.prompts_dir, step_timings)
                print(f"\n[Prompts saved: {exp_prefix} ({result['n_samples']} samples)]")
        else:
            # Normal batch mode: export results
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{exp_prefix}_{model_name}_{args.reasoning}_{args.llm_seed}_{timestamp}"
            base_filepath = os.path.join(args.output_dir, base)
            predictions = result.get('predictions', None)
            export_comprehensive_report(result, base_filepath, predictions=predictions)
            print(f"\nResults saved with model comparison name: {exp_prefix}")
            if args.save_prompts and 'prompts' in result and 'metadata' in result:
                # Save folder is just prefix (no model/reasoning/llm_seed/timestamp)
                step_timings = result.get('step_timings', None)
                save_prompts_to_disk(result['prompts'], result['metadata'], exp_prefix, args.seed, 
                                    args.prompts_dir, step_timings)
    
    print("\nEvaluation complete!\n")


if __name__ == "__main__":
    main()
