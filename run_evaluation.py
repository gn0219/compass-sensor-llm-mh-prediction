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

from src.sensor_transformation import load_globem_data
from src.data_utils import (
    sample_multiinstitution_testset, filter_testset_by_historical_labels,
    sample_ces_testset
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
    parser.add_argument('--strategy', type=str, default='cross_random',
                       choices=['cross_random', 'cross_retrieval', 'personal_recent', 'hybrid_blend', 'none'],
                       help='ICL strategy: cross_random (random from others), cross_retrieval (DTW from others), personal_recent (recent from self), hybrid_blend (mix), none (zero-shot)')
    parser.add_argument('--use-dtw', action='store_true',
                       help='For hybrid_blend: use DTW for cross-user part (default: random)')
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
        if args.mode == 'single':
            raise ValueError("--save-prompts-only is only available for batch mode")
        if args.load_prompts:
            raise ValueError("Cannot use --save-prompts-only with --load-prompts")
        # Automatically enable save-prompts when save-prompts-only is set
        args.save_prompts = True
    
    # Print configuration
    print("\n" + "="*60)
    print("LLM MENTAL HEALTH PREDICTION EVALUATION")
    print("="*60)
    print(f"  Mode: {args.mode} | Model: {args.model}")
    if not args.load_prompts:
        print(f"  ICL Strategy: {args.strategy} | N-Shot: {args.n_shot}")
        print(f"  Reasoning: {args.reasoning}")
        if args.mode == 'batch':
            print(f"  Samples: {args.n_samples} | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print(f"\n  Data Config (see src/config.py to change):")
        print(f"     Format: {config.DEFAULT_TARGET}")
        print(f"     Test Filter: {config.FILTER_TESTSET_BY_HISTORY} (min {config.MIN_HISTORICAL_LABELS} labels)")
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
        
        # Load dataset based on config.DATASET_TYPE
        if config.DATASET_TYPE == 'ces':
            print("\n[Loading CES dataset...]")
            t0_total = time.time()
            feat_df, test_df, train_df, cols = sample_ces_testset(
                n_users=config.CES_N_USERS,
                min_ema_per_user=config.CES_MIN_EMA_PER_USER,
                samples_per_user=config.CES_SAMPLES_PER_USER,
                random_state=args.seed
            )
            # For CES, we use test_df as lab_df (testset)
            # train_df is used for ICL examples - store it globally
            lab_df = test_df
            # Store train_df for ICL example selection
            # We'll pass it via USE_MULTI_INSTITUTION_TESTSET flag and store in config
            config.CES_TRAIN_DF = train_df
            total_time = time.time() - t0_total
            initial_timings['loading'] = total_time * 0.3
            initial_timings['test_sampling'] = total_time * 0.7
        elif config.DATASET_TYPE == 'mentaliot':
            print("\n[Loading MentalIoT dataset...]")
            t0_total = time.time()
            from src.data_utils import sample_mentaliot_testset
            feat_df, lab_df, test_df, train_df, cols = sample_mentaliot_testset(
                n_samples_per_user=10,  # 10 samples per user
                random_state=args.seed
            )
            # Store test_df and train_df for ICL examples
            config.MENTALIOT_TEST_DF = test_df
            config.MENTALIOT_TRAIN_DF = train_df
            # Use test_df as lab_df for prediction
            lab_df = test_df
            total_time = time.time() - t0_total
            initial_timings['loading'] = total_time * 0.3
            initial_timings['test_sampling'] = total_time * 0.7
        elif config.DATASET_TYPE == 'globem':
            print("\n[Loading GLOBEM dataset...]")
            
            # Check if multi-institution testset mode is enabled
            if config.USE_MULTI_INSTITUTION_TESTSET:
                t0_total = time.time()
                feat_df, lab_df, cols = sample_multiinstitution_testset(
                    institutions_config=config.MULTI_INSTITUTION_CONFIG,
                    min_ema_per_user=config.MIN_EMA_PER_USER,
                    samples_per_user=config.SAMPLES_PER_USER,
                    random_state=args.seed,
                    target=config.DEFAULT_TARGET
                )
                total_time = time.time() - t0_total
                # For multi-institution, loading and sampling happen together
                # Approximate: 30% loading, 70% sampling
                initial_timings['loading'] = total_time * 0.3
                initial_timings['test_sampling'] = total_time * 0.7
            else:
                t0_load = time.time()
                feat_df, lab_df, cols = load_globem_data(institution=config.DEFAULT_INSTITUTION, target=config.DEFAULT_TARGET)
                initial_timings['loading'] = time.time() - t0_load
                
                print(f"  Target: {config.DEFAULT_TARGET}")
                print(f"  Features: {feat_df.shape[0]} rows | Labels: {lab_df.shape[0]} rows")
                
                # Filter test set by historical labels (for fair ICL comparison)
                t0_sampling = time.time()
                if config.FILTER_TESTSET_BY_HISTORY:
                    lab_df = filter_testset_by_historical_labels(
                        lab_df, cols, min_historical=config.MIN_HISTORICAL_LABELS
                    )
                initial_timings['test_sampling'] = time.time() - t0_sampling
        else:
            raise ValueError(f"Unknown DATASET_TYPE: {config.DATASET_TYPE}. Must be 'globem' or 'ces'")
        
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
    # Run evaluation
    if args.mode == 'single':
        result = run_single_prediction(
            prompt_manager, reasoner, feat_df, lab_df, cols, 
            n_shot=args.n_shot, strategy=args.strategy, use_dtw=args.use_dtw,
            reasoning_method=args.reasoning, random_state=args.seed, 
            llm_seed=args.llm_seed, verbose=args.verbose,
            dataset=config.DATASET_TYPE
        )
        
        if result:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # dataset_sensor_nshot_source_seed_Model_reasoning_llmseed_timestamp.json
            filename = f"{exp_prefix}_{model_name}_{args.reasoning}_{args.llm_seed}_{timestamp}.json"
            result_file = os.path.join(args.output_dir, filename)
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            
            print(f"\n[Result saved to: {result_file}]")
    
    elif args.mode == 'batch':
        if args.load_prompts:
            print(f"\n[Loading prompts from: {args.load_prompts}]")
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
                strategy=args.strategy, use_dtw=args.use_dtw,
                reasoning_method=args.reasoning, random_state=args.seed,
                verbose=args.verbose,
                use_all_samples=config.USE_MULTI_INSTITUTION_TESTSET,
                initial_timings=initial_timings,
                dataset=config.DATASET_TYPE
            )
            exp_prefix = build_experiment_prefix(args.n_shot, args.strategy,
                                                reasoning=args.reasoning,
                                                seed=args.seed)
        else:
            result = run_batch_evaluation(
                prompt_manager, reasoner, feat_df, lab_df, cols, 
                n_samples=args.n_samples, n_shot=args.n_shot, 
                strategy=args.strategy, use_dtw=args.use_dtw,
                reasoning_method=args.reasoning, random_state=args.seed, 
                llm_seed=args.llm_seed,
                collect_prompts=args.save_prompts, verbose=args.verbose,
                use_all_samples=config.USE_MULTI_INSTITUTION_TESTSET,
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
