"""
Add stress predictions and true labels to existing predictions CSV.

This script:
1. Loads checkpoint JSON to extract stress predictions from LLM
2. Loads aggregated_ces.csv to get true stress labels
3. Merges with existing predictions CSV to add y_stress_pred and y_stress_real columns
"""

import json
import re
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_stress_from_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    """
    Parse stress predictions from checkpoint JSON.
    
    Returns:
        DataFrame with columns: uid, date, y_stress_pred
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    predictions = checkpoint.get('predictions', [])
    print(f"  Found {len(predictions)} predictions in checkpoint")
    
    stress_data = []
    for pred in predictions:
        uid = pred['user_id']
        date = pred['ema_date']
        
        # Extract stress prediction from LLM output
        prediction_obj = pred.get('prediction', {})
        prediction_dict = prediction_obj.get('Prediction', {})
        
        # Try to get Stress_binary first (if exists)
        stress_pred = prediction_dict.get('Stress_binary', None)
        
        # If Stress_binary doesn't exist, try to parse from "Stress" text
        if stress_pred is None:
            stress_text = prediction_dict.get('Stress', '')
            if 'High Risk' in stress_text:
                stress_pred = 1
            elif 'Low Risk' in stress_text:
                stress_pred = 0
            else:
                print(f"  Warning: Could not parse stress for {uid} on {date}, defaulting to 0")
                stress_pred = 0
        
        stress_data.append({
            'uid': uid,
            'date': date,
            'y_stress_pred': stress_pred
        })
    
    df = pd.DataFrame(stress_data)
    print(f"  Parsed stress predictions for {len(df)} samples")
    return df


def load_true_stress_labels(aggregated_ces_path: str) -> pd.DataFrame:
    """
    Load true stress labels from aggregated_ces.csv.
    
    Returns:
        DataFrame with columns: uid, date, y_stress_real
    """
    print(f"\nLoading true stress labels: {aggregated_ces_path}")
    df = pd.read_csv(aggregated_ces_path)
    
    # Convert date column to datetime and then to string for matching
    df['date'] = pd.to_datetime(df['date'])
    
    # Select only uid, date, and stress columns
    stress_df = df[['uid', 'date', 'stress']].copy()
    stress_df.rename(columns={'stress': 'y_stress_real'}, inplace=True)
    
    # Convert date to string format matching predictions CSV
    stress_df['date'] = stress_df['date'].dt.strftime('%Y-%m-%d 00:00:00')
    
    print(f"  Loaded {len(stress_df)} samples with stress labels")
    return stress_df


def calculate_and_print_metrics(df: pd.DataFrame):
    """
    Calculate and print classification metrics for anxiety, depression, and stress.
    
    Args:
        df: DataFrame with columns *_real and *_pred
    """
    print(f"\n{'='*80}")
    print("CLASSIFICATION METRICS (Macro-averaged)")
    print(f"{'='*80}")
    
    # Define targets
    targets = [
        ('Anxiety', 'y_anx_real', 'y_anx_pred'),
        ('Depression', 'y_dep_real', 'y_dep_pred'),
        ('Stress', 'y_stress_real', 'y_stress_pred'),
    ]
    
    results = []
    
    for target_name, real_col, pred_col in targets:
        # Skip if columns don't exist
        if real_col not in df.columns or pred_col not in df.columns:
            continue
        
        # Remove NaN values
        valid_mask = df[real_col].notna() & df[pred_col].notna()
        y_true = df.loc[valid_mask, real_col].astype(int)
        y_pred = df.loc[valid_mask, pred_col].astype(int)
        
        if len(y_true) == 0:
            print(f"\n⚠️  {target_name}: No valid predictions")
            continue
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results.append({
            'Target': target_name,
            'Accuracy': accuracy,
            'F1 (Macro)': f1,
            'Precision (Macro)': precision,
            'Recall (Macro)': recall,
            'N': len(y_true)
        })
        
        print(f"\n📊 {target_name.upper()}")
        print(f"  Accuracy:         {accuracy:.4f}")
        print(f"  F1 Score (Macro):  {f1:.4f}")
        print(f"  Precision (Macro): {precision:.4f}")
        print(f"  Recall (Macro):    {recall:.4f}")
        print(f"  Valid Samples:     {len(y_true)}")
    
    # Print summary table
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        results_df = pd.DataFrame(results)
        # Reorder columns for printing: Accuracy, F1, Precision, Recall
        desired_cols = ['Target', 'Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'N']
        existing = [c for c in desired_cols if c in results_df.columns]
        results_df = results_df[existing]
        print(results_df.to_string(index=False))
        print(f"{'='*80}\n")
    
    return results


def add_stress_to_predictions(predictions_csv: str, checkpoint_json: str, 
                                aggregated_ces_csv: str, output_csv: str = None):
    """
    Add stress predictions and true labels to existing predictions CSV.
    
    Args:
        predictions_csv: Path to existing predictions CSV (with anx/dep only)
        checkpoint_json: Path to checkpoint JSON (with stress predictions)
        aggregated_ces_csv: Path to aggregated_ces.csv (with true stress labels)
        output_csv: Output path (default: add _with_stress suffix)
    """
    # Load existing predictions
    print(f"\nLoading existing predictions: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)
    print(f"  Loaded {len(pred_df)} predictions")
    print(f"  Columns: {list(pred_df.columns)}")
    
    # Parse stress predictions from checkpoint
    stress_pred_df = parse_stress_from_checkpoint(checkpoint_json)
    
    # Load true stress labels
    stress_true_df = load_true_stress_labels(aggregated_ces_csv)
    
    # Merge stress predictions
    print(f"\nMerging stress predictions...")
    merged_df = pred_df.merge(
        stress_pred_df,
        on=['uid', 'date'],
        how='left'
    )
    
    # Merge true stress labels
    print(f"Merging true stress labels...")
    merged_df = merged_df.merge(
        stress_true_df,
        on=['uid', 'date'],
        how='left'
    )
    
    # Check for missing values
    missing_pred = merged_df['y_stress_pred'].isna().sum()
    missing_true = merged_df['y_stress_real'].isna().sum()
    
    if missing_pred > 0:
        print(f"  Warning: {missing_pred} samples missing stress predictions")
    if missing_true > 0:
        print(f"  Warning: {missing_true} samples missing true stress labels")
    
    # Reorder columns: uid, date, anxiety, depression, stress
    column_order = [
        'uid', 'date',
        'y_anx_real', 'y_dep_real', 'y_stress_real',
        'y_anx_pred', 'y_dep_pred', 'y_stress_pred'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[column_order]
    
    # Determine output path
    if output_csv is None:
        # Add _with_stress suffix before .csv
        pred_path = Path(predictions_csv)
        output_csv = pred_path.parent / f"{pred_path.stem}_with_stress.csv"
    
    # Save
    merged_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved updated predictions with stress: {output_csv}")
    print(f"   Total samples: {len(merged_df)}")
    print(f"   Columns: {list(merged_df.columns)}")
    
    # Print sample
    print(f"\nSample (first 5 rows):")
    print(merged_df.head())
    
    # Calculate and print metrics
    calculate_and_print_metrics(merged_df)
    
    return merged_df


def _extract_base_name(path: str) -> str:
    """
    Extract a common base name shared between predictions and checkpoint files.
    Assumes file stems look like:
      - ces_<exp>_YYYYMMDD_HHMMSS_predictions
      - ces_<exp>_YYYYMMDD_HHMMSS_checkpoint_###
    Returns the part before the last "_20" (year start) token.
    """
    stem = Path(path).stem
    idx = stem.rfind('_20')
    return stem[:idx] if idx != -1 else stem


def process_all_ces(aggregated_ces_csv: str,
                    results_dir: str = 'results',
                    checkpoints_dir: str = 'results/chk'):
    """
    Batch-process all prediction CSVs and checkpoint JSONs starting with 'ces',
    matched by a shared base name.
    Prints per-run metrics and a summary table for Stress.
    """
    script_dir = Path(__file__).parent
    results_path = Path(results_dir)
    chk_path = Path(checkpoints_dir)
    # Resolve relative dirs against script location to be robust to CWD
    if not results_path.is_absolute():
        results_path = script_dir / results_path
    if not chk_path.is_absolute():
        chk_path = script_dir / chk_path

    pred_files = list(results_path.glob('ces*_predictions.csv'))
    chk_files = list(chk_path.glob('ces*_checkpoint_*.json'))

    if not pred_files:
        print("No predictions found matching ces*_predictions.csv")
    if not chk_files:
        print("No checkpoints found matching ces*_checkpoint_*.json")

    # Build maps by base name
    preds_by_base = {}
    print(f"Found {len(pred_files)} prediction file(s)")
    for p in pred_files:
        base = _extract_base_name(p.name)
        preds_by_base[base] = p
        print(f"  PRED: base={base} file={p}")

    chks_by_base = {}
    print(f"Found {len(chk_files)} checkpoint file(s)")
    for c in chk_files:
        base = _extract_base_name(c.name)
        chks_by_base[base] = c
        print(f"  CHK:  base={base} file={c}")

    common_bases = sorted(set(preds_by_base.keys()) & set(chks_by_base.keys()))

    if not common_bases:
        print("No matching base names between predictions and checkpoints.")
        # Print helpful diagnostics
        only_preds = sorted(set(preds_by_base.keys()) - set(chks_by_base.keys()))
        only_chks = sorted(set(chks_by_base.keys()) - set(preds_by_base.keys()))
        if only_preds:
            print("Bases with predictions but no checkpoint:")
            for b in only_preds:
                print(f"  - {b} -> {preds_by_base[b]}")
        if only_chks:
            print("Bases with checkpoint but no predictions:")
            for b in only_chks:
                print(f"  - {b} -> {chks_by_base[b]}")
        return

    print(f"\nFound {len(common_bases)} matching experiment(s):")
    for b in common_bases:
        print(f"  - {b}")

    # Run all and collect Stress metrics
    summary_rows = []
    for base in common_bases:
        pred_csv = preds_by_base[base]
        chk_json = chks_by_base[base]

        print("\n" + "-"*80)
        print(f"Processing base: {base}")
        print(f"  predictions: {pred_csv}\n  checkpoint:  {chk_json}")

        merged = add_stress_to_predictions(
            predictions_csv=str(pred_csv),
            checkpoint_json=str(chk_json),
            aggregated_ces_csv=aggregated_ces_csv,
            output_csv=None,
        )

        metrics = calculate_and_print_metrics(merged)
        # Extract Stress only if present
        stress_rows = [m for m in metrics if m.get('Target') == 'Stress'] if metrics else []
        if stress_rows:
            m = stress_rows[0]
            summary_rows.append({
                'Base': base,
                'Accuracy': m['Accuracy'],
                'F1 (Macro)': m['F1 (Macro)'],
                'Precision (Macro)': m['Precision (Macro)'],
                'Recall (Macro)': m['Recall (Macro)'],
                'N': m['N'],
            })

    if summary_rows:
        print(f"\n{'='*80}")
        print("BATCH SUMMARY (Stress)")
        print(f"{'='*80}")
        df = pd.DataFrame(summary_rows)
        # Sort by F1 desc for quick inspection
        df = df.sort_values(by=['F1 (Macro)'], ascending=False)
        # Reorder columns for printing
        desired_cols = ['Base', 'Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'N']
        existing = [c for c in desired_cols if c in df.columns]
        df = df[existing]
        print(df.to_string(index=False))
        print(f"{'='*80}")


if __name__ == '__main__':
    # Default paths
    predictions_csv = 'results/ces_compass_4shot_crossrandom_cot_seed42_claude_4_5_sonnet_cot_42_20251029_112600_predictions.csv'
    checkpoint_json = 'results/chk/ces_compass_4shot_crossrandom_cot_seed42_claude_4_5_sonnet_cot_42_20251029_085428_checkpoint_300.json'
    aggregated_ces_csv = '../dataset/CES/aggregated_ces.csv'

    args = sys.argv[1:]
    all_mode = '--all-ces' in args

    # Optional overrides
    if '--aggregated' in args:
        idx = args.index('--aggregated')
        if idx + 1 < len(args):
            aggregated_ces_csv = args[idx + 1]

    if all_mode:
        print("="*80)
        print("ADD STRESS TO PREDICTIONS (BATCH MODE: ces*)")
        print("="*80)
        # Optional directory overrides
        results_dir = 'results'
        checkpoints_dir = 'results/chk'
        if '--results-dir' in args:
            i = args.index('--results-dir')
            if i + 1 < len(args):
                results_dir = args[i + 1]
        if '--checkpoints-dir' in args:
            i = args.index('--checkpoints-dir')
            if i + 1 < len(args):
                checkpoints_dir = args[i + 1]
        process_all_ces(aggregated_ces_csv=aggregated_ces_csv,
                        results_dir=results_dir,
                        checkpoints_dir=checkpoints_dir)
    else:
        # Positional single-run mode (backwards compatible)
        if len(args) >= 1 and not args[0].startswith('--'):
            predictions_csv = args[0]
        if len(args) >= 2 and not args[1].startswith('--'):
            checkpoint_json = args[1]
        if len(args) >= 3 and not args[2].startswith('--'):
            aggregated_ces_csv = args[2]
        output_csv = None
        if len(args) >= 4 and not args[3].startswith('--'):
            output_csv = args[3]

        print("="*80)
        print("ADD STRESS TO PREDICTIONS")
        print("="*80)

        add_stress_to_predictions(
            predictions_csv=predictions_csv,
            checkpoint_json=checkpoint_json,
            aggregated_ces_csv=aggregated_ces_csv,
            output_csv=output_csv
        )

