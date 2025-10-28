"""
Performance Evaluation Module

Handles classification metrics (accuracy, F1, AUROC) and efficiency metrics (latency, tokens).
"""

import warnings
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from .utils import NumpyEncoder

# Suppress sklearn warnings for single-class confusion matrix
warnings.filterwarnings('ignore', message='.*single label.*', category=UserWarning)


def calculate_binary_metrics(y_true: List[int], y_pred: List[int], 
                             y_proba: Optional[List[float]] = None, label_name: str = "") -> Dict:
    """Calculate classification metrics for binary prediction."""
    if len(y_true) == 0:
        return {'accuracy': 0.0, 'balanced_accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 
                'auroc': None, 'confusion_matrix': [[0, 0], [0, 0]], 
                'support': {'class_0': 0, 'class_1': 0},
                'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Explicitly specify labels to avoid warnings
    
    # Extract TP, TN, FP, FN from confusion matrix
    # cm is [[TN, FP], [FN, TP]] for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present in predictions
        if y_true[0] == y_pred[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        # Handle edge case where only one class is predicted
        tn, fp, fn, tp = 0, 0, 0, 0
        if 0 in y_pred and 0 in y_true:
            tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        if 1 in y_true and 0 in y_pred:
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        if 0 in y_true and 1 in y_pred:
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        if 1 in y_pred and 1 in y_true:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    
    auroc = None
    if y_proba is not None and len(set(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_proba)
        except Exception as e:
            print(f"Warning: Could not calculate AUROC for {label_name}: {e}")
    
    unique, counts = np.unique(y_true, return_counts=True)
    support = dict(zip([f'class_{int(u)}' for u in unique], counts.tolist()))
    
    return {
        'accuracy': float(accuracy), 'balanced_accuracy': float(balanced_acc),
        'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1),
        'auroc': float(auroc) if auroc else None,
        'confusion_matrix': cm.tolist(), 'support': support,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def calculate_mental_health_metrics(results: List[Dict], anxiety_label_key: str = 'phq4_anxiety_EMA',
                                    depression_label_key: str = 'phq4_depression_EMA',
                                    stress_label_key: str = 'stress') -> Dict:
    """Calculate metrics for anxiety, depression, and stress predictions."""
    # Define targets with their label keys and prediction keys
    targets = [
        ('anxiety', anxiety_label_key, 'Anxiety'),
        ('depression', depression_label_key, 'Depression'),
        ('stress', stress_label_key, 'Stress')
    ]
    
    # Collect predictions for each target
    target_data = {name: {'true': [], 'pred': [], 'proba': []} for name, _, _ in targets}
    
    for result in results:
        if 'labels' not in result or 'prediction' not in result:
            continue
        
        for target_name, label_key, pred_key in targets:
            pred_binary_key = f'{pred_key}_binary'
            
            if label_key in result['labels'] and pred_binary_key in result['prediction']:
                target_data[target_name]['true'].append(result['labels'][label_key])
                target_data[target_name]['pred'].append(result['prediction'][pred_binary_key])
                
                if 'proba' in result and pred_key in result['proba']:
                    target_data[target_name]['proba'].append(result['proba'][pred_key])
    
    # Calculate metrics for each target
    result_dict = {}
    for target_name, _, pred_key in targets:
        true_vals = target_data[target_name]['true']
        pred_vals = target_data[target_name]['pred']
        proba_vals = target_data[target_name]['proba']
        
        # Only include if we have predictions for this target
        if true_vals:
            metrics = calculate_binary_metrics(
                true_vals, pred_vals,
                proba_vals if proba_vals else None,
                pred_key
            )
            result_dict[target_name] = metrics
    
    return result_dict


def calculate_classification_metrics(y_true: List[int], y_pred: List[int], target: str = '') -> Dict:
    """Wrapper for classification metrics."""
    return calculate_binary_metrics(y_true, y_pred, None, target)


def calculate_efficiency_metrics(usage_stats: Dict) -> Dict:
    """Calculate efficiency metrics from API usage statistics with standard deviations."""
    num_requests = usage_stats.get('num_requests', 0)
    
    if num_requests == 0:
        return {
            'latency': {'avg_seconds': 0, 'std_seconds': 0, 'total_seconds': 0},
            'cost': {'total_usd': 0, 'per_sample_usd': 0, 'per_sample_std': 0},
            'throughput': {'tokens_per_second': 0, 'samples_per_minute': 0},
            'tokens': {'total': 0, 'prompt': 0, 'completion': 0, 'avg_per_request': 0, 'std_per_request': 0,
                      'prompt_avg': 0, 'prompt_std': 0, 'completion_avg': 0, 'completion_std': 0},
            'gpu': {'available': False}
        }
    
    total_latency = usage_stats.get('total_latency', 0)
    total_tokens = usage_stats.get('total_tokens', 0)
    total_cost = usage_stats.get('total_cost', 0)
    prompt_tokens = usage_stats.get('prompt_tokens', 0)
    completion_tokens = usage_stats.get('completion_tokens', 0)
    
    # Get per-request lists for standard deviation calculation
    latencies = usage_stats.get('latencies', [])
    costs = usage_stats.get('costs', [])
    total_tokens_list = usage_stats.get('total_tokens_list', [])
    prompt_tokens_list = usage_stats.get('prompt_tokens_list', [])
    completion_tokens_list = usage_stats.get('completion_tokens_list', [])
    
    # Latency metrics
    avg_latency = total_latency / num_requests
    std_latency = float(np.std(latencies)) if len(latencies) > 1 else 0.0
    
    # Throughput
    tokens_per_second = total_tokens / total_latency if total_latency > 0 else 0
    samples_per_minute = (num_requests / total_latency) * 60 if total_latency > 0 else 0
    requests_per_minute = samples_per_minute  # Same for now
    
    # Cost breakdown
    cost_per_request = total_cost / num_requests
    std_cost = float(np.std(costs)) if len(costs) > 1 else 0.0

    # Token metrics
    avg_tokens_per_request = total_tokens / num_requests
    std_tokens = float(np.std(total_tokens_list)) if len(total_tokens_list) > 1 else 0.0
    avg_prompt = float(np.mean(prompt_tokens_list)) if prompt_tokens_list else 0.0
    std_prompt = float(np.std(prompt_tokens_list)) if len(prompt_tokens_list) > 1 else 0.0
    avg_completion = float(np.mean(completion_tokens_list)) if completion_tokens_list else 0.0
    std_completion = float(np.std(completion_tokens_list)) if len(completion_tokens_list) > 1 else 0.0
    
    # GPU metrics (if available)
    gpu_memory_list = usage_stats.get('gpu_memory_used', [])
    gpu_util_list = usage_stats.get('gpu_utilization', [])
    peak_gpu_memory = usage_stats.get('peak_gpu_memory', 0.0)
    
    gpu_metrics = {'available': False}
    if gpu_memory_list and gpu_util_list:
        gpu_metrics = {
            'available': True,
            'avg_memory_mb': float(np.mean(gpu_memory_list)),
            'std_memory_mb': float(np.std(gpu_memory_list)) if len(gpu_memory_list) > 1 else 0.0,
            'peak_memory_mb': float(peak_gpu_memory),
            'avg_utilization_percent': float(np.mean(gpu_util_list)),
            'std_utilization_percent': float(np.std(gpu_util_list)) if len(gpu_util_list) > 1 else 0.0
        }
    
    return {
        'latency': {
            'avg_seconds': float(avg_latency), 
            'std_seconds': float(std_latency),
            'total_seconds': float(total_latency)
        },
        'cost': {
            'total_usd': float(total_cost), 
            'per_sample_usd': float(cost_per_request),
            'per_sample_std': float(std_cost)
        },
        'throughput': {
            'tokens_per_second': float(tokens_per_second), 
            'samples_per_minute': float(samples_per_minute)
        },
        'tokens': {
            'total': int(total_tokens), 
            'prompt': int(prompt_tokens), 
            'completion': int(completion_tokens), 
            'avg_per_request': float(avg_tokens_per_request),
            'std_per_request': float(std_tokens),
            'prompt_avg': float(avg_prompt),
            'prompt_std': float(std_prompt),
            'completion_avg': float(avg_completion),
            'completion_std': float(std_completion)
        },
        'gpu': gpu_metrics
    }


def generate_comprehensive_report(results: List[Dict], usage_stats: Dict, config: Optional[Dict] = None) -> Dict:
    """Generate comprehensive performance report with classification and efficiency metrics."""
    classification_metrics = calculate_mental_health_metrics(results)
    efficiency_metrics = calculate_efficiency_metrics(usage_stats)
    
    # Summary statistics
    summary = {
        'total_samples': len(results),
        'total_requests': usage_stats.get('num_requests', 0),
        'anxiety_accuracy': classification_metrics['anxiety']['accuracy'],
        'anxiety_f1': classification_metrics['anxiety']['f1_score'],
        'depression_accuracy': classification_metrics['depression']['accuracy'],
        'depression_f1': classification_metrics['depression']['f1_score'],
        'total_cost_usd': efficiency_metrics['cost']['total_usd'],
        'avg_latency_sec': efficiency_metrics['latency']['avg_seconds']
    }
    
    # Add stress metrics if available
    if 'stress' in classification_metrics:
        summary['stress_accuracy'] = classification_metrics['stress']['accuracy']
        summary['stress_f1'] = classification_metrics['stress']['f1_score']
    
    return {
        'classification_performance': classification_metrics,
        'cost_efficiency': efficiency_metrics,
        'config': config or {},
        'timestamp': datetime.now().isoformat(),
        'summary': summary
    }


def print_comprehensive_report(report: Dict):
    """Pretty print comprehensive performance report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("="*80)
    print(f"\nTimestamp: {report['timestamp']}")
    
    if report.get('config'):
        print("Configuration:")
        for key, value in report['config'].items():
            print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    summary = report['summary']
    print(f"  Total Samples:         {summary['total_samples']}")
    print(f"  Total Requests:        {summary['total_requests']}")
    
    # Print accuracy and F1 for each target dynamically
    for target in ['anxiety', 'depression', 'stress']:
        if f'{target}_accuracy' in summary:
            target_display = target.capitalize()
            print(f"  {target_display} Accuracy:      {summary[f'{target}_accuracy']:.4f}")
            print(f"  {target_display} F1 Score:      {summary[f'{target}_f1']:.4f}")
    
    print(f"  Total Cost:            ${summary['total_cost_usd']:.4f}")
    print(f"  Avg Latency:           {summary['avg_latency_sec']:.2f}s")
    
    # Classification Performance
    print("\n" + "="*80)
    print("7.1 CLASSIFICATION PERFORMANCE")
    print("="*80)
    
    classification = report['classification_performance']
    targets = ['anxiety', 'depression']
    if 'stress' in classification:
        targets.append('stress')
    
    for target in targets:
        metrics = classification[target]
        print(f"\n📊 {target.upper()} PREDICTION")
        print("-" * 80)
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1 Score:          {metrics['f1_score']:.4f}")
        print(f"  AUROC:             {metrics['auroc']:.4f}" if metrics['auroc'] else "  AUROC:             N/A")
        print(f"  Support:           {metrics['support']}")
        print(f"  Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    
    # Cost & Efficiency
    print("\n" + "="*80)
    print("7.2 COST & EFFICIENCY")
    print("="*80)
    
    efficiency = report['cost_efficiency']
    
    print("\n⏱️  LATENCY")
    print("-" * 80)
    print(f"  Average:       {efficiency['latency']['avg_seconds']:.2f} ± {efficiency['latency']['std_seconds']:.2f} seconds")
    print(f"  Total:         {efficiency['latency']['total_seconds']:.2f} seconds")
    
    print("\n💰 COST")
    print("-" * 80)
    print(f"  Total:         ${efficiency['cost']['total_usd']:.4f}")
    print(f"  Per Sample:    ${efficiency['cost']['per_sample_usd']:.6f} ± ${efficiency['cost']['per_sample_std']:.6f}")
    
    print("\n🚀 THROUGHPUT")
    print("-" * 80)
    print(f"  Tokens/Second:     {efficiency['throughput']['tokens_per_second']:.2f}")
    print(f"  Samples/Minute:    {efficiency['throughput']['samples_per_minute']:.2f}")
    
    print("\n🔢 TOKENS")
    print("-" * 80)
    print(f"  Total:         {efficiency['tokens']['total']:,}")
    print(f"  Prompt:        {efficiency['tokens']['prompt']:,} (avg: {efficiency['tokens']['prompt_avg']:.0f} ± {efficiency['tokens']['prompt_std']:.0f})")
    print(f"  Completion:    {efficiency['tokens']['completion']:,} (avg: {efficiency['tokens']['completion_avg']:.0f} ± {efficiency['tokens']['completion_std']:.0f})")
    print(f"  Avg/Request:   {efficiency['tokens']['avg_per_request']:.0f} ± {efficiency['tokens']['std_per_request']:.0f}")
    
    # GPU metrics (if available)
    if efficiency['gpu']['available']:
        print("\n🎮 GPU METRICS (On-Device Model)")
        print("-" * 80)
        print(f"  Avg Memory Used:    {efficiency['gpu']['avg_memory_mb']:.1f} ± {efficiency['gpu']['std_memory_mb']:.1f} MB")
        print(f"  Peak Memory Used:   {efficiency['gpu']['peak_memory_mb']:.1f} MB")
        print(f"  Avg Utilization:    {efficiency['gpu']['avg_utilization_percent']:.1f} ± {efficiency['gpu']['std_utilization_percent']:.1f} %")
    
    print("\n" + "="*80 + "\n")


def print_metrics_summary(metrics: Dict):
    """Pretty print metrics summary (shorter version)."""
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    
    for target in ['anxiety', 'depression']:
        m = metrics[target]
        print(f"\n📊 {target.upper()}")
        print("-" * 60)
        print(f"  Accuracy:          {m['accuracy']:.4f}  |  Precision: {m['precision']:.4f}")
        print(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}  |  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:          {m['f1_score']:.4f}", end="")
        if m['auroc']:
            print(f"  |  AUROC:     {m['auroc']:.4f}")
        else:
            print()
    
    print("="*60 + "\n")


def save_predictions_to_csv(predictions: List[Dict], filepath: str, reasoning_method: str = 'direct'):
    """
    Save prediction results to CSV file.
    
    Args:
        predictions: List of prediction dictionaries with user_id, ema_date, 
                    true_anxiety, true_depression, pred_anxiety, pred_depression
        filepath: Output CSV file path
        reasoning_method: Reasoning method used ('direct', 'cot', 'self_feedback')
    """
    records = []
    
    for pred in predictions:
        record = {
            'uid': pred['user_id'],
            'date': pred['ema_date'].strftime('%Y-%m-%d') if hasattr(pred['ema_date'], 'strftime') else str(pred['ema_date']),
            'y_anx_real': pred['true_anxiety'],
            'y_dep_real': pred['true_depression']
        }
        
        # For self_feedback, save all iteration results
        if reasoning_method == 'self_feedback' and 'prediction' in pred:
            prediction_data = pred['prediction']
            
            # Extract iteration-specific predictions and confidences
            # These were added by predict_with_self_feedback()
            iteration_num = 1
            while f'pred_iteration_{iteration_num}' in prediction_data:
                pred_iter = prediction_data[f'pred_iteration_{iteration_num}']
                conf_iter = prediction_data.get(f'conf_iteration_{iteration_num}', {})
                diff_iter = prediction_data.get(f'difficulty_iteration_{iteration_num}', 'N/A')
                
                record[f'y_anx_pred_{iteration_num}'] = pred_iter.get('Anxiety_binary', 0)
                record[f'y_dep_pred_{iteration_num}'] = pred_iter.get('Depression_binary', 0)
                record[f'y_anx_conf_{iteration_num}'] = conf_iter.get('Anxiety', 'N/A')
                record[f'y_dep_conf_{iteration_num}'] = conf_iter.get('Depression', 'N/A')
                record[f'difficulty_{iteration_num}'] = diff_iter
                
                iteration_num += 1
            
            # Add total iterations count
            if 'Reasoning' in prediction_data:
                record['total_iterations'] = prediction_data['Reasoning'].get('total_iterations', 1)
        
        # Always save final prediction
        record['y_anx_pred'] = pred['pred_anxiety']
        record['y_dep_pred'] = pred['pred_depression']
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"✅ Predictions saved: {filepath}")


def metrics_to_dataframe(metrics: Dict) -> pd.DataFrame:
    """Convert metrics dictionary to pandas DataFrame with confusion matrix values.
    
    Supports anxiety, depression, and stress (if available).
    """
    # Define target names and their display names
    targets = [
        ('anxiety', 'Anxiety'),
        ('depression', 'Depression'),
        ('stress', 'Stress')
    ]
    
    # Build data dictionary dynamically
    data = {}
    for target_key, target_display in targets:
        if target_key in metrics:
            m = metrics[target_key]
            data[target_display] = [
                m['accuracy'],
                m['balanced_accuracy'],
                m['precision'],
                m['recall'],
                m['f1_score'],
                m['auroc'] if m['auroc'] else 'N/A',
                m['tp'],
                m['tn'],
                m['fp'],
                m['fn']
            ]
    
    index_labels = [
        'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC',
        'TP (True Positive)', 'TN (True Negative)', 'FP (False Positive)', 'FN (False Negative)'
    ]
    
    return pd.DataFrame(data, index=index_labels)


def save_metrics_to_csv(metrics: Dict, filepath: str):
    """Save metrics to CSV file."""
    df = metrics_to_dataframe(metrics)
    df.to_csv(filepath)
    print(f"Metrics saved to: {filepath}")


def export_comprehensive_report(report: Dict, base_filepath: str, predictions: List[Dict] = None, reasoning_method: str = 'direct'):
    """Export comprehensive report to JSON and CSV formats."""
    # JSON export with NumpyEncoder to handle DataFrames and numpy types
    json_path = f"{base_filepath}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Full report saved: {json_path}")
    
    # Classification CSV
    class_df = metrics_to_dataframe(report['classification_performance'])
    class_path = f"{base_filepath}_classification.csv"
    class_df.to_csv(class_path)
    print(f"✅ Classification metrics saved: {class_path}")
    
    # Efficiency CSV with standard deviations
    efficiency = report['cost_efficiency']
    eff_metrics = [
        'Avg Latency (s)', 'Std Latency (s)', 'Total Latency (s)', 
        'Total Cost ($)', 'Cost per Sample ($)', 'Std Cost per Sample ($)',
        'Tokens/Second', 'Samples/Minute', 
        'Total Tokens', 'Avg Tokens per Request', 'Std Tokens per Request',
        'Prompt Tokens (Total)', 'Prompt Tokens (Avg)', 'Prompt Tokens (Std)',
        'Completion Tokens (Total)', 'Completion Tokens (Avg)', 'Completion Tokens (Std)'
    ]
    eff_values = [
        efficiency['latency']['avg_seconds'], efficiency['latency']['std_seconds'], 
        efficiency['latency']['total_seconds'],
        efficiency['cost']['total_usd'], efficiency['cost']['per_sample_usd'], 
        efficiency['cost']['per_sample_std'],
        efficiency['throughput']['tokens_per_second'], efficiency['throughput']['samples_per_minute'],
        efficiency['tokens']['total'], efficiency['tokens']['avg_per_request'], 
        efficiency['tokens']['std_per_request'],
        efficiency['tokens']['prompt'], efficiency['tokens']['prompt_avg'], 
        efficiency['tokens']['prompt_std'],
        efficiency['tokens']['completion'], efficiency['tokens']['completion_avg'], 
        efficiency['tokens']['completion_std']
    ]
    
    # Add GPU metrics if available
    if efficiency['gpu']['available']:
        eff_metrics.extend([
            'GPU Avg Memory (MB)', 'GPU Std Memory (MB)', 'GPU Peak Memory (MB)',
            'GPU Avg Utilization (%)', 'GPU Std Utilization (%)'
        ])
        eff_values.extend([
            efficiency['gpu']['avg_memory_mb'], efficiency['gpu']['std_memory_mb'],
            efficiency['gpu']['peak_memory_mb'],
            efficiency['gpu']['avg_utilization_percent'], efficiency['gpu']['std_utilization_percent']
        ])
    
    eff_data = {
        'Metric': eff_metrics,
        'Value': eff_values
    }
    eff_df = pd.DataFrame(eff_data)
    eff_path = f"{base_filepath}_efficiency.csv"
    eff_df.to_csv(eff_path, index=False)
    print(f"✅ Efficiency metrics saved: {eff_path}")
    
    # Predictions CSV
    if predictions:
        pred_path = f"{base_filepath}_predictions.csv"
        save_predictions_to_csv(predictions, pred_path, reasoning_method=reasoning_method)


def compare_experiments(reports: List[Dict], names: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare multiple experiment reports side-by-side."""
    if names is None:
        names = [f"Experiment {i+1}" for i in range(len(reports))]
    
    comparison_data = {
        'Metric': ['--- Classification ---', 'Anxiety Accuracy', 'Anxiety Balanced Acc', 'Anxiety F1', 
                  'Depression Accuracy', 'Depression Balanced Acc', 'Depression F1', 
                  '--- Efficiency ---', 'Avg Latency (s)', 'Std Latency (s)', 
                  'Total Cost ($)', 'Cost per Sample ($)', 'Std Cost per Sample ($)',
                  'Tokens/Second', 'Samples/Minute', 'Total Tokens', 'Avg Tokens/Request', 'Std Tokens/Request']
    }
    
    for name, report in zip(names, reports):
        class_perf = report['classification_performance']
        efficiency = report['cost_efficiency']
        
        comparison_data[name] = [
            '---',
            f"{class_perf['anxiety']['accuracy']:.4f}",
            f"{class_perf['anxiety']['balanced_accuracy']:.4f}",
            f"{class_perf['anxiety']['f1_score']:.4f}",
            f"{class_perf['depression']['accuracy']:.4f}",
            f"{class_perf['depression']['balanced_accuracy']:.4f}",
            f"{class_perf['depression']['f1_score']:.4f}",
            '---',
            f"{efficiency['latency']['avg_seconds']:.2f}",
            f"{efficiency['latency']['std_seconds']:.2f}",
            f"${efficiency['cost']['total_usd']:.4f}",
            f"${efficiency['cost']['per_sample_usd']:.6f}",
            f"${efficiency['cost']['per_sample_std']:.6f}",
            f"{efficiency['throughput']['tokens_per_second']:.2f}",
            f"{efficiency['throughput']['samples_per_minute']:.2f}",
            f"{efficiency['tokens']['total']:,}",
            f"{efficiency['tokens']['avg_per_request']:.0f}",
            f"{efficiency['tokens']['std_per_request']:.0f}"
        ]
    
    return pd.DataFrame(comparison_data)