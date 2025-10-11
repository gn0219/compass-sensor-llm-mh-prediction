"""
Performance Evaluation Module

Handles classification metrics (accuracy, F1, AUROC) and efficiency metrics (latency, tokens).
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .utils import NumpyEncoder


def calculate_binary_metrics(y_true: List[int], y_pred: List[int], 
                             y_proba: Optional[List[float]] = None, label_name: str = "") -> Dict:
    """Calculate classification metrics for binary prediction."""
    if len(y_true) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 
                'auroc': None, 'confusion_matrix': [[0, 0], [0, 0]], 
                'support': {'class_0': 0, 'class_1': 0},
                'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
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
        'accuracy': float(accuracy), 'precision': float(precision), 
        'recall': float(recall), 'f1_score': float(f1),
        'auroc': float(auroc) if auroc else None,
        'confusion_matrix': cm.tolist(), 'support': support,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def calculate_mental_health_metrics(results: List[Dict], anxiety_label_key: str = 'phq4_anxiety_EMA',
                                    depression_label_key: str = 'phq4_depression_EMA') -> Dict:
    """Calculate metrics for anxiety and depression predictions."""
    anxiety_true, anxiety_pred, anxiety_proba = [], [], []
    depression_true, depression_pred, depression_proba = [], [], []
    
    for result in results:
        if 'labels' not in result or 'prediction' not in result:
            continue
        
        if anxiety_label_key in result['labels'] and 'Anxiety_binary' in result['prediction']:
            anxiety_true.append(result['labels'][anxiety_label_key])
            anxiety_pred.append(result['prediction']['Anxiety_binary'])
            if 'proba' in result and 'Anxiety' in result['proba']:
                anxiety_proba.append(result['proba']['Anxiety'])
        
        if depression_label_key in result['labels'] and 'Depression_binary' in result['prediction']:
            depression_true.append(result['labels'][depression_label_key])
            depression_pred.append(result['prediction']['Depression_binary'])
            if 'proba' in result and 'Depression' in result['proba']:
                depression_proba.append(result['proba']['Depression'])
    
    anxiety_metrics = calculate_binary_metrics(anxiety_true, anxiety_pred, 
                                               anxiety_proba if anxiety_proba else None, "Anxiety")
    depression_metrics = calculate_binary_metrics(depression_true, depression_pred,
                                                  depression_proba if depression_proba else None, "Depression")
    
    overall_metrics = {
        'accuracy': (anxiety_metrics['accuracy'] + depression_metrics['accuracy']) / 2,
        'precision': (anxiety_metrics['precision'] + depression_metrics['precision']) / 2,
        'recall': (anxiety_metrics['recall'] + depression_metrics['recall']) / 2,
        'f1_score': (anxiety_metrics['f1_score'] + depression_metrics['f1_score']) / 2,
        'n_samples': len(results)
    }
    
    if anxiety_metrics['auroc'] and depression_metrics['auroc']:
        overall_metrics['auroc'] = (anxiety_metrics['auroc'] + depression_metrics['auroc']) / 2
    else:
        overall_metrics['auroc'] = None
    
    return {'anxiety': anxiety_metrics, 'depression': depression_metrics, 'overall': overall_metrics}


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
                      'prompt_avg': 0, 'prompt_std': 0, 'completion_avg': 0, 'completion_std': 0}
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
        }
    }


def generate_comprehensive_report(results: List[Dict], usage_stats: Dict, config: Optional[Dict] = None) -> Dict:
    """Generate comprehensive performance report with classification and efficiency metrics."""
    classification_metrics = calculate_mental_health_metrics(results)
    efficiency_metrics = calculate_efficiency_metrics(usage_stats)
    
    # Summary statistics
    summary = {
        'total_samples': len(results),
        'total_requests': usage_stats.get('num_requests', 0),
        'overall_accuracy': classification_metrics['overall']['accuracy'],
        'overall_f1': classification_metrics['overall']['f1_score'],
        'total_cost_usd': efficiency_metrics['cost']['total_usd'],
        'avg_latency_sec': efficiency_metrics['latency']['avg_seconds']
    }
    
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
    print(f"  Total Samples:     {summary['total_samples']}")
    print(f"  Total Requests:    {summary['total_requests']}")
    print(f"  Overall Accuracy:  {summary['overall_accuracy']:.4f}")
    print(f"  Overall F1 Score:  {summary['overall_f1']:.4f}")
    print(f"  Total Cost:        ${summary['total_cost_usd']:.4f}")
    print(f"  Avg Latency:       {summary['avg_latency_sec']:.2f}s")
    
    # Classification Performance
    print("\n" + "="*80)
    print("7.1 CLASSIFICATION PERFORMANCE")
    print("="*80)
    
    classification = report['classification_performance']
    for target in ['anxiety', 'depression']:
        metrics = classification[target]
        print(f"\n📊 {target.upper()} PREDICTION")
        print("-" * 80)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}" if metrics['auroc'] else "  AUROC:     N/A")
        print(f"  Support:   {metrics['support']}")
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
        print(f"  Accuracy:  {m['accuracy']:.4f}  |  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}  |  F1 Score:  {m['f1_score']:.4f}")
        if m['auroc']:
            print(f"  AUROC:     {m['auroc']:.4f}")
    
    print("\n📊 OVERALL (Macro Average)")
    print("-" * 60)
    overall = metrics['overall']
    print(f"  Accuracy:  {overall['accuracy']:.4f}  |  F1 Score:  {overall['f1_score']:.4f}")
    print("="*60 + "\n")


def metrics_to_dataframe(metrics: Dict) -> pd.DataFrame:
    """Convert metrics dictionary to pandas DataFrame with confusion matrix values."""
    data = {
        'Anxiety': [
            metrics['anxiety']['accuracy'],
            metrics['anxiety']['precision'],
            metrics['anxiety']['recall'],
            metrics['anxiety']['f1_score'],
            metrics['anxiety']['auroc'] if metrics['anxiety']['auroc'] else 'N/A',
            metrics['anxiety']['tp'],
            metrics['anxiety']['tn'],
            metrics['anxiety']['fp'],
            metrics['anxiety']['fn']
        ],
        'Depression': [
            metrics['depression']['accuracy'],
            metrics['depression']['precision'],
            metrics['depression']['recall'],
            metrics['depression']['f1_score'],
            metrics['depression']['auroc'] if metrics['depression']['auroc'] else 'N/A',
            metrics['depression']['tp'],
            metrics['depression']['tn'],
            metrics['depression']['fp'],
            metrics['depression']['fn']
        ]
    }
    return pd.DataFrame(data, index=[
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC',
        'TP (True Positive)', 'TN (True Negative)', 'FP (False Positive)', 'FN (False Negative)'
    ])


def save_metrics_to_csv(metrics: Dict, filepath: str):
    """Save metrics to CSV file."""
    df = metrics_to_dataframe(metrics)
    df.to_csv(filepath)
    print(f"Metrics saved to: {filepath}")


def export_comprehensive_report(report: Dict, base_filepath: str):
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
    eff_data = {
        'Metric': [
            'Avg Latency (s)', 'Std Latency (s)', 'Total Latency (s)', 
            'Total Cost ($)', 'Cost per Sample ($)', 'Std Cost per Sample ($)',
            'Tokens/Second', 'Samples/Minute', 
            'Total Tokens', 'Avg Tokens per Request', 'Std Tokens per Request',
            'Prompt Tokens (Total)', 'Prompt Tokens (Avg)', 'Prompt Tokens (Std)',
            'Completion Tokens (Total)', 'Completion Tokens (Avg)', 'Completion Tokens (Std)'
        ],
        'Value': [
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
    }
    eff_df = pd.DataFrame(eff_data)
    eff_path = f"{base_filepath}_efficiency.csv"
    eff_df.to_csv(eff_path, index=False)
    print(f"✅ Efficiency metrics saved: {eff_path}")


def compare_experiments(reports: List[Dict], names: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare multiple experiment reports side-by-side."""
    if names is None:
        names = [f"Experiment {i+1}" for i in range(len(reports))]
    
    comparison_data = {
        'Metric': ['--- Classification ---', 'Anxiety Accuracy', 'Anxiety F1', 'Depression Accuracy', 
                  'Depression F1', 'Overall Accuracy', 'Overall F1',
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
            f"{class_perf['anxiety']['f1_score']:.4f}",
            f"{class_perf['depression']['accuracy']:.4f}",
            f"{class_perf['depression']['f1_score']:.4f}",
            f"{class_perf['overall']['accuracy']:.4f}",
            f"{class_perf['overall']['f1_score']:.4f}",
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