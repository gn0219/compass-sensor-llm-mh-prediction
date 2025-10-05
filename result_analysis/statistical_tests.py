"""
Statistical Significance Testing Module

Provides statistical tests for comparing model performance.
Essential for claiming that one method is significantly better than another.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import accuracy_score


def mcnemar_test(
    y_true: List[int],
    y_pred1: List[int],
    y_pred2: List[int],
    alpha: float = 0.05
) -> Dict:
    """
    McNemar's test for paired predictions.
    
    Tests whether two models have significantly different error rates on the same samples.
    Appropriate for comparing classifiers on the same test set.
    
    Args:
        y_true: Ground truth labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        alpha: Significance level (default: 0.05)
    
    Returns:
        Dictionary with:
            - statistic: McNemar's chi-square statistic
            - p_value: Two-tailed p-value
            - significant: Whether difference is significant at alpha level
            - effect_size: Difference in accuracy
            - interpretation: Human-readable result
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    # Build contingency table
    # correct1_correct2: Both models correct
    # correct1_wrong2: Model 1 correct, Model 2 wrong
    # wrong1_correct2: Model 1 wrong, Model 2 correct
    # wrong1_wrong2: Both models wrong
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    correct1_wrong2 = np.sum(correct1 & ~correct2)  # b
    wrong1_correct2 = np.sum(~correct1 & correct2)  # c
    
    # McNemar's test statistic with continuity correction
    if (correct1_wrong2 + wrong1_correct2) == 0:
        # No disagreements, models are identical
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = ((abs(correct1_wrong2 - wrong1_correct2) - 1) ** 2) / (correct1_wrong2 + wrong1_correct2)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Effect size (difference in accuracy)
    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)
    effect_size = acc2 - acc1
    
    # Interpretation
    if p_value < alpha:
        if effect_size > 0:
            interpretation = f"Model 2 is significantly better (p={p_value:.4f}, Δacc={effect_size:.4f})"
        else:
            interpretation = f"Model 1 is significantly better (p={p_value:.4f}, Δacc={abs(effect_size):.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f}, Δacc={effect_size:.4f})"
    
    return {
        'test': 'mcnemar',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'effect_size': float(effect_size),
        'accuracy_model1': float(acc1),
        'accuracy_model2': float(acc2),
        'interpretation': interpretation,
        'contingency_table': {
            'both_correct': int(np.sum(correct1 & correct2)),
            'model1_correct_model2_wrong': int(correct1_wrong2),
            'model1_wrong_model2_correct': int(wrong1_correct2),
            'both_wrong': int(np.sum(~correct1 & ~correct2))
        }
    }


def bootstrap_confidence_interval(
    y_true: List[int],
    y_pred: List[int],
    metric_func=accuracy_score,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict:
    """
    Bootstrap confidence interval for a metric.
    
    Estimates the uncertainty in a performance metric through resampling.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        metric_func: Metric function (default: accuracy_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        random_state: Random seed
    
    Returns:
        Dictionary with:
            - metric: Point estimate
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            - std_error: Standard error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        metric_val = metric_func(y_true_boot, y_pred_boot)
        bootstrap_metrics.append(metric_val)
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    # Point estimate
    metric = metric_func(y_true, y_pred)
    
    # Standard error
    std_error = np.std(bootstrap_metrics)
    
    return {
        'metric': float(metric),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std_error': float(std_error),
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap
    }


def compare_two_models(
    y_true: List[int],
    y_pred1: List[int],
    y_pred2: List[int],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive comparison of two models.
    
    Combines McNemar's test with bootstrap confidence intervals.
    
    Args:
        y_true: Ground truth labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        alpha: Significance level
    
    Returns:
        Dictionary with all test results and interpretation
    """
    # McNemar's test
    mcnemar_result = mcnemar_test(y_true, y_pred1, y_pred2, alpha)
    
    # Bootstrap CIs for both models
    ci1 = bootstrap_confidence_interval(y_true, y_pred1)
    ci2 = bootstrap_confidence_interval(y_true, y_pred2)
    
    # Check if CIs overlap
    ci_overlap = not (ci1['ci_upper'] < ci2['ci_lower'] or ci2['ci_upper'] < ci1['ci_lower'])
    
    # Overall interpretation
    if mcnemar_result['significant']:
        if mcnemar_result['effect_size'] > 0:
            overall = f"{model2_name} significantly outperforms {model1_name}"
        else:
            overall = f"{model1_name} significantly outperforms {model2_name}"
    else:
        overall = f"No significant difference between {model1_name} and {model2_name}"
    
    return {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'mcnemar_test': mcnemar_result,
        'model1_ci': ci1,
        'model2_ci': ci2,
        'ci_overlap': ci_overlap,
        'overall_interpretation': overall
    }


def compare_multiple_models(
    y_true: List[int],
    predictions: Dict[str, List[int]],
    alpha: float = 0.05
) -> Dict:
    """
    Pairwise comparison of multiple models.
    
    Args:
        y_true: Ground truth labels
        predictions: Dict mapping model names to predictions
        alpha: Significance level
    
    Returns:
        Dictionary with pairwise comparison matrix
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    # Pairwise comparisons
    comparisons = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            name1 = model_names[i]
            name2 = model_names[j]
            
            comparison = compare_two_models(
                y_true,
                predictions[name1],
                predictions[name2],
                name1,
                name2,
                alpha
            )
            
            comparisons[f"{name1}_vs_{name2}"] = comparison
    
    return {
        'n_models': n_models,
        'model_names': model_names,
        'pairwise_comparisons': comparisons,
        'alpha': alpha
    }


def print_comparison_results(comparison: Dict):
    """
    Pretty print comparison results.
    
    Args:
        comparison: Results from compare_two_models()
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    
    print(f"\n{comparison['model1_name']} vs {comparison['model2_name']}")
    print("-" * 80)
    
    # Model 1 stats
    ci1 = comparison['model1_ci']
    print(f"\n{comparison['model1_name']}:")
    print(f"  Accuracy: {ci1['metric']:.4f} (95% CI: [{ci1['ci_lower']:.4f}, {ci1['ci_upper']:.4f}])")
    
    # Model 2 stats
    ci2 = comparison['model2_ci']
    print(f"\n{comparison['model2_name']}:")
    print(f"  Accuracy: {ci2['metric']:.4f} (95% CI: [{ci2['ci_lower']:.4f}, {ci2['ci_upper']:.4f}])")
    
    # McNemar's test
    mcnemar = comparison['mcnemar_test']
    print(f"\nMcNemar's Test:")
    print(f"  Statistic: {mcnemar['statistic']:.4f}")
    print(f"  P-value: {mcnemar['p_value']:.4f}")
    print(f"  Significant: {mcnemar['significant']} (α={mcnemar['alpha']})")
    print(f"  Effect size: {mcnemar['effect_size']:.4f}")
    
    # Interpretation
    print(f"\n📊 Interpretation:")
    print(f"  {mcnemar['interpretation']}")
    print(f"  Confidence intervals {'overlap' if comparison['ci_overlap'] else 'do not overlap'}")
    
    # Overall
    print(f"\n✅ Conclusion:")
    print(f"  {comparison['overall_interpretation']}")
    
    print("="*80 + "\n")

