"""
Analyze retrieval similarity for MentalIoT cross_retrieval strategy.

This script analyzes:
1. Cosine similarity scores between target and retrieved examples
2. Feature-level similarity (which features are most similar)
3. Label agreement (do similar samples have similar labels?)
4. Top features contributing to similarity
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.data_utils import sample_mentaliot_testset
from src.example_selection import build_retrieval_candidate_pool, select_icl_examples
from src import config

def analyze_retrieval_quality(
    n_samples_to_analyze=20,
    n_shot=4,
    random_state=42,
    output_file='retrieval_similarity_analysis.csv'
):
    """
    Analyze the quality of retrieved examples.
    
    Args:
        n_samples_to_analyze: Number of test samples to analyze
        n_shot: Number of examples to retrieve per sample
        random_state: Random seed for reproducibility
        output_file: Output CSV file path
    """
    
    # Load data
    print("Loading MentalIoT data...")
    feat_df, lab_df, test_df, train_df, cols = sample_mentaliot_testset(
        n_samples_per_user=10,
        random_state=random_state
    )
    
    # Build candidate pool
    print("Building candidate pool...")
    config.MENTALIOT_TRAIN_DF = train_df
    retrieval_candidates = build_retrieval_candidate_pool(
        feat_df, train_df, cols,
        max_pool_size=300,
        random_state=random_state,
        dataset='mentaliot'
    )
    
    # Get statistical features for similarity computation
    stat_features = list(cols['feature_set']['statistical'].keys())
    
    # Analysis results
    results = []
    
    print(f"\nAnalyzing {n_samples_to_analyze} samples...")
    
    for i in range(min(n_samples_to_analyze, len(test_df))):
        row = test_df.iloc[i]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Get target sample features
        feat_row = feat_df[
            (feat_df[cols['user_id']] == user_id) &
            (feat_df[cols['date']] == ema_date)
        ]
        
        if len(feat_row) == 0:
            continue
        
        agg_feats = feat_row.iloc[0].to_dict()
        labels = row[cols['labels']].to_dict()
        
        target_sample = {
            'aggregated_features': agg_feats,
            'labels': labels,
            'user_id': user_id,
            'ema_date': ema_date
        }
        
        # Build target feature vector
        target_vector = []
        for feat in stat_features:
            val = agg_feats.get(feat, 0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0
            target_vector.append(float(val))
        target_vector = np.array(target_vector).reshape(1, -1)
        
        # Get retrieved examples
        examples = select_icl_examples(
            feat_df, train_df, cols,
            target_user_id=user_id,
            target_ema_date=ema_date,
            n_shot=n_shot,
            strategy='cross_retrieval',
            retrieval_candidates=retrieval_candidates,
            target_sample=target_sample,
            dataset='mentaliot'
        )
        
        if not examples or len(examples) == 0:
            print(f"Sample {i}: No examples retrieved")
            continue
        
        print(f"\nSample {i} (user={user_id}, date={ema_date}):")
        print(f"  Target labels: anxiety={labels['phq4_anxiety_EMA']}, "
              f"depression={labels['phq4_depression_EMA']}, stress={labels['stress']}")
        
        # Analyze each retrieved example
        for ex_idx, example in enumerate(examples):
            # Build example feature vector
            ex_vector = []
            for feat in stat_features:
                val = example['aggregated_features'].get(feat, 0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0
                ex_vector.append(float(val))
            ex_vector = np.array(ex_vector).reshape(1, -1)
            
            # Compute cosine similarity
            similarity = cosine_similarity(target_vector, ex_vector)[0][0]
            
            # Label agreement
            ex_labels = example['labels']
            anxiety_match = (labels['phq4_anxiety_EMA'] == ex_labels['phq4_anxiety_EMA'])
            depression_match = (labels['phq4_depression_EMA'] == ex_labels['phq4_depression_EMA'])
            stress_match = (labels['stress'] == ex_labels['stress'])
            all_match = anxiety_match and depression_match and stress_match
            
            # Feature-level differences (top contributing features)
            feature_diffs = []
            for j, feat in enumerate(stat_features):
                target_val = target_vector[0][j]
                ex_val = ex_vector[0][j]
                if target_val != 0 or ex_val != 0:  # Skip if both are 0
                    diff = abs(target_val - ex_val)
                    feature_diffs.append((feat, diff, target_val, ex_val))
            
            # Sort by difference
            feature_diffs.sort(key=lambda x: x[1], reverse=True)
            top_diff_features = feature_diffs[:5]
            
            print(f"  Example {ex_idx+1}:")
            print(f"    Cosine similarity: {similarity:.4f}")
            print(f"    Labels: anxiety={ex_labels['phq4_anxiety_EMA']} "
                  f"(match={anxiety_match}), depression={ex_labels['phq4_depression_EMA']} "
                  f"(match={depression_match}), stress={ex_labels['stress']} (match={stress_match})")
            print(f"    All labels match: {all_match}")
            print(f"    Top 3 different features:")
            for feat_name, diff, t_val, e_val in top_diff_features[:3]:
                print(f"      {feat_name}: target={t_val:.2f}, example={e_val:.2f}, diff={diff:.2f}")
            
            # Store results
            results.append({
                'sample_idx': i,
                'target_user': user_id,
                'target_date': ema_date,
                'target_anxiety': labels['phq4_anxiety_EMA'],
                'target_depression': labels['phq4_depression_EMA'],
                'target_stress': labels['stress'],
                'example_idx': ex_idx + 1,
                'example_user': example['user_id'],
                'example_date': example['ema_date'],
                'example_anxiety': ex_labels['phq4_anxiety_EMA'],
                'example_depression': ex_labels['phq4_depression_EMA'],
                'example_stress': ex_labels['stress'],
                'cosine_similarity': similarity,
                'anxiety_match': anxiety_match,
                'depression_match': depression_match,
                'stress_match': stress_match,
                'all_labels_match': all_match,
                'top_diff_feature_1': top_diff_features[0][0] if len(top_diff_features) > 0 else None,
                'top_diff_value_1': top_diff_features[0][1] if len(top_diff_features) > 0 else None,
                'top_diff_feature_2': top_diff_features[1][0] if len(top_diff_features) > 1 else None,
                'top_diff_value_2': top_diff_features[1][1] if len(top_diff_features) > 1 else None,
                'top_diff_feature_3': top_diff_features[2][0] if len(top_diff_features) > 2 else None,
                'top_diff_value_3': top_diff_features[2][1] if len(top_diff_features) > 2 else None,
            })
    
    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n[OK] Saved analysis to {output_file}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total retrievals analyzed: {len(df_results)}")
    print(f"\nCosine Similarity:")
    print(f"  Mean: {df_results['cosine_similarity'].mean():.4f}")
    print(f"  Median: {df_results['cosine_similarity'].median():.4f}")
    print(f"  Min: {df_results['cosine_similarity'].min():.4f}")
    print(f"  Max: {df_results['cosine_similarity'].max():.4f}")
    
    print(f"\nLabel Agreement Rates:")
    print(f"  Anxiety: {df_results['anxiety_match'].mean()*100:.1f}%")
    print(f"  Depression: {df_results['depression_match'].mean()*100:.1f}%")
    print(f"  Stress: {df_results['stress_match'].mean()*100:.1f}%")
    print(f"  All labels match: {df_results['all_labels_match'].mean()*100:.1f}%")
    
    # Correlation between similarity and label agreement
    print(f"\nCorrelation (Similarity vs Label Agreement):")
    for label in ['anxiety', 'depression', 'stress']:
        corr = df_results['cosine_similarity'].corr(df_results[f'{label}_match'].astype(int))
        print(f"  {label.capitalize()}: {corr:.4f}")
    
    return df_results


if __name__ == "__main__":
    analyze_retrieval_quality(
        n_samples_to_analyze=20,
        n_shot=4,
        random_state=42,
        output_file='mentaliot_retrieval_similarity_analysis.csv'
    )

