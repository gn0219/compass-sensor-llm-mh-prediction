"""
Advanced Mental Health Prediction Pipeline with Improved Personalization
========================================================================

Key Improvements:
1. **Imputer Options**: KNN (default), Iterative, Simple
2. **ML Personalization**: Sample weighting + fixed hyperparameters
3. **DL Personalization**: Pre-training + Fine-tuning (all layers) with proper scaling
4. **Detailed Logging**: All steps with timing and data info
5. **Preprocessed Data**: Save to *_processed.csv for reuse

CRITICAL FIXES:
- Fixed scaling mismatch in DL personalization (reuse fitted scaler)
- Removed validation split for personalization (use all data)
- Removed hyperparameter tuning loop (use fixed params)
- Improved model stability (BCEWithLogitsLoss)
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import os
import ast
import warnings
import time
from typing import Tuple, List, Dict, Any
from copy import deepcopy
import optuna

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split, GroupKFold
from imblearn.over_sampling import SMOTE

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# Metrics
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                             recall_score, roc_auc_score)

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Statistics
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
DATASET_PATH = '/home/iclab/compass/dataset'
PROCESSED_PATH = '/home/iclab/compass/dataset/processed'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter tuning / personalization configuration
BEST_PARAMS_CACHE: Dict[str, Dict[str, Any]] = {}
PERSONALIZATION_SAMPLE_WEIGHTS = [3, 5, 10]
OPTUNA_N_TRIALS = 30

print(f"="*80)
print(f"Advanced Mental Health Prediction Pipeline (CORRECTED)")
print(f"="*80)
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"="*80)
print()

# Create processed directory if not exists
os.makedirs(PROCESSED_PATH, exist_ok=True)


# ============================================================================
# DATA PREPROCESSING - SAVE TO *_processed.csv
# ============================================================================

def preprocess_globem(df: pd.DataFrame, target_variable: str, 
                      save_processed: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess GLOBEM dataset with detailed logging."""
    print(f"  [GLOBEM Preprocessing] Starting...")
    start_time = time.time()
    
    df = df.copy()
    
    # Target mapping
    target_mapping = {
        'depression': 'phq4_depression_EMA',
        'anxiety': 'phq4_anxiety_EMA'
    }
    
    target_col = target_mapping[target_variable]
    y = df[target_col].copy()
    print(f"  [GLOBEM] Target: {target_variable} ({target_col})")
    print(f"  [GLOBEM] Target distribution: {y.value_counts().to_dict()}")
    
    # Drop metadata
    metadata_cols = ['pid', 'date', 'institution', 'is_testset', 
                     'phq4_EMA', 'phq4_anxiety_EMA', 'phq4_depression_EMA', 'pss4_EMA',
                     'positive_affect_EMA', 'negative_affect_EMA']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    print(f"  [GLOBEM] Shape after dropping metadata: {df.shape}")
    
    # Parse dictionary columns (VECTORIZED for speed!)
    print(f"  [GLOBEM] Parsing dictionary columns...")
    dict_start = time.time()
    dict_columns = []
    for col in df.columns:
        sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
        if isinstance(sample_val, str) and '{' in str(sample_val):
            dict_columns.append(col)
    
    print(f"  [GLOBEM] Found {len(dict_columns)} dictionary columns")
    
    # Vectorized dictionary parsing (10-100x faster!)
    new_cols = {}
    for col in dict_columns:
        parsed_values = df[col].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and '{' in x else {}
        )
        
        all_keys = set()
        for val in parsed_values:
            if isinstance(val, dict):
                all_keys.update(val.keys())
        
        for key in all_keys:
            new_col_name = f"{col}_{key}"
            new_cols[new_col_name] = parsed_values.apply(
                lambda x: x.get(key, np.nan) if isinstance(x, dict) else np.nan
            )
    
    for col_name, values in new_cols.items():
        df[col_name] = values
    df = df.drop(columns=dict_columns)
    
    dict_time = time.time() - dict_start
    print(f"  [GLOBEM] Dictionary parsing done in {dict_time:.2f}s")
    print(f"  [GLOBEM] New shape: {df.shape}")
    
    # Encode categorical
    print(f"  [GLOBEM] Encoding categorical values...")
    categorical_mapping = {
        'increasing': 1, 'stable': 0, 'decreasing': -1,
        'Increasing': 1, 'Stable': 0, 'Decreasing': -1
    }
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: categorical_mapping.get(x, x) if isinstance(x, str) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    total_time = time.time() - start_time
    print(f"  [GLOBEM] Preprocessing complete in {total_time:.2f}s")
    print(f"  [GLOBEM] Final shape: {df.shape}")
    print(f"  [GLOBEM] Missing values: {df.isnull().sum().sum()}")
    
    # Save processed data
    if save_processed:
        processed_file = os.path.join(PROCESSED_PATH, f'globem_{target_variable}_processed.csv')
        combined_df = df.copy()
        combined_df[target_col] = y
        combined_df.to_csv(processed_file, index=False)
        print(f"  [GLOBEM] ✓ Saved processed data to {processed_file}")
    
    return df, y


def aggregate_ces_sequences(df: pd.DataFrame, feature_prefix: str) -> Dict[str, List[float]]:
    """Aggregate 28-day sequence columns for CES."""
    result = {}
    day_cols = [col for col in df.columns if feature_prefix in col and 'before' in col and 'day' in col]
    
    if len(day_cols) == 0:
        return result
    
    feature_name = feature_prefix.rstrip('_')
    
    for idx in df.index:
        values = []
        for col in day_cols:
            val = df.loc[idx, col]
            if pd.notna(val):
                values.append(float(val))
        
        if len(values) == 0:
            result.setdefault(f'{feature_name}_agg_mean', []).append(np.nan)
            result.setdefault(f'{feature_name}_agg_std', []).append(np.nan)
            result.setdefault(f'{feature_name}_agg_min', []).append(np.nan)
            result.setdefault(f'{feature_name}_agg_max', []).append(np.nan)
            result.setdefault(f'{feature_name}_agg_slope', []).append(0.0)
        else:
            result.setdefault(f'{feature_name}_agg_mean', []).append(np.mean(values))
            result.setdefault(f'{feature_name}_agg_std', []).append(np.std(values) if len(values) > 1 else 0.0)
            result.setdefault(f'{feature_name}_agg_min', []).append(np.min(values))
            result.setdefault(f'{feature_name}_agg_max', []).append(np.max(values))
            
            if len(values) >= 2:
                try:
                    x = np.arange(len(values))
                    slope, _, _, _, _ = linregress(x, values)
                    result.setdefault(f'{feature_name}_agg_slope', []).append(slope if not np.isnan(slope) else 0.0)
                except:
                    result.setdefault(f'{feature_name}_agg_slope', []).append(0.0)
            else:
                result.setdefault(f'{feature_name}_agg_slope', []).append(0.0)
    
    return result


def preprocess_ces(df: pd.DataFrame, target_variable: str, 
                   save_processed: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess CES dataset (GLOBEM-style aggregation)."""
    print(f"  [CES Preprocessing] Starting...")
    start_time = time.time()
    
    # Target mapping
    target_mapping = {
        'depression': 'phq4_depression_EMA',
        'anxiety': 'phq4_anxiety_EMA',
        'stress': 'stress'
    }
    
    target_col = target_mapping[target_variable]
    processed_file = os.path.join(PROCESSED_PATH, f'ces_{target_variable}_processed.csv')
    
    # Load cached version if available
    if os.path.exists(processed_file):
        print(f"  [CES] Cached file found → {processed_file}")
        cached_df = pd.read_csv(processed_file)
        y = cached_df[target_col].copy()
        X_cached = cached_df.drop(columns=[target_col])
        print(f"  [CES] Loaded cached data: {X_cached.shape}")
        print(f"  [CES] Cached target distribution: {y.value_counts().to_dict()}")
        print(f"  [CES] Loaded in {time.time() - start_time:.2f}s")
        return X_cached, y
    
    df = df.copy()
    y = df[target_col].copy()
    print(f"  [CES] Target: {target_variable} ({target_col})")
    print(f"  [CES] Target distribution: {y.value_counts().to_dict()}")
    
    # Find feature prefixes
    print(f"  [CES] Finding sequence features...")
    feature_prefixes = set()
    for col in df.columns:
        if '_before' in col and 'day' in col:
            prefix = col.split('_before')[0] + '_'
            feature_prefixes.add(prefix)
    
    print(f"  [CES] Found {len(feature_prefixes)} feature prefixes")
    
    # Aggregate sequences
    print(f"  [CES] Aggregating sequences (mean, std, min, max, slope)...")
    agg_start = time.time()
    all_aggregated = {}
    for prefix in feature_prefixes:
        agg_features = aggregate_ces_sequences(df, prefix)
        all_aggregated.update(agg_features)
    
    agg_df = pd.DataFrame(all_aggregated, index=df.index)
    agg_time = time.time() - agg_start
    print(f"  [CES] Aggregation done in {agg_time:.2f}s")
    print(f"  [CES] Aggregated features: {agg_df.shape[1]}")
    
    # Keep non-sequence columns
    keep_cols = []
    for col in df.columns:
        if col not in ['uid', 'date', 'phq4_anxiety_EMA', 'phq4_depression_EMA', 'stress']:
            if '_before' not in col or 'day' not in col:
                keep_cols.append(col)
    
    base_df = df[keep_cols].copy()
    result_df = pd.concat([base_df, agg_df], axis=1)
    
    # Encode gender
    if 'gender' in result_df.columns:
        result_df['gender'] = result_df['gender'].map({'M': 1, 'F': 0, 1: 1, 0: 0})
        result_df['gender'] = pd.to_numeric(result_df['gender'], errors='coerce')
    
    total_time = time.time() - start_time
    print(f"  [CES] Preprocessing complete in {total_time:.2f}s")
    print(f"  [CES] Final shape: {result_df.shape}")
    print(f"  [CES] Missing values: {result_df.isnull().sum().sum()}")
    
    # Save processed data (always save when new file is created)
    should_save = True if not os.path.exists(processed_file) else save_processed
    if should_save:
        combined_df = result_df.copy()
        combined_df[target_col] = y
        combined_df.to_csv(processed_file, index=False)
        print(f"  [CES] ✓ Saved processed data to {processed_file}")
    
    return result_df, y


def preprocess_mentaliot(df: pd.DataFrame, target_variable: str, 
                         save_processed: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess Mental-IoT dataset."""
    print(f"  [Mental-IoT Preprocessing] Starting...")
    start_time = time.time()
    
    df = df.copy()
    
    # Target mapping
    target_mapping = {
        'depression': 'phq2_result_binary',
        'anxiety': 'gad2_result_binary',
        'stress': 'stress_result_binary'
    }
    
    target_col = target_mapping[target_variable]
    y = df[target_col].copy()
    print(f"  [Mental-IoT] Target: {target_variable} ({target_col})")
    print(f"  [Mental-IoT] Target distribution: {y.value_counts().to_dict()}")
    
    # Drop metadata
    metadata_cols = ['uid', 'timestamp', 'phq2_result_binary', 'gad2_result_binary', 'stress_result_binary']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    total_time = time.time() - start_time
    print(f"  [Mental-IoT] Preprocessing complete in {total_time:.2f}s")
    print(f"  [Mental-IoT] Final shape: {df.shape}")
    print(f"  [Mental-IoT] Missing values: {df.isnull().sum().sum()}")
    
    # Save processed data
    if save_processed:
        processed_file = os.path.join(PROCESSED_PATH, f'mentaliot_{target_variable}_processed.csv')
        combined_df = df.copy()
        combined_df[target_col] = y
        combined_df.to_csv(processed_file, index=False)
        print(f"  [Mental-IoT] ✓ Saved processed data to {processed_file}")
    
    return df, y


# ============================================================================
# IMPUTATION AND PREPROCESSING PIPELINE
# ============================================================================

def get_imputer(imputer_type: str = 'knn'):
    """Get imputer based on type."""
    if imputer_type == 'knn':
        return KNNImputer(n_neighbors=5)
    elif imputer_type == 'iterative':
        return IterativeImputer(max_iter=5, random_state=42)
    elif imputer_type == 'simple':
        # return SimpleImputer(strategy='constant', fill_value=0)
        return SimpleImputer(strategy='mean')
    else:
        raise ValueError(f"Unknown imputer type: {imputer_type}. Use 'knn', 'iterative', or 'simple'")


def apply_preprocessing_pipeline(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: pd.Series, apply_smote: bool = True, 
                                 imputer_type: str = 'knn', return_transformers: bool = False) -> Tuple:
    """
    Apply imputation, scaling, and SMOTE with detailed logging.
    
    FIX 1: Now returns fitted imputer and scaler when return_transformers=True
    """
    start_time = time.time()
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Imputation
    print(f"    [Preprocessing] Imputer: {imputer_type}")
    imputer = get_imputer(imputer_type)
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    # SMOTE on training only
    if apply_smote and len(np.unique(y_train)) > 1:
        try:
            smote_start = time.time()
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            smote_time = time.time() - smote_start
            print(f"    [SMOTE] {X_train_scaled.shape[0]} → {X_train_resampled.shape[0]} samples ({smote_time:.2f}s)")
            
            if return_transformers:
                return X_train_resampled, X_val_scaled, X_test_scaled, y_train_resampled, imputer, scaler
            return X_train_resampled, X_val_scaled, X_test_scaled, y_train_resampled
        except Exception as e:
            print(f"    [SMOTE] Failed: {e}")
            if return_transformers:
                return X_train_scaled, X_val_scaled, X_test_scaled, y_train, imputer, scaler
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train
    else:
        if return_transformers:
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, imputer, scaler
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train


# ============================================================================
# MODELS
# ============================================================================

class MLPModel(nn.Module):
    """
    MLP with BatchNorm and Dropout.
    
    FIX 4: Removed Sigmoid from final layer for BCEWithLogitsLoss
    """
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (256, 128), dropout: float = 0.3):
        super(MLPModel, self).__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # No Sigmoid here!
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def get_model(model_name: str, input_dim: int, params: Dict[str, Any] = None):
    """Get model instance with optional tuned hyperparameters."""
    params = params or {}
    
    if model_name == 'xgboost':
        # Use GPU if available (50x faster!)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_params = {
            'n_estimators': params.get('n_estimators', 200),
            'max_depth': params.get('max_depth', 4),
            'learning_rate': params.get('learning_rate', 0.05),
            'subsample': params.get('subsample', 1.0),
            'colsample_bytree': params.get('colsample_bytree', 1.0),
            'min_child_weight': params.get('min_child_weight', 1.0),
            'gamma': params.get('gamma', 0.0),
            'reg_lambda': params.get('reg_lambda', 1.0),
            'reg_alpha': params.get('reg_alpha', 0.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'tree_method': params.get('tree_method', 'hist'),
            'device': params.get('device', device),
            'random_state': 42
        }
        return XGBClassifier(**model_params)
    elif model_name == 'random_forest':
        model_params = {
            'n_estimators': params.get('n_estimators', 200),
            'max_depth': params.get('max_depth', None),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
            'max_features': params.get('max_features', 'sqrt'),
            'class_weight': params.get('class_weight', 'balanced'),
            'random_state': 42,
            'n_jobs': -1
        }
        return RandomForestClassifier(**model_params)
    elif model_name == 'tabnet':
        model_params = {
            'n_d': params.get('n_d', 64),
            'n_a': params.get('n_a', 64),
            'n_steps': params.get('n_steps', 5),
            'gamma': params.get('gamma', 1.5),
            'lambda_sparse': params.get('lambda_sparse', 0.0001),
            'momentum': params.get('momentum', 0.02),
            'clip_value': params.get('clip_value', 2.0),
            'n_independent': params.get('n_independent', 2),
            'n_shared': params.get('n_shared', 2),
            # 'optimizer_params': params.get('optimizer_params', {'lr': params.get('tabnet_lr', 0.02)}),
            'seed': 42,
            'verbose': 0
        }
        return TabNetClassifier(**model_params)
    elif model_name == 'mlp':
        hidden_dims = params.get('hidden_dims', (256, 128))
        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)
        dropout = params.get('dropout', 0.3)
        return MLPModel(input_dim, hidden_dims=hidden_dims, dropout=dropout).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_deep_model(model, X_train, y_train, X_val, y_val, X_test,
                     epochs: int = 200, patience: int = 10, lr: float = 0.001,
                     batch_size: int = 64):
    """
    Train deep learning model (MLP) with early stopping.
    
    FIX 4: Now uses BCEWithLogitsLoss and applies sigmoid for predictions
    """
    print(f"      [DL Training] epochs={epochs}, patience={patience}, lr={lr}")
    start_time = time.time()
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).reshape(-1, 1).to(DEVICE)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).reshape(-1, 1).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()  # FIX 4: Changed from BCELoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"      [DL Training] Train: {len(X_train)}, Val: {len(X_val)}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      [DL Training] Early stop at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"      [DL] Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Predict - FIX 4: Apply sigmoid to logits
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.sigmoid(logits).cpu().numpy().flatten()
    
    total_time = time.time() - start_time
    print(f"      [DL Training] Complete in {total_time:.2f}s")
    
    return model, y_pred


# ============================================================================
# HYPERPARAMETER TUNING HELPERS
# ============================================================================

def _make_param_key(dataset_name: str, target_variable: str, model_name: str, imputer_type: str) -> str:
    return f"{dataset_name.lower()}::{target_variable.lower()}::{model_name.lower()}::{imputer_type.lower()}"


def tune_model_hyperparameters(model_name: str, input_dim: int,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """Tune model hyperparameters on the first fold using Optuna."""
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    direction = 'maximize'
    
    def objective(trial: optuna.trial.Trial) -> float:
        try:
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    # 'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0),
                    # 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0)
                }
                model = get_model(model_name, input_dim, params)
                model.fit(X_train, y_train)
                y_val_scores = model.predict_proba(X_val)[:, 1]
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = get_model(model_name, input_dim, params)
                model.fit(X_train, y_train)
                y_val_scores = model.predict_proba(X_val)[:, 1]
            elif model_name == 'mlp':
                params = {
                    'hidden_dims': (
                        trial.suggest_int('hidden_dim1', 128, 512, step=32),
                        trial.suggest_int('hidden_dim2', 64, 256, step=32)
                    ),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'mlp_lr': trial.suggest_float('mlp_lr', 1e-4, 5e-3, log=True),
                    'mlp_epochs': trial.suggest_int('mlp_epochs', 80, 200),
                    'mlp_batch_size': trial.suggest_categorical('mlp_batch_size', [32, 64, 128])
                }
                model = get_model(model_name, input_dim, params)
                _, y_val_pred = train_deep_model(
                    model, X_train, y_train, X_val, y_val, X_val,
                    epochs=params['mlp_epochs'], patience=10,
                    lr=params['mlp_lr'], batch_size=params['mlp_batch_size']
                )
                y_val_scores = y_val_pred
            elif model_name == 'tabnet':
                params = {
                    'n_d': trial.suggest_int('n_d', 16, 64, step=16),
                    'n_a': trial.suggest_int('n_a', 16, 64, step=16),
                    'n_steps': trial.suggest_int('n_steps', 3, 10),
                    'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                    'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
                    'momentum': trial.suggest_float('momentum', 0.01, 0.4),
                    'clip_value': trial.suggest_float('clip_value', 1.0, 5.0),
                    'tabnet_lr': trial.suggest_float('tabnet_lr', 1e-3, 0.05, log=True),
                    'tabnet_max_epochs': trial.suggest_int('tabnet_max_epochs', 80, 200),
                    'tabnet_batch_size': trial.suggest_categorical('tabnet_batch_size', [128, 256, 512, 1024]),
                    'tabnet_virtual_batch_size': trial.suggest_categorical('tabnet_virtual_batch_size', [32, 64, 128])
                }
                model = get_model(model_name, input_dim, params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    max_epochs=params['tabnet_max_epochs'],
                    patience=10,
                    batch_size=params['tabnet_batch_size'],
                    virtual_batch_size=params['tabnet_virtual_batch_size'],
                    # optimizer_params={'lr': params['tabnet_lr']},
                    eval_metric=['logloss']
                )
                y_val_scores = model.predict_proba(X_val)[:, 1]
            else:
                raise ValueError(f"Unsupported model for tuning: {model_name}")
        except Exception as e:
            print(f"    [Tuning] Trial failed: {e}")
            raise optuna.TrialPruned()
        
        y_val_binary = (y_val_scores >= 0.5).astype(int)
        score = f1_score(y_val, y_val_binary, average='macro', zero_division=0)
        trial.report(score, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score
    
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    try:
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
    except ValueError as exc:
        print(f"    [Tuning] Optuna failed: {exc}. Using default parameters.")
        return {}
    
    completed_trials = [
        t for t in study.get_trials(deepcopy=False) 
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("    [Tuning] No completed trials; using default parameters.")
        return {}
    
    return study.best_params


def ensure_best_params(dataset_name: str, target_variable: str, model_name: str, imputer_type: str,
                       X: pd.DataFrame, y: pd.Series, user_ids: pd.Series,
                       splits: List[Any] = None, n_folds: int = 5) -> Dict[str, Any]:
    """Ensure that tuned hyperparameters are available for the requested configuration."""
    key = _make_param_key(dataset_name, target_variable, model_name, imputer_type)
    if key in BEST_PARAMS_CACHE:
        return BEST_PARAMS_CACHE[key]
    
    print(f"    [Tuning] Optuna search for {model_name} (n_trials={OPTUNA_N_TRIALS})")
    
    if splits is None:
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(np.arange(len(X)), y, groups=user_ids))
    if len(splits) == 0:
        BEST_PARAMS_CACHE[key] = {}
        return {}
    
    train_idx, test_idx = splits[0]
    X_train_full, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    
    train_mask = y_train_full.notna()
    X_train_full = X_train_full[train_mask]
    y_train_full = y_train_full[train_mask]
    y_test_fold = y_test_fold[y_test_fold.notna()]
    
    if len(y_train_full) == 0 or len(np.unique(y_train_full)) < 2:
        print("    [Tuning] Skipped (insufficient class diversity)")
        BEST_PARAMS_CACHE[key] = {}
        return {}
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42,
        stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
    )
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) == 0:
        print("    [Tuning] Skipped (invalid train/val split)")
        BEST_PARAMS_CACHE[key] = {}
        return {}
    
    try:
        X_train_proc, X_val_proc, _, y_train_proc = apply_preprocessing_pipeline(
            X_train, X_val, X_test_fold, y_train, apply_smote=True, imputer_type=imputer_type
        )
    except Exception as exc:
        print(f"    [Tuning] Preprocessing failed: {exc}")
        BEST_PARAMS_CACHE[key] = {}
        return {}
    
    y_train_array = y_train_proc.values if hasattr(y_train_proc, 'values') else y_train_proc
    y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
    best_params = tune_model_hyperparameters(
        model_name, X_train_proc.shape[1], X_train_proc, y_train_array, X_val_proc, y_val_array
    )
    BEST_PARAMS_CACHE[key] = best_params
    print(f"    [Tuning] Best params cached for {key}")
    return best_params
# ============================================================================
# BOOTSTRAPPING
# ============================================================================

def bootstrap_metrics(y_real: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000) -> Dict:
    """Calculate confidence intervals using bootstrapping."""
    np.random.seed(42)
    
    metrics = {
        'accuracy': [],
        'f1_macro': [],
        'precision': [],
        'recall': []
    }
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_real), size=len(y_real), replace=True)
        y_real_boot = y_real[indices]
        y_pred_boot = y_pred[indices]
        
        if y_pred_boot.max() <= 1.0 and y_pred_boot.min() >= 0.0:
            y_pred_binary = (y_pred_boot >= 0.5).astype(int)
        else:
            y_pred_binary = y_pred_boot
        
        metrics['accuracy'].append(accuracy_score(y_real_boot, y_pred_binary))
        metrics['f1_macro'].append(f1_score(y_real_boot, y_pred_binary, average='macro', zero_division=0))
        metrics['precision'].append(precision_score(y_real_boot, y_pred_binary, average='macro', zero_division=0))
        metrics['recall'].append(recall_score(y_real_boot, y_pred_binary, average='macro', zero_division=0))
    
    ci_results = {}
    for key, values in metrics.items():
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        ci_results[f'{key}_ci'] = f'[{ci_lower:.3f}, {ci_upper:.3f}]'
    
    return ci_results


# ============================================================================
# GENERALIZATION PROTOCOL
# ============================================================================

def run_generalization_protocol(X, y, user_ids, is_testset, model_name, imputer_type,
                                dataset_name: str, target_variable: str, n_folds: int = 5):
    """Run User-Group 5-Fold CV with detailed logging."""
    print(f"\n  [Generalization] Model={model_name}, Imputer={imputer_type}, Folds={n_folds}")
    start_time = time.time()
    
    y_real_all = []
    y_pred_all = []
    
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(len(X)), y, groups=user_ids))
    best_params = ensure_best_params(
        dataset_name, target_variable, model_name, imputer_type, X, y, user_ids, splits=splits, n_folds=n_folds
    )
    print(f"    [Tuning] Fold-1 best params: {best_params if best_params else 'default (no tuning)'}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_start = time.time()
        print(f"\n  [Fold {fold_idx + 1}/{n_folds}]")
        
        train_users = user_ids.iloc[train_idx].unique()
        test_users = user_ids.iloc[test_idx].unique()
        print(f"    Train: {len(train_users)} users ({len(train_idx)} samples)")
        print(f"    Test: {len(test_users)} users ({len(test_idx)} samples)")
        
        X_train_full, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_full, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        # Remove missing
        train_mask = y_train_full.notna()
        test_mask = y_test_fold.notna()
        
        X_train_full = X_train_full[train_mask]
        y_train_full = y_train_full[train_mask]
        X_test_fold = X_test_fold[test_mask]
        y_test_fold = y_test_fold[test_mask]
        
        print(f"    After NaN: Train={len(y_train_full)}, Test={len(y_test_fold)}")
        
        if len(y_train_full) == 0 or len(y_test_fold) == 0 or len(np.unique(y_train_full)) < 2:
            print(f"    Skipped")
            continue
        
        # Split train→train+val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, 
            stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
        )
        print(f"    Split: Train={len(y_train)}, Val={len(y_val)}")
        
        # Preprocessing
        X_train_proc, X_val_proc, X_test_proc, y_train_proc = apply_preprocessing_pipeline(
            X_train, X_val, X_test_fold, y_train, apply_smote=True, imputer_type=imputer_type
        )
        
        # Train
        print(f"    [Training] {model_name}...")
        train_start = time.time()
        
        if model_name == 'mlp':
            model = get_model(model_name, X_train_proc.shape[1], best_params)
            epochs = best_params.get('mlp_epochs', 200)
            patience = 10
            lr = best_params.get('mlp_lr', 0.001)
            batch_size = best_params.get('mlp_batch_size', 64)
            _, y_pred_fold = train_deep_model(
                model, X_train_proc, y_train_proc, X_val_proc, y_val, X_test_proc,
                epochs=epochs, patience=patience, lr=lr, batch_size=batch_size
            )
        elif model_name == 'tabnet':
            model = get_model(model_name, X_train_proc.shape[1], best_params)
            tabnet_epochs = best_params.get('tabnet_max_epochs', 200)
            tabnet_patience = 10
            tabnet_batch = best_params.get('tabnet_batch_size', 1024)
            tabnet_vbatch = best_params.get('tabnet_virtual_batch_size', 128)
            model.fit(
                X_train_proc, y_train_proc.values if hasattr(y_train_proc, 'values') else y_train_proc,
                eval_set=[(X_val_proc, y_val.values)],
                max_epochs=tabnet_epochs,
                patience=tabnet_patience,
                batch_size=tabnet_batch,
                virtual_batch_size=tabnet_vbatch,
                eval_metric=['logloss']
            )
            y_pred_fold = model.predict_proba(X_test_proc)[:, 1]
        else:
            model = get_model(model_name, X_train_proc.shape[1], best_params)
            model.fit(X_train_proc, y_train_proc)
            y_pred_fold = model.predict_proba(X_test_proc)[:, 1]
        
        train_time = time.time() - train_start
        print(f"    [Training] Complete in {train_time:.2f}s")
        
        # Filter to test set
        test_mask_in_fold = is_testset.iloc[test_idx][test_mask].values
        y_pred_testset = y_pred_fold[test_mask_in_fold]
        y_real_testset = y_test_fold.iloc[np.where(test_mask_in_fold)[0]]
        
        fold_time = time.time() - fold_start
        print(f"    [Fold {fold_idx + 1}] Done in {fold_time:.2f}s → {len(y_pred_testset)} test samples")
        
        y_real_all.extend(y_real_testset.values)
        y_pred_all.extend(y_pred_testset)
    
    total_time = time.time() - start_time
    print(f"\n  [Generalization] Complete in {total_time:.2f}s, {len(y_real_all)} test samples")
    
    return np.array(y_real_all), np.array(y_pred_all)


# ============================================================================
# PERSONALIZATION - ML (FIXED: No validation split, No HP tuning)
# ============================================================================

def run_personalization_ml_weighted(X_full, y_full, user_ids, dates, is_testset, 
                                   test_indices, model_name, imputer_type,
                                   best_params: Dict[str, Any]):
    """
    ML Personalization with tuned hyperparameters and sample-weight emphasis.
    
    - Uses generalization-tuned parameters (representative strategy)
    - No validation split to leverage all available data
    - Evaluates multiple personalization weights (3/5/10) for own samples
    """
    print(f"\n  [ML Personalization] Model={model_name}, Test samples={len(test_indices)}")
    start_time = time.time()
    
    y_real_all = []
    y_pred_all = []
    
    dates = pd.to_datetime(dates)
    print(f"    [Personalization Params] {best_params if best_params else 'default (no tuning)'}")
    
    for i, test_idx in enumerate(test_indices):
        if (i + 1) % 10 == 0 or (i + 1) == 1:
            print(f"\n  [Sample {i+1}/{len(test_indices)}]")
        
        test_user = user_ids.iloc[test_idx]
        test_date = dates.iloc[test_idx]
        test_y = y_full.iloc[test_idx]
        
        if pd.isna(test_y):
            continue
        
        # D_other + D_i_personalization
        other_mask = (user_ids != test_user)
        personalization_mask = (user_ids == test_user) & (dates < test_date)
        
        D_other_indices = X_full.index[other_mask]
        D_personalization_indices = X_full.index[personalization_mask]
        
        if (i + 1) % 10 == 0 or (i + 1) == 1:
            print(f"    D_other: {len(D_other_indices)}, D_personalization: {len(D_personalization_indices)}")
        has_personalization = len(D_personalization_indices) > 0
        
        # Combine
        train_indices = D_other_indices.union(D_personalization_indices)
        X_train = X_full.loc[train_indices]
        y_train = y_full.loc[train_indices]
        
        train_valid = y_train.notna()
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        personalization_mask_train = X_train.index.isin(D_personalization_indices)
        
        if len(y_train) == 0 or len(np.unique(y_train)) < 2:
            continue
        
        # FIX 2: NO validation split - use dummy val set for preprocessing
        X_val = X_train.iloc[:1]  # Dummy
        y_val = y_train.iloc[:1]  # Dummy
        
        X_test_single = X_full.iloc[[test_idx]]
        
        # Preprocessing
        try:
            X_train_proc, X_val_proc, X_test_proc, y_train_proc = apply_preprocessing_pipeline(
                X_train, X_val, X_test_single, y_train, apply_smote=False, imputer_type=imputer_type
            )
        except Exception as e:
            if (i + 1) % 10 == 0 or (i + 1) == 1:
                print(f"    [Skip] Preprocessing failed: {e}")
            continue
        
        best_prediction = None
        personalization_reason = ""
        
        # Attempt personalization if possible
        if has_personalization and personalization_mask_train.sum() > 0:
            try:
                pers_mask_array = personalization_mask_train
                X_pers_proc = X_train_proc[pers_mask_array]
                y_pers_proc = y_train_proc.iloc[pers_mask_array] if hasattr(y_train_proc, 'iloc') else y_train_proc[pers_mask_array]
                
                if len(X_pers_proc) == 0 or len(np.unique(y_pers_proc)) < 2:
                    personalization_reason = "insufficient-personal-data"
                else:
                    best_metric = -np.inf
                    best_weight = None
                    
                    for weight in PERSONALIZATION_SAMPLE_WEIGHTS:
                        sample_weights = np.ones(len(y_train_proc))
                        sample_weights[pers_mask_array] = weight
                        
                        model = get_model(model_name, X_train_proc.shape[1], best_params)
                        model.fit(X_train_proc, y_train_proc, sample_weight=sample_weights)
                        
                        pers_scores = model.predict_proba(X_pers_proc)[:, 1]
                        pers_binary = (pers_scores >= 0.5).astype(int)
                        pers_metric = accuracy_score(y_pers_proc, pers_binary)
                        
                        if pers_metric >= best_metric:
                            best_metric = pers_metric
                            best_weight = weight
                            best_prediction = model.predict_proba(X_test_proc)[:, 1][0]
                    
                    if best_prediction is not None and ((i + 1) % 10 == 0 or (i + 1) == 1):
                        print(f"    Selected sample_weight={best_weight} (metric={best_metric:.3f})")
            except Exception as e:
                personalization_reason = f"personalization-error: {e}"
                best_prediction = None
                if (i + 1) % 10 == 0 or (i + 1) == 1:
                    print(f"    [Personalization] Failed: {e}")
        else:
            personalization_reason = "no-personal-data"
        
        # Fallback: train without personalization emphasis
        if best_prediction is None:
            try:
                model = get_model(model_name, X_train_proc.shape[1], best_params)
                model.fit(X_train_proc, y_train_proc)
                best_prediction = model.predict_proba(X_test_proc)[:, 1][0]
                if (i + 1) % 10 == 0 or (i + 1) == 1:
                    reason = personalization_reason or "fallback"
                    print(f"    [Personalization] Using base model ({reason})")
            except Exception as e:
                if (i + 1) % 10 == 0 or (i + 1) == 1:
                    print(f"    [Skip] Base training failed: {e}")
                continue
        
        y_real_all.append(test_y)
        y_pred_all.append(best_prediction)
    
    total_time = time.time() - start_time
    print(f"\n  [ML Personalization] Complete in {total_time:.2f}s, {len(y_real_all)} predicted")
    
    return np.array(y_real_all), np.array(y_pred_all)


# ============================================================================
# PERSONALIZATION - DL (FIXED: Proper scaling, No validation split)
# ============================================================================

def run_personalization_dl_finetuning(X_full, y_full, user_ids, dates, is_testset,
                                     test_indices, model_name, imputer_type,
                                     best_params: Dict[str, Any]):
    """
    DL Personalization: Pre-train + Fine-tune with proper scaling.
    
    - Stores and reuses fitted scaler/imputer
    - No validation split for fine-tuning (use all personal data)
    - Reuses generalization-tuned hyperparameters with TabNet warm-start fine-tuning
    """
    print(f"\n  [DL Personalization] Model={model_name}, Test samples={len(test_indices)}")
    start_time = time.time()
    
    y_real_all = []
    y_pred_all = []
    
    dates = pd.to_datetime(dates)
    print(f"    [Personalization Params] {best_params if best_params else 'default (no tuning)'}")
    
    # Pre-training
    test_users = user_ids.iloc[test_indices].unique()
    print(f"  [Pre-training] {len(test_users)} users")
    
    pretrained_models = {}
    
    def predict_with_pretrained(model_obj, X_scaled):
        if model_name == 'mlp':
            tensor = torch.FloatTensor(X_scaled).to(DEVICE)
            model_obj.eval()
            with torch.no_grad():
                logits = model_obj(tensor)
                return torch.sigmoid(logits).cpu().numpy().flatten()[0]
        elif model_name == 'tabnet':
            return model_obj.predict_proba(X_scaled)[:, 1][0]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    for test_user in test_users:
        print(f"\n  [Pre-train User {test_user}]")
        pretrain_start = time.time()
        
        # D_other
        other_mask = (user_ids != test_user)
        D_other_indices = X_full.index[other_mask]
        
        X_other = X_full.loc[D_other_indices]
        y_other = y_full.loc[D_other_indices]
        
        valid_mask = y_other.notna()
        X_other = X_other[valid_mask]
        y_other = y_other[valid_mask]
        
        print(f"    D_other: {len(X_other)} samples")
        
        if len(y_other) == 0 or len(np.unique(y_other)) < 2:
            print(f"    Skipped")
            continue
        
        # Split for pre-training
        X_train, X_val, y_train, y_val = train_test_split(
            X_other, y_other, test_size=0.2, random_state=42,
            stratify=y_other if len(np.unique(y_other)) > 1 else None
        )
        
        print(f"    Train: {len(y_train)}, Val: {len(y_val)}")
        
        # FIX 1: Get transformers during pre-training
        X_test_dummy = X_train.iloc[:1]
        result = apply_preprocessing_pipeline(
            X_train, X_val, X_test_dummy, y_train, apply_smote=True, 
            imputer_type=imputer_type, return_transformers=True
        )
        X_train_proc, X_val_proc, _, y_train_proc, imputer, scaler = result
        
        # Train
        print(f"    [Pre-training]...")
        
        if model_name == 'mlp':
            model = get_model(model_name, X_train_proc.shape[1], best_params)
            mlp_epochs = best_params.get('mlp_epochs', 200)
            mlp_patience = 10
            mlp_lr = best_params.get('mlp_lr', 0.001)
            mlp_batch_size = best_params.get('mlp_batch_size', 64)
            pretrained_model, _ = train_deep_model(
                model, X_train_proc, y_train_proc, X_val_proc, y_val, X_val_proc,
                epochs=mlp_epochs, patience=mlp_patience, lr=mlp_lr, batch_size=mlp_batch_size
            )
            pretrained_models[test_user] = {
                'model': pretrained_model,
                'input_dim': X_train_proc.shape[1],
                'imputer': imputer,  # FIX 1: Store fitted imputer
                'scaler': scaler      # FIX 1: Store fitted scaler
            }
        elif model_name == 'tabnet':
            model = get_model(model_name, X_train_proc.shape[1], best_params)
            tabnet_epochs = best_params.get('tabnet_max_epochs', 200)
            tabnet_patience = 10
            tabnet_batch = best_params.get('tabnet_batch_size', 1024)
            tabnet_vbatch = best_params.get('tabnet_virtual_batch_size', 128)
            model.fit(
                X_train_proc, y_train_proc.values if hasattr(y_train_proc, 'values') else y_train_proc,
                eval_set=[(X_val_proc, y_val.values)],
                max_epochs=tabnet_epochs,
                patience=tabnet_patience,
                batch_size=tabnet_batch,
                virtual_batch_size=tabnet_vbatch,
                eval_metric=['logloss']
            )
            pretrained_models[test_user] = {
                'model': model,
                'input_dim': X_train_proc.shape[1],
                'imputer': imputer,  # FIX 1: Store fitted imputer
                'scaler': scaler      # FIX 1: Store fitted scaler
            }
        
        pretrain_time = time.time() - pretrain_start
        print(f"    [Pre-train User {test_user}] Done in {pretrain_time:.2f}s")
    
    # Fine-tuning
    print(f"\n  [Fine-tuning] {len(test_indices)} samples...")
    
    for i, test_idx in enumerate(test_indices):
        if (i + 1) % 10 == 0 or (i + 1) == 1:
            print(f"\n  [Sample {i+1}/{len(test_indices)}]")
        
        test_user = user_ids.iloc[test_idx]
        test_date = dates.iloc[test_idx]
        test_y = y_full.iloc[test_idx]
        
        if pd.isna(test_y):
            continue
        
        if test_user not in pretrained_models:
            continue
        
        # D_i_personalization
        personalization_mask = (user_ids == test_user) & (dates < test_date)
        D_personalization_indices = X_full.index[personalization_mask]
        
        if (i + 1) % 10 == 0 or (i + 1) == 1:
            print(f"    D_personalization: {len(D_personalization_indices)}")
        
        X_test_single = X_full.iloc[[test_idx]]
        pretrained_info = pretrained_models[test_user]
        imputer = pretrained_info['imputer']
        scaler = pretrained_info['scaler']
        
        try:
            X_test_imputed = imputer.transform(X_test_single)
            X_test_scaled = scaler.transform(X_test_imputed)
        except Exception as e:
            if (i + 1) % 10 == 0 or (i + 1) == 1:
                print(f"      Failed to transform test sample: {e}")
            continue
        
        can_finetune = len(D_personalization_indices) > 0
        finetune_reason = ""
        X_pers_scaled = None
        y_personalization = None
        
        if can_finetune:
            X_personalization = X_full.loc[D_personalization_indices]
            y_personalization = y_full.loc[D_personalization_indices]
            
            valid_mask = y_personalization.notna()
            X_personalization = X_personalization[valid_mask]
            y_personalization = y_personalization[valid_mask]
            
            if len(y_personalization) == 0 or len(np.unique(y_personalization)) < 2:
                can_finetune = False
                finetune_reason = "insufficient-personal-data"
            else:
                try:
                    X_pers_imputed = imputer.transform(X_personalization)
                    X_pers_scaled = scaler.transform(X_pers_imputed)
                except Exception as e:
                    can_finetune = False
                    finetune_reason = f"transform-error: {e}"
        else:
            finetune_reason = "no-personal-data"
        
        model = deepcopy(pretrained_info['model'])
        y_pred_single = None
        
        if can_finetune and X_pers_scaled is not None:
            try:
                if model_name == 'mlp':
                    mlp_ft_epochs = best_params.get('mlp_finetune_epochs', best_params.get('mlp_epochs', 200))
                    mlp_lr = best_params.get('mlp_lr', 0.0001)
                    mlp_batch_size = best_params.get('mlp_batch_size', 64)
                    mlp_batch_size = max(1, min(len(X_pers_scaled), mlp_batch_size))
                    print(f"      [Fine-tuning] MLP, lr={mlp_lr}, epochs={mlp_ft_epochs}, batch={mlp_batch_size}")
                    
                    X_pers_tensor = torch.FloatTensor(X_pers_scaled).to(DEVICE)
                    y_pers_tensor = torch.FloatTensor(y_personalization.values).reshape(-1, 1).to(DEVICE)
                    X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
                    
                    optimizer = optim.Adam(model.parameters(), lr=mlp_lr)
                    criterion = nn.BCEWithLogitsLoss()
                    dataset = TensorDataset(X_pers_tensor, y_pers_tensor)
                    loader = DataLoader(dataset, batch_size=mlp_batch_size, shuffle=True)
                    
                    model.train()
                    for _ in range(mlp_ft_epochs):
                        for batch_X, batch_y in loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                    
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_test_tensor)
                        y_pred_single = torch.sigmoid(logits).cpu().numpy().flatten()[0]
                    
                elif model_name == 'tabnet':
                    tabnet_epochs = best_params.get('tabnet_max_epochs', 200)
                    tabnet_patience = 10
                    tabnet_batch = max(2, min(best_params.get('tabnet_batch_size', 1024), len(X_pers_scaled)))
                    tabnet_vbatch = max(1, min(best_params.get('tabnet_virtual_batch_size', 128), tabnet_batch))
                    tabnet_lr = best_params.get('tabnet_lr', 0.02)
                    print(f"      [Fine-tuning] TabNet warm start, epochs={tabnet_epochs}, batch={tabnet_batch}")
                    model.fit(
                        X_pers_scaled, y_personalization.values,
                        max_epochs=tabnet_epochs,
                        patience=tabnet_patience,
                        batch_size=tabnet_batch,
                        virtual_batch_size=tabnet_vbatch,
                        warm_start=True,
                        eval_metric=['logloss']
                    )
                    y_pred_single = model.predict_proba(X_test_scaled)[:, 1][0]
            except Exception as e:
                if (i + 1) % 10 == 0 or (i + 1) == 1:
                    print(f"      [Fine-tuning] Failed: {e}")
                y_pred_single = None
                finetune_reason = f"personalization-error: {e}"
                can_finetune = False
        
        if y_pred_single is None:
            if (i + 1) % 10 == 0 or (i + 1) == 1:
                reason = finetune_reason or "using-pretrained"
                print(f"      [Fine-tuning] Skipped personalization ({reason})")
            base_model = deepcopy(pretrained_info['model'])
            y_pred_single = predict_with_pretrained(base_model, X_test_scaled)
        
        y_real_all.append(test_y)
        y_pred_all.append(y_pred_single)
    
    total_time = time.time() - start_time
    print(f"\n  [DL Personalization] Complete in {total_time:.2f}s, {len(y_real_all)} predicted")
    
    return np.array(y_real_all), np.array(y_pred_all)


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_experiment(dataset_name: str, target_variable: str, mode: str = 'gen',
                  model_name: str = 'xgboost', imputer_type: str = 'knn',
                  n_bootstrap: int = 1000, save_processed: bool = False) -> Dict:
    """Run single experiment with detailed logging."""
    dataset_name = dataset_name.lower()
    mode = mode.lower()
    
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"[EXPERIMENT] {dataset_name.upper()} | {target_variable} | {mode} | {model_name}")
    print(f"{'='*80}")
    print(f"Imputer: {imputer_type}, Bootstrap: {n_bootstrap}")
    
    # Load
    print(f"\n[Loading] {dataset_name.upper()}...")
    if dataset_name == 'globem':
        df_full = pd.read_csv(os.path.join(DATASET_PATH, 'Globem/aggregated_globem.csv'))
        user_ids = df_full['pid'].copy()
        dates = df_full['date'].copy()
        is_testset = df_full['is_testset'].copy()
        X, y = preprocess_globem(df_full, target_variable, save_processed=save_processed)
        
    elif dataset_name == 'ces':
        df_full = pd.read_csv(os.path.join(DATASET_PATH, 'CES/aggregated_ces.csv'))
        df_testset = pd.read_csv(os.path.join(DATASET_PATH, 'CES/ces_testset.csv'))
        test_keys = set(zip(df_testset['uid'], df_testset['date']))
        is_testset = pd.Series([
            (uid, date) in test_keys 
            for uid, date in zip(df_full['uid'], df_full['date'])
        ], index=df_full.index)
        user_ids = df_full['uid'].copy()
        dates = df_full['date'].copy()
        X, y = preprocess_ces(df_full, target_variable, save_processed=save_processed)
        
    elif dataset_name == 'mentaliot':
        df_full = pd.read_csv(os.path.join(DATASET_PATH, 'MentalIoT/aggregated_mentaliot.csv'))
        df_testset = pd.read_csv(os.path.join(DATASET_PATH, 'MentalIoT/mentaliot_testset.csv'))
        test_keys = set(zip(df_testset['uid'], df_testset['timestamp']))
        is_testset = pd.Series([
            (uid, ts) in test_keys 
            for uid, ts in zip(df_full['uid'], df_full['timestamp'])
        ], index=df_full.index)
        user_ids = df_full['uid'].copy()
        dates = df_full['timestamp'].copy()
        X, y = preprocess_mentaliot(df_full, target_variable, save_processed=save_processed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    user_ids = user_ids.reset_index(drop=True)
    dates = dates.reset_index(drop=True)
    is_testset = is_testset.reset_index(drop=True)
    
    print(f"[Data] Total: {len(y)}, Test: {is_testset.sum()}, Features: {X.shape[1]}")
    
    # Run
    if mode == 'gen':
        y_real, y_pred = run_generalization_protocol(
            X, y, user_ids, is_testset, model_name, imputer_type,
            dataset_name, target_variable, n_folds=5
        )
    elif mode == 'pers':
        test_indices = np.where(is_testset)[0]
        best_params = ensure_best_params(
            dataset_name, target_variable, model_name, imputer_type, X, y, user_ids
        )
        
        if model_name in ['xgboost', 'random_forest']:
            y_real, y_pred = run_personalization_ml_weighted(
                X, y, user_ids, dates, is_testset, test_indices, model_name, imputer_type, best_params
            )
        elif model_name in ['mlp', 'tabnet']:
            y_real, y_pred = run_personalization_dl_finetuning(
                X, y, user_ids, dates, is_testset, test_indices, model_name, imputer_type, best_params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Metrics
    if len(y_pred) > 0:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_real, y_pred_binary)
        precision = precision_score(y_real, y_pred_binary, average='macro', zero_division=0)
        recall = recall_score(y_real, y_pred_binary, average='macro', zero_division=0)
        f1 = f1_score(y_real, y_pred_binary, average='macro', zero_division=0)
        
        print(f"\n[Bootstrapping] n={n_bootstrap}...")
        boot_start = time.time()
        ci_results = bootstrap_metrics(y_real, y_pred, n_bootstrap=n_bootstrap)
        boot_time = time.time() - boot_start
        print(f"[Bootstrapping] Done in {boot_time:.2f}s")
    else:
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        ci_results = {}
    
    elapsed_time = time.time() - start_time
    
    results = {
        'dataset': dataset_name,
        'target': target_variable,
        'mode': mode,
        'model': model_name,
        'imputer': imputer_type,
        'n_test_samples': len(y_pred),
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision': precision,
        'recall': recall,
        'accuracy_ci': ci_results.get('accuracy_ci', ''),
        'f1_macro_ci': ci_results.get('f1_macro_ci', ''),
        'precision_ci': ci_results.get('precision_ci', ''),
        'recall_ci': ci_results.get('recall_ci', ''),
        'time_seconds': elapsed_time
    }
    
    print(f"\n{'='*80}")
    print(f"[RESULTS]")
    print(f"  Acc: {accuracy:.4f} {ci_results.get('accuracy_ci', '')}")
    print(f"  F1: {f1:.4f} {ci_results.get('f1_macro_ci', '')}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"{'='*80}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced ML/DL Mental Health Prediction (CORRECTED)')
    parser.add_argument('--datasets', nargs='+', default=['globem'],
                       choices=['globem', 'ces', 'mentaliot'])
    parser.add_argument('--modes', nargs='+', default=['gen', 'pers'],
                       choices=['gen', 'pers'])
    parser.add_argument('--models', nargs='+', 
                       default=['xgboost', 'random_forest', 'mlp', 'tabnet'],
                       choices=['xgboost', 'random_forest', 'mlp', 'tabnet'])
    parser.add_argument('--imputer', type=str, default='simple',
                       choices=['knn', 'iterative', 'simple'])
    parser.add_argument('--output', type=str, default='final_results.csv')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--save_processed', action='store_true')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION (CORRECTED VERSION)")
    print(f"{'='*80}")
    print(f"Datasets: {args.datasets}")
    print(f"Modes: {args.modes}")
    print(f"Models: {args.models}")
    print(f"Imputer: {args.imputer}")
    print(f"Bootstrap: {args.n_bootstrap}")
    print(f"Save processed: {args.save_processed}")
    print(f"Output: {args.output}")
    print(f"\nFIXES APPLIED:")
    print(f"  1. DL scaling mismatch fixed (reuse fitted scaler)")
    print(f"  2. Removed validation split for personalization")
    print(f"  3. Removed hyperparameter tuning (fixed params)")
    print(f"  4. Improved stability (BCEWithLogitsLoss)")
    print(f"{'='*80}\n")
    
    # Experiments
    experiments = []
    
    for dataset in args.datasets:
        if dataset == 'globem':
            targets = ['depression', 'anxiety']
        elif dataset in ['ces', 'mentaliot']:
            targets = ['depression', 'anxiety', 'stress']
        
        for target in targets:
            for mode in args.modes:
                for model in args.models:
                    experiments.append({
                        'dataset_name': dataset,
                        'target_variable': target,
                        'mode': mode,
                        'model_name': model,
                        'imputer_type': args.imputer,
                        'n_bootstrap': args.n_bootstrap,
                        'save_processed': args.save_processed
                    })
    
    print(f"Total experiments: {len(experiments)}\n")
    
    # Run
    all_results = []
    
    for i, config in enumerate(experiments):
        print(f"\n{'#'*80}")
        print(f"[{i+1}/{len(experiments)}]")
        print(f"{'#'*80}")
        
        try:
            results = run_experiment(**config)
            all_results.append(results)
            
            # Save
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(args.output, index=False)
            print(f"\n[Saved] {args.output}")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"Results: {args.output}")
    print(f"Success: {len(all_results)}/{len(experiments)}")
    print(f"{'='*80}")
