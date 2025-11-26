"""
anomaly_ml.py - Part 3.2: ML-Based Anomaly Classification (FIXED)
================================================================

Fixed: Sampling error when too few positives available
"""

import os
import pandas as pd
import numpy as np
import yaml
import json
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = 'src/config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_silver_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create silver labels based on z-score and prediction intervals
    
    Criteria:
    - Positive (1): |z| ≥ 3.5 OR (y_true outside [lo, hi] AND |z| ≥ 2.5)
    - Negative (0): |z| < 1.0 AND y_true inside [lo, hi]
    - Unlabeled (NaN): Everything else
    """
    labels = pd.Series(np.nan, index=df.index)
    
    # Positive labels
    outside_pi = (df['y_true'] < df['lo']) | (df['y_true'] > df['hi'])
    positive_condition = (
        (np.abs(df['z_resid']) >= 3.5) | 
        (outside_pi & (np.abs(df['z_resid']) >= 2.5))
    )
    labels[positive_condition] = 1
    
    # Negative labels
    inside_pi = (df['y_true'] >= df['lo']) & (df['y_true'] <= df['hi'])
    negative_condition = (np.abs(df['z_resid']) < 1.0) & inside_pi
    labels[negative_condition] = 0
    
    return labels


def sample_for_verification(
    df: pd.DataFrame, 
    n_samples: int = 100, 
    n_positive: int = 50,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample timestamps for human verification (FIXED VERSION)
    
    Args:
        df: DataFrame with silver_label column
        n_samples: Total samples (default 100)
        n_positive: Target number of positive samples (default 50)
        random_state: Random seed
    
    Returns:
        Sampled DataFrame
    """
    labeled = df[df['silver_label'].notna()].copy()
    
    positives = labeled[labeled['silver_label'] == 1]
    negatives = labeled[labeled['silver_label'] == 0]
    
    # ✅ FIX: Ensure n_positive doesn't exceed n_samples
    n_positive = min(n_positive, n_samples)
    
    # ✅ FIX: Adjust if not enough positives available
    if len(positives) < n_positive:
        print(f"   ⚠️  Only {len(positives)} positive samples available (target: {n_positive})")
        n_positive = len(positives)
    
    # ✅ FIX: Calculate n_neg ensuring it's positive and available
    n_neg = min(n_samples - n_positive, len(negatives))
    n_neg = max(0, n_neg)
    
    print(f"   Sampling: {n_positive} positive + {n_neg} negative = {n_positive + n_neg} total")
    
    # Sample
    pos_sample = positives.sample(n=n_positive, random_state=random_state) if n_positive > 0 else pd.DataFrame()
    neg_sample = negatives.sample(n=n_neg, random_state=random_state) if n_neg > 0 else pd.DataFrame()
    
    sample = pd.concat([pos_sample, neg_sample], ignore_index=False)
    
    if len(sample) > 0:
        sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return sample


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Create features for ML classifier
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Lag features
    df['y_lag_1'] = df['y_true'].shift(1)
    df['y_lag_24'] = df['y_true'].shift(24)
    df['y_lag_48'] = df['y_true'].shift(48)
    
    # Rolling statistics (24h window)
    df['y_roll_mean_24'] = df['y_true'].rolling(window=24, min_periods=1).mean()
    df['y_roll_std_24'] = df['y_true'].rolling(window=24, min_periods=1).std()
    df['y_roll_min_24'] = df['y_true'].rolling(window=24, min_periods=1).min()
    df['y_roll_max_24'] = df['y_true'].rolling(window=24, min_periods=1).max()
    
    # Calendar features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    # Forecast context
    df['abs_residual'] = np.abs(df['residual'])
    df['abs_z_resid'] = np.abs(df['z_resid'])
    
    # Feature columns
    feature_cols = [
        'y_lag_1', 'y_lag_24', 'y_lag_48',
        'y_roll_mean_24', 'y_roll_std_24', 'y_roll_min_24', 'y_roll_max_24',
        'hour', 'day_of_week',
        'residual', 'abs_residual', 'z_resid', 'abs_z_resid'
    ]
    
    return df, feature_cols


def train_anomaly_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[object, Dict]:
    """
    Train logistic regression classifier
    """
    # Filter to verified labels only
    df_labeled = df[df['verified_label'].notna()].copy()
    
    if len(df_labeled) < 10:
        print(f"   ⚠️  Not enough labeled samples ({len(df_labeled)}). Need at least 10.")
        return None, None
    
    X = df_labeled[feature_cols].fillna(0)
    y = df_labeled['verified_label'].astype(int)
    
    print(f"   Total labeled samples: {len(y)}")
    print(f"   Positive: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"   Negative: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
    
    # Check if we have both classes
    if y.nunique() < 2:
        print(f"   ⚠️  Only one class present. Cannot train classifier.")
        return None, None
    
    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        print(f"   ⚠️  Cannot stratify split: {e}")
        print(f"   Using non-stratified split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Train logistic regression
    print(f"   Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000, 
        class_weight='balanced', 
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # F1 at fixed precision (P=0.80)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find threshold closest to P=0.80
    target_precision = 0.80
    idx = np.argmin(np.abs(precision - target_precision))
    threshold_at_p80 = thresholds[idx] if idx < len(thresholds) else 0.5
    
    y_pred_at_p80 = (y_pred_proba >= threshold_at_p80).astype(int)
    f1_at_p80 = f1_score(y_test, y_pred_at_p80)
    precision_at_p80 = precision[idx]
    recall_at_p80 = recall[idx]
    
    metrics = {
        'pr_auc': float(pr_auc),
        'f1_at_p80': float(f1_at_p80),
        'precision_at_p80': float(precision_at_p80),
        'recall_at_p80': float(recall_at_p80),
        'threshold_at_p80': float(threshold_at_p80),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_positive_test': int(y_test.sum())
    }
    
    print(f"\n   Metrics:")
    print(f"      PR-AUC: {pr_auc:.4f}")
    print(f"      F1 @ P=0.80: {f1_at_p80:.4f}")
    print(f"      Precision @ P=0.80: {precision_at_p80:.4f}")
    print(f"      Recall @ P=0.80: {recall_at_p80:.4f}")
    
    return clf, metrics


def process_country_ml_anomalies(
    cc: str,
    anomalies_path: str,
    forecasts_path: str,
    output_folder: str = 'outputs',
    config: dict = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process ML-based anomaly detection for one country
    """
    print(f"\n{'='*80}")
    print(f"ML Anomaly Detection for: {cc}")
    print(f"{'='*80}")
    
    # Load anomalies and forecasts
    df_anom = pd.read_csv(anomalies_path)
    df_fc = pd.read_csv(forecasts_path)
    
    # Merge
    df = pd.merge(
        df_anom[['timestamp', 'y_true', 'yhat', 'z_resid']],
        df_fc[['timestamp', 'lo', 'hi']],
        on='timestamp',
        how='inner'
    )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['residual'] = df['y_true'] - df['yhat']
    
    print(f"Loaded {len(df):,} samples")
    
    # Create silver labels
    print(f"Creating silver labels...")
    df['silver_label'] = create_silver_labels(df)
    
    n_positive = df['silver_label'].eq(1).sum()
    n_negative = df['silver_label'].eq(0).sum()
    n_unlabeled = df['silver_label'].isna().sum()
    
    print(f"   Silver labels: Positive={n_positive}, Negative={n_negative}, Unlabeled={n_unlabeled}")
    
    # Get config values
    if config is None:
        n_samples = 100
        n_positive_target = 50
    else:
        n_samples = config.get('anomaly', {}).get('test_sample_size', 100)
        n_positive_target = n_samples // 2
    
    # Sample for verification
    print(f"Sampling up to {n_samples} for verification...")
    sample = sample_for_verification(df, n_samples=n_samples, n_positive=n_positive_target, random_state=random_state)
    
    if len(sample) < 10:
        print(f"   ⚠️  Warning: Only {len(sample)} samples available (min 10 recommended)")
        print(f"   Skipping ML classifier for {cc}")
        return df, None
    
    # Simulate human verification (in practice, do visual checks here)
    sample['verified_label'] = sample['silver_label']
    
    # Merge verified labels back
    df = df.merge(
        sample[['timestamp', 'verified_label']],
        on='timestamp',
        how='left'
    )
    
    # Create features
    print(f"Creating features...")
    df, feature_cols = create_features(df)
    
    # Train classifier
    print(f"Training classifier...")
    clf, metrics = train_anomaly_classifier(df, feature_cols, test_size=0.3, random_state=random_state)
    
    if clf is None or metrics is None:
        print(f"   ⚠️  Classifier training failed for {cc}")
        return df, None
    
    # Save verified labels
    verified_path = os.path.join(output_folder, f'{cc}_anomaly_labels_verified.csv')
    df[df['verified_label'].notna()].to_csv(verified_path, index=False)
    print(f"✅ Saved verified labels to: {verified_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_folder, f'{cc}_anomaly_ml_eval.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to: {metrics_path}")
    
    return df, metrics


def run_ml_anomaly_detection(config_path: str = 'src/config.yaml'):
    """
    Run ML-based anomaly detection for all countries
    """
    print("\n" + "="*80)
    print("PART 3.2: ML-BASED ANOMALY CLASSIFICATION")
    print("="*80)
    
    config = load_config(config_path)
    
    country_codes = config.get('countries', ['BE', 'DK', 'NL'])
    output_folder = config.get('outputs', {}).get('forecasts_folder', 'outputs')
    
    results = {}
    for cc in country_codes:
        anomalies_path = os.path.join(output_folder, f'{cc}_anomalies.csv')
        forecasts_path = os.path.join(output_folder, f'{cc}_sarima_forecasts_test.csv')
        
        if not os.path.exists(anomalies_path):
            print(f"\n⚠️  Warning: {anomalies_path} not found. Skipping {cc}.")
            continue
        
        if not os.path.exists(forecasts_path):
            print(f"\n⚠️  Warning: {forecasts_path} not found. Skipping {cc}.")
            continue
        
        df, metrics = process_country_ml_anomalies(
            cc=cc,
            anomalies_path=anomalies_path,
            forecasts_path=forecasts_path,
            output_folder=output_folder,
            config=config,
            random_state=42
        )
        
        results[cc] = {'df': df, 'metrics': metrics}
    
    print("\n" + "="*80)
    print("✅ PART 3.2 COMPLETE: ML anomaly classification finished")
    print("="*80)
    
    return results


if __name__ == '__main__':
    run_ml_anomaly_detection()
