import os
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score
import joblib

# Module-level defaults (edit these values directly if you prefer not to use CLI)
DEFAULTS = {
    'input': os.path.join('outputs'),
    'out': os.path.join('outputs'),
}


def load_anomalies(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


def build_features(df, max_lag=24):
    # df must be sorted by timestamp
    df = df.sort_values('timestamp').copy()
    for lag in range(1, max_lag + 1):
        df[f'resid_lag_{lag}'] = df['resid'].shift(lag)
        df[f'y_lag_{lag}'] = df['y_true'].shift(lag)

    # rolling features
    df['resid_roll24_mean'] = df['resid'].rolling(24, min_periods=1).mean()
    df['resid_roll24_std'] = df['resid'].rolling(24, min_periods=1).std()
    df['resid_roll48_mean'] = df['resid'].rolling(48, min_periods=1).mean()

    # calendar
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'] >= 5

    # prediction interval breach (requires lo/hi)
    if 'lo' in df.columns and 'hi' in df.columns:
        df['pi_breach'] = ((df['y_true'] < df['lo']) | (df['y_true'] > df['hi'])).astype(int)
    else:
        df['pi_breach'] = 0

    # abs z
    df['abs_z'] = df['z_resid'].abs()

    return df


def make_silver_labels(df):
    # positive if (|zt| >= 3.5) OR (y_true outside [lo,hi] AND |zt| >= 2.5)
    cond_pos = (df['abs_z'] >= 3.5)
    cond_pos2 = (('lo' in df.columns and 'hi' in df.columns) & (df['pi_breach'] == 1) & (df['abs_z'] >= 2.5))
    df['label_silver'] = 0
    df.loc[cond_pos | cond_pos2, 'label_silver'] = 1

    # negative if |zt| < 1.0 AND y_true inside [lo,hi]
    cond_neg = (df['abs_z'] < 1.0) & (('lo' not in df.columns) | (df['pi_breach'] == 0))
    df.loc[cond_neg, 'label_silver'] = 0

    # only keep rows that are clearly labeled (either pos or neg by rules); others -> -1
    labeled = df[(cond_pos | cond_pos2) | cond_neg].copy()
    return labeled


def sample_for_human_check(df_labeled, n_samples=100, target_pos_ratio=0.5, rng=42):
    pos = df_labeled[df_labeled['label_silver'] == 1]
    neg = df_labeled[df_labeled['label_silver'] == 0]
    n_pos = int(n_samples * target_pos_ratio)
    n_neg = n_samples - n_pos
    pos_s = pos.sample(n=min(n_pos, len(pos)), random_state=rng)
    neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=rng)
    sample_df = pd.concat([pos_s, neg_s]).sample(frac=1, random_state=rng)
    return sample_df


def train_classifier(df_labeled, feature_cols):
    X = df_labeled[feature_cols].fillna(0).values
    y = df_labeled['label_silver'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_s, y_train)

    probs = clf.predict_proba(X_test_s)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    # Find threshold achieving precision >= 0.80 if possible
    th = None
    f1_at_p = None
    idxs = np.where(precision >= 0.80)[0]
    if len(idxs) > 0:
        # choose threshold with max recall among these
        best_idx = idxs[np.argmax(recall[idxs])]
        if best_idx < len(thresholds):
            th = thresholds[best_idx]
        else:
            th = thresholds[-1]
        y_pred = (probs >= th).astype(int)
        f1_at_p = f1_score(y_test, y_pred)
    else:
        # fallback: threshold 0.5
        th = 0.5
        y_pred = (probs >= th).astype(int)
        f1_at_p = f1_score(y_test, y_pred)

    metrics = {
        'pr_auc': float(pr_auc),
        'f1_at_precision_0.80': (None if f1_at_p is None else float(f1_at_p)),
        'chosen_threshold': float(th)
    }

    return clf, scaler, metrics


def save_outputs(cc, labeled_df, sample_df, clf, scaler, metrics, out_folder='outputs'):
    os.makedirs(out_folder, exist_ok=True)
    labeled_path = os.path.join(out_folder, f"{cc}_anomaly_silver_labels.csv")
    sample_path = os.path.join(out_folder, f"{cc}_anomaly_labels_sample.csv")
    model_path = os.path.join(out_folder, f"{cc}_anomaly_model.joblib")
    eval_path = os.path.join(out_folder, f"{cc}_anomaly_ml_eval.json")

    labeled_df.to_csv(labeled_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    joblib.dump({'model': clf, 'scaler': scaler}, model_path)
    with open(eval_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved silver labels to {labeled_path}")
    print(f"Saved sampled labels for human check to {sample_path}")
    print(f"Saved model to {model_path} and metrics to {eval_path}")


def process_file(anom_path, out_folder='outputs'):
    df = load_anomalies(anom_path)

    # we need resid, z_resid, y_true, lo, hi if available
    if 'resid' not in df.columns:
        # try to compute resid from y_true and yhat
        if 'y_true' in df.columns and 'yhat' in df.columns:
            df['resid'] = df['y_true'] - df['yhat']
        else:
            raise ValueError('Input anomalies file must contain resid or y_true and yhat')

    df = build_features(df)
    labeled = make_silver_labels(df)
    if labeled.empty:
        raise ValueError('No rows matched the silver-label rules; cannot train')

    sample_df = sample_for_human_check(labeled, n_samples=100)

    # choose feature columns: lags, roll stats, calendar, pi_breach, abs_z
    feature_cols = [c for c in labeled.columns if ('resid_lag_' in c or 'y_lag_' in c)]
    feature_cols += [c for c in ['resid_roll24_mean', 'resid_roll24_std', 'resid_roll48_mean', 'hour', 'dow', 'is_weekend', 'pi_breach', 'abs_z'] if c in labeled.columns]

    clf, scaler, metrics = train_classifier(labeled, feature_cols)

    # save outputs
    basename = os.path.basename(anom_path)
    cc = basename.split('_')[0]
    save_outputs(cc, labeled, sample_df, clf, scaler, metrics, out_folder=out_folder)


def main():
    # Default run when module executed directly. Edit DEFAULTS at top to change behavior.
    input_path = DEFAULTS['input']
    out_folder = DEFAULTS['out']

    if os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            if fn.endswith('_anomalies.csv'):
                inpath = os.path.join(input_path, fn)
                try:
                    process_file(inpath, out_folder=out_folder)
                except Exception as e:
                    print(f"Failed for {inpath}: {e}")
    else:
        process_file(input_path, out_folder=out_folder)


if __name__ == '__main__':
    main()
