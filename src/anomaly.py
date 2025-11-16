import os
import json
import numpy as np
import pandas as pd

# Module-level defaults (edit these values directly if you prefer not to use CLI)
DEFAULTS = {
    'input': os.path.join('outputs'),
    'out': os.path.join('outputs'),
    'do_cusum': True,
}


def load_forecast(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]) if os.path.exists(path) else None
    return df


def compute_residuals(df):
    if 'y_true' not in df.columns or 'yhat' not in df.columns:
        raise ValueError("Input forecast CSV must contain 'y_true' and 'yhat' columns")
    df = df.sort_values('timestamp').copy()
    df['resid'] = df['y_true'] - df['yhat']
    return df


def rolling_z_score(df, column='resid', window=336, min_periods=168):
    # compute rolling mean/std and z-score
    roll_mean = df[column].rolling(window=window, min_periods=min_periods).mean()
    roll_std = df[column].rolling(window=window, min_periods=min_periods).std()
    z = (df[column] - roll_mean) / roll_std
    return z, roll_mean, roll_std


def compute_cusum(z_series, k=0.5, h=5.0):
    # One-sided CUSUM (Page's CUSUM) for positive and negative shifts
    s_pos = []
    s_neg = []
    sp = 0.0
    sn = 0.0
    for v in z_series.fillna(0).values:
        sp = max(0.0, sp + v - k)
        sn = min(0.0, sn + v + k)
        s_pos.append(sp)
        s_neg.append(sn)
    s_pos = np.array(s_pos)
    s_neg = np.array(s_neg)
    flag = ((s_pos > h) | (np.abs(s_neg) > h)).astype(int)
    return flag, s_pos, s_neg


def process_file(inpath, out_folder='outputs', do_cusum=True):
    df = load_forecast(inpath)
    if df is None:
        raise FileNotFoundError(inpath)

    # try to use 1-step residuals if horizon column present
    if 'horizon' in df.columns:
        # prefer horizon == 1 rows for z-score computation
        one_step = df[df['horizon'] == 1]
        if one_step.empty:
            working = df.copy()
        else:
            working = one_step.copy()
    else:
        working = df.copy()

    working = compute_residuals(working)
    z, mu, sigma = rolling_z_score(working, column='resid', window=336, min_periods=168)
    working['z_resid'] = z
    working['flag_z'] = (working['z_resid'].abs() >= 3.0).astype(int)

    if do_cusum:
        flag_cusum, s_pos, s_neg = compute_cusum(working['z_resid'], k=0.5, h=5.0)
        working['flag_cusum'] = flag_cusum
        working['cusum_pos'] = s_pos
        working['cusum_neg'] = s_neg

    # If original df had multiple horizons, join anomalies back to original timestamps when possible
    # We'll output the working frame (which is the natural set of 1-step residuals if available)

    # prepare output filename
    basename = os.path.basename(inpath)
    cc = basename.split('_')[0]
    outpath = os.path.join(out_folder, f"{cc}_anomalies.csv")
    os.makedirs(out_folder, exist_ok=True)

    cols = [c for c in ['timestamp', 'y_true', 'yhat', 'z_resid', 'flag_z'] if c in working.columns]
    if do_cusum:
        cols += ['flag_cusum']

    working.to_csv(outpath, columns=cols, index=False)
    print(f"Wrote anomalies to {outpath}")
    return outpath


def run(input_path=None, out_folder=None, do_cusum=None):
    """Run anomaly processing. If arguments are None, use module-level DEFAULTS.
    You can import this module and call run(...) or edit DEFAULTS above and
    execute the module directly.
    """
    if input_path is None:
        input_path = DEFAULTS['input']
    if out_folder is None:
        out_folder = DEFAULTS['out']
    if do_cusum is None:
        do_cusum = DEFAULTS['do_cusum']

    if os.path.isdir(input_path):
        # process all *_forecasts_test.csv in directory
        for fn in os.listdir(input_path):
            if fn.endswith('_forecasts_test.csv'):
                inpath = os.path.join(input_path, fn)
                try:
                    process_file(inpath, out_folder=out_folder, do_cusum=do_cusum)
                except Exception as e:
                    print(f"Failed for {inpath}: {e}")
    else:
        process_file(input_path, out_folder=out_folder, do_cusum=do_cusum)


def main():
    # Default run when module executed directly. Edit DEFAULTS above to change behavior.
    run()


if __name__ == '__main__':
    main()
