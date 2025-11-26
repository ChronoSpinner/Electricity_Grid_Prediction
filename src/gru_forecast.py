"""
GRU_PYTORCH_WORKING.py - GRU Forecasting using PyTorch (FULLY FIXED)
====================================================================
Final fix: Don't pass .values to generate_forecasts, keep pandas Series
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CONFIG_PATH = "src/config.yaml"
OUTPUT_PATH = "outputs"

NINPUT = 168      # 7 days lookback
NOUTPUT = 24      # 24-hour ahead
BATCH_SIZE = 32
EPOCHS = 40
GRU_UNITS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series windowed data"""
    def __init__(self, X, y):
        # Reshape X to (samples, sequence, features)
        self.X = torch.FloatTensor(X).reshape(-1, X.shape[1], 1)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRUModel(nn.Module):
    """GRU forecasting model"""
    def __init__(self, input_size=1, gru_units=50, output_size=24):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, gru_units, batch_first=True)
        self.fc = nn.Linear(gru_units, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output


def read_config(config_path: str) -> dict:
    """Read config.yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_country_data(cc: str, config: dict) -> pd.DataFrame:
    """Load country CSV following SARIMA format"""
    input_folder = config['dataset']['input_folder']
    filepath = os.path.join(input_folder, f"{cc}.csv")
    
    if not os.path.exists(filepath):
        print(f"Warning: file for country {cc} not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Drop cet_cest_timestamp if present
    df = df.drop(columns=['cet_cest_timestamp'], errors='ignore')
    
    # Rename columns to standard names
    df.rename(columns={
        'utc_timestamp': 'timestamp',
        f'{cc}_load_actual_entsoe_transparency': 'load',
        f'{cc}_solar_generation_actual': 'solar',
        f'{cc}_wind_generation_actual': 'wind',
    }, inplace=True, errors='ignore')
    
    # Ensure timestamp exists and convert to datetime
    if 'timestamp' not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {filepath}. Columns: {list(df.columns)}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Ensure load exists
    if 'load' not in df.columns:
        raise ValueError(f"No load column found in {filepath}. Columns: {list(df.columns)}")
    
    # Drop rows with missing load
    df.dropna(subset=['load'], inplace=True)
    
    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Loaded {cc} from {filepath}: {len(df)} rows")
    return df


def create_windows(data, ninput, noutput):
    """Create windowed samples from time series
    data: pandas Series
    """
    X, y = [], []
    for i in range(len(data) - ninput - noutput + 1):
        X.append(data.iloc[i:i + ninput].values)
        y.append(data.iloc[i + ninput:i + ninput + noutput].values)
    return np.array(X), np.array(y).reshape(-1, noutput)


def split_data(df_load: pd.Series, config: dict):
    """Split into train/dev/test using config ratios"""
    ratios = config.get('forecasting', {}).get('ratios', {})
    train_ratio = float(ratios.get('train_ratio', 0.8))
    val_ratio = float(ratios.get('val_ratio', 0.1))
    
    n = len(df_load)
    ntrain = int(n * train_ratio)
    nval = int(n * val_ratio)
    
    df_train = df_load.iloc[:ntrain]
    df_dev = df_load.iloc[ntrain:ntrain + nval]
    df_test = df_load.iloc[ntrain + nval:]
    
    return df_train, df_dev, df_test


def train_gru_model(X_train, y_train, X_dev, y_dev):
    """Train GRU model using PyTorch"""
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_dev shape: {X_dev.shape}, y_dev shape: {y_dev.shape}")
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    dev_dataset = TimeSeriesDataset(X_dev, y_dev)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model
    model = GRUModel(input_size=1, gru_units=GRU_UNITS, output_size=NOUTPUT).to(DEVICE)
    
    print("  Model Architecture:")
    print(f"    Input: (batch, {NINPUT}, 1)")
    print(f"    GRU({1}, {GRU_UNITS})")
    print(f"    Linear({GRU_UNITS}, {NOUTPUT})")
    print(f"    Output: (batch, {NOUTPUT})")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("  Training...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in dev_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(dev_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    return model


def generate_forecasts(model, scaler, df_split_series, split_timestamps_series, split_name="test"):
    """Generate forecasts for a split
    FIX: Keep pandas Series, don't pass .values (numpy array)
    """
    # df_split_series: pandas Series (NOT numpy array)
    X, y = create_windows(df_split_series, NINPUT, NOUTPUT)
    
    print(f"  Generating {split_name} forecasts ({len(X)} samples)...")
    
    # Create dataset
    dataset = TimeSeriesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get predictions
    model.eval()
    scaled_predictions = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            scaled_predictions.append(preds.cpu().numpy())
    
    scaled_predictions = np.vstack(scaled_predictions)
    
    # Inverse transform to get actual values
    dummy_pred = np.zeros((scaled_predictions.shape[0], scaler.n_features_in_))
    dummy_pred[:, 0] = 1
    dummy_scaled_pred = np.column_stack([scaled_predictions, dummy_pred])
    predictions = scaler.inverse_transform(dummy_scaled_pred)[:, :NOUTPUT]
    
    # Inverse transform true values
    dummy_true = np.zeros((y.shape[0], scaler.n_features_in_))
    dummy_true[:, 0] = 1
    dummy_scaled_true = np.column_stack([y, dummy_true])
    true_values = scaler.inverse_transform(dummy_scaled_true)[:, :NOUTPUT]
    
    # Build forecast DataFrame
    rows = []
    # timestamps.iloc[NINPUT:NINPUT + len(predictions)] - slice the timestamps series
    timestamps = split_timestamps_series.iloc[NINPUT:NINPUT + len(predictions)]
    
    for i in range(len(predictions)):
        pred_vals = predictions[i]
        true_vals = true_values[i]
        
        # Compute prediction intervals
        residuals = true_vals - pred_vals
        std_residual = np.std(residuals) if len(residuals) > 1 else 1.0
        z_alpha = 1.96
        
        for h in range(NOUTPUT):
            ts = timestamps.iloc[i] + pd.Timedelta(hours=h+1)
            rows.append({
                'timestamp': ts,
                'yhat': float(pred_vals[h]),
                'lo': max(0, float(pred_vals[h] - z_alpha * std_residual)),
                'hi': float(pred_vals[h] + z_alpha * std_residual),
                'horizon': h + 1,
                'trainend': timestamps.iloc[i],
                'y_true': float(true_vals[h])
            })
    
    df_forecast = pd.DataFrame(rows)
    df_forecast = df_forecast.sort_values(['timestamp', 'horizon']).reset_index(drop=True)
    
    return df_forecast


def forecast_one_country(cc: str, config: dict):
    """Run GRU forecasting for one country"""
    print(f"\n{'='*70}")
    print(f"GRU Forecasting (PyTorch - FINAL): {cc}")
    print(f"{'='*70}")
    
    # Load data
    df = load_country_data(cc, config)
    if df is None:
        print(f"❌ Failed to load data for {cc}")
        return
    
    load_series = df['load'].astype(float)
    timestamps_series = df['timestamp']
    
    # Split
    df_train, df_dev, df_test = split_data(load_series, config)
    print(f"  Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")
    
    # Get corresponding timestamps
    train_ts = timestamps_series.iloc[:len(df_train)]
    dev_ts = timestamps_series.iloc[len(df_train):len(df_train) + len(df_dev)]
    test_ts = timestamps_series.iloc[len(df_train) + len(df_dev):]
    
    # Scale
    scaler = MinMaxScaler()
    scaled_train = pd.Series(
        scaler.fit_transform(df_train.values.reshape(-1, 1)).flatten(),
        index=df_train.index
    )
    scaled_dev = pd.Series(
        scaler.transform(df_dev.values.reshape(-1, 1)).flatten(),
        index=df_dev.index
    )
    scaled_test = pd.Series(
        scaler.transform(df_test.values.reshape(-1, 1)).flatten(),
        index=df_test.index
    )
    
    print(f"  Data normalized using MinMaxScaler (range: 0-1)")
    
    # Create windows
    X_train, y_train = create_windows(scaled_train, NINPUT, NOUTPUT)
    X_dev, y_dev = create_windows(scaled_dev, NINPUT, NOUTPUT)
    
    # Train model
    model = train_gru_model(X_train, y_train, X_dev, y_dev)
    
    # Generate forecasts - KEEP AS PANDAS SERIES!
    print(f"\nGenerating forecasts for {cc}...")
    df_dev_fc = generate_forecasts(model, scaler, scaled_dev, dev_ts, "dev")
    df_test_fc = generate_forecasts(model, scaler, scaled_test, test_ts, "test")
    
    # Save
    dev_path = os.path.join(OUTPUT_PATH, f"{cc}_gru_forecasts_dev.csv")
    test_path = os.path.join(OUTPUT_PATH, f"{cc}_gru_forecasts_test.csv")
    
    df_dev_fc[['timestamp', 'yhat', 'lo', 'hi', 'horizon', 'trainend', 'y_true']].to_csv(dev_path, index=False)
    df_test_fc[['timestamp', 'yhat', 'lo', 'hi', 'horizon', 'trainend', 'y_true']].to_csv(test_path, index=False)
    
    print(f"✅ Saved: {dev_path} ({len(df_dev_fc)} rows)")
    print(f"✅ Saved: {test_path} ({len(df_test_fc)} rows)")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()


def main():
    print("\n" + "="*70)
    print("GRU FORECASTING - PyTorch (FULLY FIXED - Ready to Use)")
    print("="*70)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load config
    config = read_config(CONFIG_PATH)
    countries = config['countries']
    
    print(f"Countries: {countries}")
    print(f"Input folder: {config['dataset']['input_folder']}")
    print(f"Model: GRU(input=1, units={GRU_UNITS}) → Dense({NOUTPUT})")
    print(f"Lookback: {NINPUT} hours | Forecast: {NOUTPUT} hours")
    
    for cc in countries:
        try:
            forecast_one_country(cc, config)
        except Exception as e:
            print(f"❌ Error processing {cc}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ GRU forecasting complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
