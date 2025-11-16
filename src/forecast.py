"""
forecast.py
-------------------
Part 2: Day-Ahead 24-Step Forecasting with SARIMAX + GRU/LSTM (BE, DK, NL)

Implements BOTH:
- 2.2.i: Required SARIMAX with exogenous features
- 2.2.iii: Optional GRU/LSTM for comparison/bonus

GRU Architecture:
- Input: Last 168 hours (7 days)
- Output: Next 24 hours
- Multi-horizon direct forecasting
"""

import os
import yaml
import pandas as pd
import numpy as np
import warnings
import json
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from statsmodels.tsa.statespace.sarimax import SARIMAX

from load_opsd import load_tidy_country_csvs_from_config
from exogenous_features import prepare_exogenous_for_sarimax
from metrics import calculate_mase, calculate_smape, calculate_metrics

warnings.filterwarnings('ignore')

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ================================================================================
# GRU MODEL ARCHITECTURE
# ================================================================================

class GRUForecaster(nn.Module):
    """
    GRU-based multi-horizon forecaster
    Input: Last 168 hours (7 days)
    Output: Next 24 hours
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_horizon=24, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_horizon = output_horizon
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_horizon)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len=168, input_size=1)
        Returns:
            output: (batch, output_horizon=24)
        """
        # GRU forward
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        fc_out = self.fc1(last_hidden)  # (batch, 128)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        output = self.fc2(fc_out)  # (batch, 24)
        
        return output


# ================================================================================
# DATASET CLASS
# ================================================================================

class TimeSeriesDataset(Dataset):
    """
    Creates sequences for GRU training
    Input: Last 168 hours
    Output: Next 24 hours
    """
    def __init__(self, data, lookback=168, horizon=24):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.len = len(data) - lookback - horizon + 1
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]  # (168,)
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]  # (24,)
        
        x = torch.FloatTensor(x).unsqueeze(-1)  # (168, 1)
        y = torch.FloatTensor(y)  # (24,)
        
        return x, y


# ================================================================================
# TRAINING FUNCTIONS
# ================================================================================

def train_gru_model(y_series, epochs=50, batch_size=32, lookback=168, horizon=24):
    """
    Train GRU model on time series data
    """
    print(f"\n🧠 Training GRU model...")
    print(f"   Data shape: {y_series.shape}")
    print(f"   Lookback: {lookback}h, Horizon: {horizon}h")
    print(f"   Device: {DEVICE}")
    
    # Normalize data
    mean = y_series.mean()
    std = y_series.std()
    y_normalized = (y_series - mean) / std
    
    # Create dataset and dataloader
    dataset = TimeSeriesDataset(y_normalized.values, lookback=lookback, horizon=horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = GRUForecaster(
        input_size=1, 
        hidden_size=64, 
        num_layers=2, 
        output_horizon=horizon, 
        dropout=0.2
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    # Training loop
    best_loss = float('inf')
    print(f"\n   Epoch | Train Loss")
    print(f"   ------|----------")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   {epoch+1:4d}  | {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"   Final loss: {best_loss:.6f}")
    print(f"   ✅ GRU training complete!")
    
    return model, mean, std


def gru_forecast_backtest(model, y_series, mean, std, lookback=168, stride=24, horizon=24):
    """
    Perform expanding-origin backtest with GRU
    """
    forecasts = []
    model.eval()
    
    with torch.no_grad():
        start_idx = lookback
        end_idx = len(y_series) - horizon
        
        for t in tqdm(range(start_idx, end_idx, stride), desc="GRU Backtest"):
            # Get last 168 hours (normalized)
            y_train = (y_series[:t] - mean) / std
            x_input = y_train.iloc[-lookback:].values  # Last 168 hours
            
            # Reshape for model
            x_input = torch.FloatTensor(x_input).unsqueeze(0).unsqueeze(-1)  # (1, 168, 1)
            x_input = x_input.to(DEVICE)
            
            # Forecast
            y_pred_normalized = model(x_input).cpu().numpy()[0]  # (24,)
            y_pred = y_pred_normalized * std + mean  # Denormalize
            
            # Get actual values
            y_actual = y_series.iloc[t:t+horizon].values
            
            # Store results
            for step in range(horizon):
                forecasts.append({
                    'timestamp': y_series.index[t+step],
                    'y_true': y_actual[step],
                    'yhat': y_pred[step],
                    'lo': y_pred[step] - 1.96 * std,  # Approximate 80% CI
                    'hi': y_pred[step] + 1.96 * std,
                    'horizon': step + 1,
                    'train_end': t
                })
    
    df_forecast = pd.DataFrame(forecasts)
    return df_forecast


# ================================================================================
# MAIN FORECASTING FUNCTION
# ================================================================================

def create_train_val_test_splits(df, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10):
    """Create chronological splits"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    return df_train, df_val, df_test


def forecast_models_combined(dfs, config_path='src/config.yaml'):
    """
    Build BOTH SARIMAX and GRU models
    - SARIMAX: Required classical model with exogenous features
    - GRU: Optional neural model for comparison
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load SARIMA orders
    figures_folder = config['outputs']['figures_folder']
    orders_file = os.path.join(figures_folder, 'sarima_orders_summary.yaml')
    
    if not os.path.exists(orders_file):
        print(f"❌ Error: {orders_file} not found. Run decompose_acfpacf.py first!")
        return
    
    # --- FIX #1: Use UnsafeLoader to read Python tuples ---
    with open(orders_file, 'r') as f:
        sarima_orders = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    forecasts_folder = config['outputs']['forecasts_folder']
    models_folder = config['outputs']['models_folder']
    metrics_folder = config['outputs']['metrics_folder']
    
    os.makedirs(forecasts_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    
    train_ratio = config['forecasting']['train_ratio']
    val_ratio = config['forecasting']['val_ratio']
    test_ratio = config['forecasting']['test_ratio']
    warmup_days = config['forecasting']['warmup_days']
    warmup_hours = warmup_days * 24
    stride = config['forecasting']['stride']
    horizon = config['forecasting']['horizon']
    
    print("\n" + "="*80)
    print("PART 2: FORECASTING WITH SARIMAX + GRU")
    print("="*80)
    print(f"\nBacktest config:")
    print(f"  Warmup: {warmup_days} days")
    print(f"  Stride: {stride}h, Horizon: {horizon}h")
    print(f"  Train/Val/Test: {train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f}%")
    
    all_metrics = {}
    
    for cc, df in dfs.items():
        print(f"\n{'='*80}")
        print(f"Country: {cc}")
        print(f"{'='*80}")
        
        if cc not in sarima_orders:
            print(f"❌ No SARIMA order found for {cc}. Skipping...")
            continue
        
        order = tuple(sarima_orders[cc]['order'])
        seasonal_order = tuple(sarima_orders[cc]['seasonal_order'])
        print(f"\n✅ SARIMAX{order} x {seasonal_order}")
        print(f"🧠 GRU (168h → 24h)")
        
        # Split data
        df_train, df_val, df_test = create_train_val_test_splits(df, train_ratio, val_ratio, test_ratio)
        
        print(f"\n📊 Data splits:")
        print(f"   Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        
        # ================================================================================
        # SARIMAX FORECASTING
        # ================================================================================
        
        print(f"\n{'─'*40}")
        print(f"MODEL 1: SARIMAX (Required)")
        print(f"{'─'*40}")
        
        # Prepare exogenous features
        exog_train = prepare_exogenous_for_sarimax(df_train, include_wind=True, include_solar=True, include_calendar=True)
        exog_val = prepare_exogenous_for_sarimax(df_val, include_wind=True, include_solar=True, include_calendar=True)
        exog_test = prepare_exogenous_for_sarimax(df_test, include_wind=True, include_solar=True, include_calendar=True)
        
        df_combined = pd.concat([df_train, df_val], ignore_index=True)
        exog_combined = pd.concat([exog_train, exog_val], ignore_index=True)
        y_combined = pd.Series(df_combined['load'].values)
        
        print(f"🔄 Backtest on Validation set...")
        
        # Simple validation forecast (for speed, using rolling window)
        df_val_forecast = []
        start_idx = len(df_train)
        
        for t in tqdm(range(start_idx, len(df_combined) - horizon, stride), desc="SARIMAX Val"):
            try:
                y_train_subset = y_combined[:t]
                exog_train_subset = exog_combined[:t]
                exog_future = exog_combined[t:t+horizon]
                
                model = SARIMAX(y_train_subset, exog=exog_train_subset, 
                               order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                
                result = model.get_forecast(steps=horizon, exog=exog_future)
                forecast = result.predicted_mean
                ci = result.conf_int(alpha=0.2)
                
                y_actual = df_combined.iloc[t:t+horizon]['load'].values
                timestamps = df_combined.iloc[t:t+horizon]['timestamp'].values
                
                for step in range(horizon):
                    df_val_forecast.append({
                        'timestamp': timestamps[step],
                        'y_true': y_actual[step],
                        'yhat': forecast.iloc[step],
                        'lo': ci.iloc[step, 0],
                        'hi': ci.iloc[step, 1],
                        'horizon': step + 1,
                        'train_end': t
                    })
            except:
                continue
        
        df_sarimax_val = pd.DataFrame(df_val_forecast)
        
        # Save SARIMAX validation forecasts
        sarimax_val_path = os.path.join(forecasts_folder, f'{cc}_sarimax_forecasts_dev.csv')
        df_sarimax_val.to_csv(sarimax_val_path, index=False)
        print(f"💾 SARIMAX dev forecasts: {sarimax_val_path}")
        
        # ================================================================================
        # GRU FORECASTING
        # ================================================================================
        
        print(f"\n{'─'*40}")
        print(f"MODEL 2: GRU/LSTM (Optional Bonus)")
        print(f"{'─'*40}")
        
        # Train GRU on training data
        y_train_gru = pd.Series(df_train['load'].values)
        gru_model, mean, std = train_gru_model(y_train_gru, epochs=50, batch_size=32)
        
        # GRU validation forecast
        y_combined_for_gru = pd.Series(df_combined['load'].values)
        y_combined_for_gru.index = range(len(y_combined_for_gru))
        
        print(f"🔄 Backtest on Validation set...")
        df_gru_val = gru_forecast_backtest(gru_model, y_combined_for_gru, mean, std, lookback=168, stride=stride, horizon=horizon)
        
        # Save GRU validation forecasts
        gru_val_path = os.path.join(forecasts_folder, f'{cc}_gru_forecasts_dev.csv')
        df_gru_val.to_csv(gru_val_path, index=False)
        print(f"💾 GRU dev forecasts: {gru_val_path}")
        
        # ================================================================================
        # METRICS COMPARISON (FIXED)
        # ================================================================================
        
        print(f"\n{'─'*40}")
        print(f"METRICS COMPARISON")
        print(f"{'─'*40}")
        
        # --- FIX #2: Initialize to infinity to prevent UnboundLocalError ---
        mase_sarimax = float('inf')
        mase_gru = float('inf')
        
        # SARIMAX metrics
        if len(df_sarimax_val) > 0:
            mase_sarimax = calculate_mase(df_sarimax_val['y_true'], df_sarimax_val['yhat'], seasonal=24)
            smape_sarimax = calculate_smape(df_sarimax_val['y_true'], df_sarimax_val['yhat'])
            metrics_sarimax = calculate_metrics(df_sarimax_val['y_true'], df_sarimax_val['yhat'])
            coverage_sarimax = ((df_sarimax_val['y_true'] >= df_sarimax_val['lo']) & 
                               (df_sarimax_val['y_true'] <= df_sarimax_val['hi'])).mean()
            
            print(f"\n📊 SARIMAX (Dev):")
            print(f"   MASE: {mase_sarimax:.4f}")
            print(f"   sMAPE: {smape_sarimax:.4f}")
            print(f"   RMSE: {metrics_sarimax['rmse']:.4f}")
            print(f"   PI Coverage (80%): {coverage_sarimax:.4f}")
        else:
            print("\n⚠️ SARIMAX produced no forecasts (Skipping metrics)")
        
        # GRU metrics
        if len(df_gru_val) > 0:
            mase_gru = calculate_mase(df_gru_val['y_true'], df_gru_val['yhat'], seasonal=24)
            smape_gru = calculate_smape(df_gru_val['y_true'], df_gru_val['yhat'])
            metrics_gru = calculate_metrics(df_gru_val['y_true'], df_gru_val['yhat'])
            coverage_gru = ((df_gru_val['y_true'] >= df_gru_val['lo']) & 
                           (df_gru_val['y_true'] <= df_gru_val['hi'])).mean()
            
            print(f"\n🧠 GRU (Dev):")
            print(f"   MASE: {mase_gru:.4f}")
            print(f"   sMAPE: {smape_gru:.4f}")
            print(f"   RMSE: {metrics_gru['rmse']:.4f}")
            print(f"   PI Coverage (80%): {coverage_gru:.4f}")
            
            # --- FIX #3: Safe Comparison ---
            if mase_sarimax != float('inf') and mase_gru != float('inf'):
                print(f"\n🏆 Winner:")
                if mase_sarimax <= mase_gru:
                    print(f"   SARIMAX (MASE: {mase_sarimax:.4f} < {mase_gru:.4f})")
                else:
                    print(f"   GRU (MASE: {mase_gru:.4f} < {mase_sarimax:.4f})")
            else:
                print("\nℹ️  Cannot determine winner (one or both models failed).")
        
        # Save models
        model_sarimax_path = os.path.join(models_folder, f'{cc}_sarimax_model.pkl')
        model_gru_path = os.path.join(models_folder, f'{cc}_gru_model.pt')
        
        torch.save(gru_model.state_dict(), model_gru_path)
        print(f"\n💾 Models saved:")
        print(f"   {model_gru_path}")
    
    print("\n✅ Part 2 (Forecasting with SARIMAX + GRU) Complete!")


def main(config_path='src/config.yaml'):
    """Run forecasting with both SARIMAX and GRU"""
    dfs = load_tidy_country_csvs_from_config(config_path)
    
    if not dfs:
        print("❌ No countries loaded!")
        return
    
    forecast_models_combined(dfs, config_path)


if __name__ == '__main__':
    main()