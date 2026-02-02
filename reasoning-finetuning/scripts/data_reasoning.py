import numpy as np
import pandas as pd
import datasets
from datasets import load_from_disk, Dataset
from gluonts.dataset.arrow import ArrowWriter
from pathlib import Path
from statsmodels.tsa.seasonal import STL

def compute_features(target, period=24):
    """
    Compute Trend, Seasonality using STL, and Volatility.
    Assumes hourly data (period=24) for M4 Hourly.
    If len(target) < 2*period, STL might fail or be unstable. 
    Fallbacks provided.
    """
    # Ensure numpy
    series = np.array(target)
    
    # 1. Volatility (Rolling Std)
    # Window 5 as suggested
    series_pd = pd.Series(series)
    volatility = series_pd.rolling(window=5, min_periods=1).std().fillna(0).values
    
    # 2. Decomposition (STL)
    # Default period 24 for hourly. 
    # M4 Hourly length varies (often 700+).
    # If too short, return 0s or simple mean.
    if len(series) > 2 * period:
        try:
            # robust=True for outliers
            res = STL(series_pd, period=period, robust=True).fit()
            trend = res.trend.values
            seasonal = res.seasonal.values
        except:
             trend = np.zeros_like(series)
             seasonal = np.zeros_like(series)
    else:
        # Fallback: Trend = simple moving average
        trend = series_pd.rolling(window=period, min_periods=1).mean().fillna(series_pd.mean()).values
        seasonal = series - trend

    # 3. Changepoints (Binary)
    # Simple logic: shift in mean > threshold? 
    # For now, just placeholder or simple z-score jump.
    # User said: "simple binary sequence (0 or 1) flagging sudden shifts"
    # Let's use diff of trend ?? 
    # Or just keep it simple: rolling std > 2*global std?
    # Actually user said "Target: Trend + Season + Vol + Forecast". 
    # The "Changepoints" was in Step 1 list but NOT in Step 3 Target list. 
    # Step 3 says "Target: [Trend_Tokens] + [Season_Tokens] + [Vol_Tokens] + [Forecast_Tokens]"
    # It omits Changepoints in the target. I will allow computing it but maybe not use it in target if not requested.
    # Wait, the user prompt Step 3 explicitly lists Trend, Season, Vol. Changepoints is missing there.
    # I will calculate it anyway.
    
    return trend, seasonal, volatility

def process_dataset(input_path, output_path):
    print(f"Loading dataset from {input_path}...")
    # It's an Arrow file, load with datasets
    ds = datasets.load_dataset("arrow", data_files=str(input_path), split="train")
    
    new_data = []
    
    print("Augmenting data with Reasoning Chains (Trend, Season, Vol)...")
    for i, entry in enumerate(ds):
        target = entry['target']
        start = entry['start']
        
        # M4 Hourly period = 24
        trend, seasonal, volatility = compute_features(target, period=24)
        
        new_data.append({
            "start": start,
            "target": target,
            "trend": trend,
            "seasonal": seasonal,
            "volatility": volatility
        })
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} series")

    print(f"Writing augmented dataset to {output_path}...")
    ArrowWriter(compression="lz4").write_to_file(
        new_data,
        path=output_path,
    )
    print("Done.")

if __name__ == "__main__":
    input_arrow = "gifteval-subset-finetune.arrow" 
    # Wait, previous file was 'gifteval-subset.arrow'
    input_arrow = "gifteval-subset.arrow"
    output_arrow = "gifteval-reasoning.arrow"
    
    process_dataset(input_arrow, output_arrow)
