#!/usr/bin/env python3
"""
Generate verification plots for the reasoning fine-tuning exercise.

Plot 1: Fast mode — Finetuned vs Base Pretrained forecast comparison
        (top: context + forecasts + ground truth, bottom: difference bar chart)

Plot 2: Reasoning mode — Model decomposition vs STL ground truth
        (3 panels: Trend, Seasonality, Volatility)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import pandas as pd
import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from chronos import ChronosPipeline

# Constants
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
DECOMPOSITION_LENGTH = 64
FAST_MODE_TOKEN_ID = 4096
REASONING_MODE_TOKEN_ID = 4097
SEASONAL_AMP = 10.0
VOLATILITY_AMP = 50.0

FINETUNED_PATH = '/root/chronos-forecasting/output/reasoning-v4-from-amazon/'
DATA_PATH = '/root/chronos-forecasting/reasoning-finetuning/data/gifteval-reasoning.arrow'
OUTPUT_DIR = '/root/chronos-forecasting/reasoning-finetuning/figures'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models():
    print('Loading finetuned model...')
    finetuned = ChronosPipeline.from_pretrained(
        FINETUNED_PATH, device_map=device, torch_dtype=torch.float32
    )
    print(f'  Finetuned vocab: {finetuned.model.model.shared.num_embeddings}')

    print('Loading base model...')
    base = ChronosPipeline.from_pretrained(
        'amazon/chronos-t5-small', device_map=device, torch_dtype=torch.float32
    )
    print(f'  Base vocab: {base.model.model.shared.num_embeddings}')

    return finetuned, base

def load_validation_data():
    arrow_ds = datasets.load_dataset('arrow', data_files=DATA_PATH, split='train')
    split_ds = arrow_ds.train_test_split(test_size=0.2, seed=42)
    return split_ds['test']

def extract_context_future(target):
    target = np.array(target)
    total_needed = CONTEXT_LENGTH + PREDICTION_LENGTH
    if len(target) >= total_needed:
        context = target[-(total_needed):-PREDICTION_LENGTH]
        future = target[-PREDICTION_LENGTH:]
    else:
        sp = len(target) - PREDICTION_LENGTH
        context = target[:sp]
        future = target[sp:]
    return context, future

def compute_stl_decomposition(series, period=7):
    series_clean = pd.Series(series).ffill().bfill().values
    if len(series_clean) < 2 * period:
        pad = 2 * period - len(series_clean)
        series_clean = np.concatenate([np.full(pad, series_clean[0]), series_clean])
    stl = STL(series_clean, period=period, robust=True)
    result = stl.fit()
    vol_window = max(3, period // 2)
    volatility = pd.Series(result.resid).rolling(window=vol_window, min_periods=1).std().values
    return {
        'trend': result.trend[-DECOMPOSITION_LENGTH:],
        'seasonal': result.seasonal[-DECOMPOSITION_LENGTH:],
        'volatility': volatility[-DECOMPOSITION_LENGTH:],
    }

def generate_fast_mode(pipeline, context_tensor, with_token=True):
    tokenizer = pipeline.tokenizer
    token_ids, attn_mask, scale = tokenizer.context_input_transform(context_tensor)

    if with_token:
        ft_tk = torch.tensor([[FAST_MODE_TOKEN_ID]], dtype=token_ids.dtype)
        ft_mk = torch.tensor([[True]], dtype=attn_mask.dtype)
        token_ids = torch.cat([ft_tk, token_ids], dim=1)
        attn_mask = torch.cat([ft_mk, attn_mask], dim=1)

    token_ids = token_ids.to(device)
    attn_mask = attn_mask.to(device)

    generated = pipeline.model(
        input_ids=token_ids, attention_mask=attn_mask,
        prediction_length=PREDICTION_LENGTH, num_samples=1,
        temperature=1.0, top_k=50, top_p=1.0,
    )
    decoded = tokenizer.output_transform(generated.cpu(), scale)
    return decoded[0, 0, :].numpy()

def generate_reasoning_mode(pipeline, context_tensor):
    tokenizer = pipeline.tokenizer
    token_ids, attn_mask, scale = tokenizer.context_input_transform(context_tensor)

    r_tk = torch.tensor([[REASONING_MODE_TOKEN_ID]], dtype=token_ids.dtype)
    r_mk = torch.tensor([[True]], dtype=attn_mask.dtype)
    token_ids = torch.cat([r_tk, token_ids], dim=1)
    attn_mask = torch.cat([r_mk, attn_mask], dim=1)

    token_ids = token_ids.to(device)
    attn_mask = attn_mask.to(device)

    max_new = 3 * DECOMPOSITION_LENGTH + PREDICTION_LENGTH  # 256
    generated = pipeline.model(
        input_ids=token_ids, attention_mask=attn_mask,
        prediction_length=max_new, num_samples=1,
        temperature=1.0, top_k=50, top_p=1.0,
    )
    decoded = tokenizer.output_transform(generated.cpu(), scale)
    values = decoded[0, 0, :].numpy()

    # Split into components (tokens are in amplified space)
    trend_amp = values[:DECOMPOSITION_LENGTH]
    seasonal_amp = values[DECOMPOSITION_LENGTH:2*DECOMPOSITION_LENGTH]
    volatility_amp = values[2*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH]
    forecast = values[3*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH+PREDICTION_LENGTH]

    # De-amplify to original scale
    trend = trend_amp  # trend has no amplification
    seasonal = seasonal_amp / SEASONAL_AMP
    volatility = volatility_amp / VOLATILITY_AMP

    return {
        'trend': trend,
        'seasonal': seasonal,
        'volatility': volatility,
        'forecast': forecast,
    }

def safe_corr(a, b):
    ml = min(len(a), len(b))
    a, b = a[:ml], b[:ml]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# =========================================================================
# Plot 1: Fast Mode — Finetuned vs Base Pretrained
# =========================================================================
def plot_fast_mode(context, future, ft_forecast, base_forecast, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle('Forecast Comparison: Finetuned vs Base Pretrained',
                 fontsize=14, fontweight='bold', y=0.98)

    # Show last 100 context points + forecast
    n_ctx_show = min(100, len(context))
    ctx_show = context[-n_ctx_show:]
    ctx_x = np.arange(-n_ctx_show, 0)
    fcast_x = np.arange(0, PREDICTION_LENGTH)

    ax1.plot(ctx_x, ctx_show, color='gray', linewidth=1.2, label='Context')
    ax1.plot(fcast_x, future, color='green', linewidth=2.0, label='Ground Truth')
    ax1.plot(fcast_x, ft_forecast, color='blue', linewidth=1.5, linestyle='--',
             label='Finetuned (Fast Mode)')
    ax1.plot(fcast_x, base_forecast, color='red', linewidth=1.5, linestyle=':',
             label='Base Pretrained')

    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Time Step (0 = forecast start)', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Difference bar chart
    diff = ft_forecast - base_forecast
    colors = ['#7B2D8E' for _ in diff]  # purple
    ax2.bar(fcast_x, diff, color=colors, width=0.8, alpha=0.85)
    ax2.set_xlabel('Time Step (0 = forecast start)', fontsize=11)
    ax2.set_ylabel('Difference (Finetuned - Base)', fontsize=11)
    ax2.set_title('Difference Between Finetuned and Base Forecasts', fontsize=12)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # Add correlation annotation
    corr_ft_gt = safe_corr(ft_forecast, future)
    corr_base_gt = safe_corr(base_forecast, future)
    corr_between = safe_corr(ft_forecast, base_forecast)
    ax1.text(0.99, 0.02,
             f'Corr(FT,GT)={corr_ft_gt:+.4f}  Corr(Base,GT)={corr_base_gt:+.4f}  Corr(FT,Base)={corr_between:+.4f}',
             transform=ax1.transAxes, fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved fast mode plot: {output_path}')
    plt.close()

# =========================================================================
# Plot 2: Reasoning Mode — Model vs STL Ground Truth
# =========================================================================
def plot_reasoning_mode(model_decomp, gt_decomp, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Overlay Comparison: Model vs Ground Truth',
                 fontsize=14, fontweight='bold', y=1.02)

    components = [
        ('Trend', 'trend'),
        ('Seasonality', 'seasonal'),
        ('Volatility', 'volatility'),
    ]

    for ax, (title, key) in zip(axes, components):
        model_vals = model_decomp[key]
        gt_vals = gt_decomp[key]
        x = np.arange(len(gt_vals))

        ax.plot(x, model_vals[:len(x)], color='blue', linewidth=1.5,
                label='Model')
        ax.plot(x, gt_vals, color='green', linewidth=1.5, linestyle='--',
                label='Ground Truth')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Correlation annotation
        corr = safe_corr(model_vals[:len(gt_vals)], gt_vals)
        ax.text(0.98, 0.02, f'Corr={corr:+.4f}',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved reasoning mode plot: {output_path}')
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    finetuned, base = load_models()
    val_ds = load_validation_data()

    # Find a good reasoning-mode sample (with sufficient data)
    reasoning_indices = [i for i in range(len(val_ds)) if val_ds[i]['mode'] == 'reasoning']
    fast_indices = [i for i in range(len(val_ds)) if val_ds[i]['mode'] == 'fast']
    print(f'Reasoning samples: {len(reasoning_indices)}, Fast samples: {len(fast_indices)}')

    # ---- FAST MODE PLOT ----
    # Use a sample with enough context for a clean plot
    best_idx = None
    for idx in range(len(val_ds)):
        target = np.array(val_ds[idx]['target'])
        if len(target) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
            best_idx = idx
            break
    if best_idx is None:
        best_idx = 0

    print(f'\n--- Fast Mode Plot (sample idx={best_idx}) ---')
    target = np.array(val_ds[best_idx]['target'])
    context, future = extract_context_future(target)
    context_tensor = torch.tensor(context).float().unsqueeze(0)

    ft_forecast = generate_fast_mode(finetuned, context_tensor, with_token=True)
    base_forecast = generate_fast_mode(base, context_tensor, with_token=False)

    plot_fast_mode(context, future, ft_forecast, base_forecast,
                   os.path.join(OUTPUT_DIR, 'fast_mode_comparison.png'))

    # ---- REASONING MODE PLOT ----
    # Find a reasoning-mode sample with good context
    best_r_idx = None
    for ri in reasoning_indices:
        target = np.array(val_ds[ri]['target'])
        if len(target) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
            best_r_idx = ri
            break
    if best_r_idx is None:
        best_r_idx = reasoning_indices[0]

    print(f'\n--- Reasoning Mode Plot (sample idx={best_r_idx}) ---')
    target = np.array(val_ds[best_r_idx]['target'])
    context, future = extract_context_future(target)
    context_tensor = torch.tensor(context).float().unsqueeze(0)

    model_decomp = generate_reasoning_mode(finetuned, context_tensor)
    gt_decomp = compute_stl_decomposition(context, period=7)

    plot_reasoning_mode(model_decomp, gt_decomp,
                        os.path.join(OUTPUT_DIR, 'reasoning_decomposition.png'))

    # Print numeric summary
    print('\n' + '='*60)
    print('NUMERIC SUMMARY')
    print('='*60)
    print(f'Trend corr:       {safe_corr(model_decomp["trend"], gt_decomp["trend"]):+.4f}')
    print(f'Seasonality corr: {safe_corr(model_decomp["seasonal"], gt_decomp["seasonal"]):+.4f}')
    print(f'Volatility corr:  {safe_corr(model_decomp["volatility"], gt_decomp["volatility"]):+.4f}')
    if len(model_decomp["forecast"]) >= len(future):
        print(f'Forecast corr:    {safe_corr(model_decomp["forecast"], future):+.4f}')

if __name__ == '__main__':
    main()
