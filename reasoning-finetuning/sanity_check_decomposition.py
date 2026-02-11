#!/usr/bin/env python3
"""
Dataset Sanity Check: Verify that reasoning-tagged data points' decomposition
tokens (trend, seasonality, volatility) match freshly computed STL decomposition
from the input context.

Steps:
  1. Extract reasoning_tokens from each reasoning-tagged sample
  2. Split tokens into trend (0-63), seasonal (64-127), volatility (128-191),
     forecast (192-255) slots
  3. Detokenize each component using the stored context_scale
  4. De-amplify: seasonal ÷ 10, volatility ÷ 50
  5. Extract the context portion and compute fresh STL decomposition
  6. Compare stored vs recomputed components (correlation + MSE)
  7. Report violations (correlation < threshold)

Usage:
    python reasoning-finetuning/sanity_check_decomposition.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import torch
import datasets
from statsmodels.tsa.seasonal import STL
from chronos import ChronosConfig

# Constants matching prepare_dataset.py
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
DECOMPOSITION_LENGTH = 64
SEASONAL_AMP = 10.0
VOLATILITY_AMP = 50.0

# Violation thresholds
CORR_THRESHOLD = 0.7        # Minimum acceptable correlation
MSE_RELATIVE_THRESHOLD = 1.0  # Maximum acceptable relative MSE

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'gifteval-reasoning.arrow')


def create_tokenizer():
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    return config.create_tokenizer()


def detokenize_component(token_ids, scale, tokenizer):
    """
    Convert token IDs back to real values using the tokenizer's bin centers
    and the stored scale.
    """
    token_tensor = torch.tensor(token_ids).long().unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    values = tokenizer.output_transform(token_tensor, scale)  # [1, 1, N]
    return values[0, 0, :].numpy()


def compute_stl(series, period=7):
    """Compute STL decomposition on a series."""
    series_clean = pd.Series(series).ffill().bfill().values
    if len(series_clean) < 2 * period:
        pad = 2 * period - len(series_clean)
        series_clean = np.concatenate([np.full(pad, series_clean[0]), series_clean])

    try:
        stl = STL(series_clean, period=period, robust=True)
        result = stl.fit()
        vol_window = max(3, period // 2)
        volatility = pd.Series(result.resid).rolling(
            window=vol_window, min_periods=1
        ).std().values
        return {
            'trend': result.trend[-DECOMPOSITION_LENGTH:],
            'seasonal': result.seasonal[-DECOMPOSITION_LENGTH:],
            'volatility': volatility[-DECOMPOSITION_LENGTH:],
        }
    except Exception as e:
        return None


def safe_corr(a, b):
    ml = min(len(a), len(b))
    a, b = a[:ml], b[:ml]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def relative_mse(a, b):
    ml = min(len(a), len(b))
    a, b = a[:ml], b[:ml]
    mse = np.mean((a - b) ** 2)
    var = np.var(b) + 1e-12
    return float(mse / var)


def main():
    print("=" * 70)
    print("DATASET SANITY CHECK: Decomposition Token Alignment")
    print("=" * 70)

    tokenizer = create_tokenizer()
    ds = datasets.load_dataset('arrow', data_files=DATA_PATH, split='train')

    # Filter to reasoning-mode samples
    reasoning_samples = [i for i in range(len(ds)) if ds[i]['mode'] == 'reasoning']
    print(f"\nTotal samples: {len(ds)}")
    print(f"Reasoning samples: {len(reasoning_samples)}")
    print(f"Fast samples: {len(ds) - len(reasoning_samples)}")

    # Results accumulators
    violations = {
        'trend': [],
        'seasonal': [],
        'volatility': [],
    }
    correlations = {
        'trend': [],
        'seasonal': [],
        'volatility': [],
    }
    rel_mses = {
        'trend': [],
        'seasonal': [],
        'volatility': [],
    }
    skipped = 0
    total_checked = 0

    print(f"\nProcessing {len(reasoning_samples)} reasoning samples...")
    print("-" * 70)

    for idx in reasoning_samples:
        sample = ds[idx]
        target = np.array(sample['target'])
        reasoning_tokens = sample['reasoning_tokens']
        stored_scale = sample.get('context_scale', None)

        # ---- Step 1: Extract context using same logic as prepare_dataset.py ----
        total_needed = CONTEXT_LENGTH + PREDICTION_LENGTH
        if len(target) >= total_needed:
            context = target[-(total_needed):-PREDICTION_LENGTH]
        else:
            sp = len(target) - PREDICTION_LENGTH
            if sp <= 0:
                skipped += 1
                continue
            context = target[:sp]

        # ---- Step 2: Split reasoning_tokens into component slots ----
        if len(reasoning_tokens) < 4 * DECOMPOSITION_LENGTH:
            print(f"  Sample {idx}: SKIP — only {len(reasoning_tokens)} tokens "
                  f"(expected {4 * DECOMPOSITION_LENGTH})")
            skipped += 1
            continue

        trend_tokens = reasoning_tokens[0:DECOMPOSITION_LENGTH]
        seasonal_tokens = reasoning_tokens[DECOMPOSITION_LENGTH:2*DECOMPOSITION_LENGTH]
        volatility_tokens = reasoning_tokens[2*DECOMPOSITION_LENGTH:3*DECOMPOSITION_LENGTH]

        # ---- Step 3: Detokenize using stored context_scale ----
        if stored_scale is None or stored_scale == 0:
            # Recompute scale from context
            ctx_tensor = torch.tensor(context).float().unsqueeze(0)
            _, _, scale_t = tokenizer.context_input_transform(ctx_tensor)
            scale = scale_t
        else:
            scale = torch.tensor([stored_scale], dtype=torch.float32)

        trend_vals = detokenize_component(trend_tokens, scale, tokenizer)
        seasonal_vals_amp = detokenize_component(seasonal_tokens, scale, tokenizer)
        volatility_vals_amp = detokenize_component(volatility_tokens, scale, tokenizer)

        # ---- Step 4: De-amplify ----
        seasonal_vals = seasonal_vals_amp / SEASONAL_AMP
        volatility_vals = volatility_vals_amp / VOLATILITY_AMP

        # ---- Step 5: Compute fresh STL from context ----
        stl = compute_stl(context, period=7)
        if stl is None:
            print(f"  Sample {idx}: SKIP — STL failed")
            skipped += 1
            continue

        total_checked += 1

        # ---- Step 6: Compare ----
        for comp_name, stored, recomputed in [
            ('trend', trend_vals, stl['trend']),
            ('seasonal', seasonal_vals, stl['seasonal']),
            ('volatility', volatility_vals, stl['volatility']),
        ]:
            corr = safe_corr(stored, recomputed)
            rmse = relative_mse(stored, recomputed)
            correlations[comp_name].append(corr)
            rel_mses[comp_name].append(rmse)

            if corr < CORR_THRESHOLD:
                violations[comp_name].append({
                    'sample_idx': idx,
                    'correlation': corr,
                    'relative_mse': rmse,
                })

    # ============================================================
    # REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("SANITY CHECK REPORT")
    print("=" * 70)

    print(f"\nSamples checked:  {total_checked}")
    print(f"Samples skipped:  {skipped}")
    print(f"Correlation threshold: {CORR_THRESHOLD}")

    print("\n--- Aggregate Statistics ---")
    print(f"{'Component':<14} {'Mean Corr':>10} {'Std Corr':>10} {'Min Corr':>10} "
          f"{'Max Corr':>10} {'Violations':>12}")
    print("-" * 70)

    total_violations = 0
    for comp in ['trend', 'seasonal', 'volatility']:
        c = correlations[comp]
        v = violations[comp]
        total_violations += len(v)
        if len(c) > 0:
            print(f"{comp:<14} {np.mean(c):>+10.4f} {np.std(c):>10.4f} "
                  f"{np.min(c):>+10.4f} {np.max(c):>+10.4f} "
                  f"{len(v):>8}/{total_checked}")
        else:
            print(f"{comp:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>8}/{total_checked}")

    print(f"\n{'TOTAL VIOLATIONS':>14}: {total_violations} across "
          f"{total_checked * 3} component checks "
          f"({total_violations/(total_checked*3)*100:.1f}% violation rate)")

    # Per-component violation details
    for comp in ['trend', 'seasonal', 'volatility']:
        v = violations[comp]
        if len(v) > 0:
            print(f"\n--- {comp.upper()} Violations ({len(v)} samples) ---")
            for vi in v[:10]:  # Show first 10
                print(f"  Sample {vi['sample_idx']:>4}: "
                      f"corr={vi['correlation']:+.4f}, "
                      f"rel_mse={vi['relative_mse']:.4f}")
            if len(v) > 10:
                print(f"  ... and {len(v) - 10} more")

    # Summary verdict
    print("\n" + "=" * 70)
    violation_rate = total_violations / (total_checked * 3) if total_checked > 0 else 0
    if violation_rate < 0.05:
        print("VERDICT: ✅ PASS — Dataset decomposition tokens align well with STL")
    elif violation_rate < 0.20:
        print("VERDICT: ⚠️  MARGINAL — Some decomposition misalignment detected")
    else:
        print("VERDICT: ❌ FAIL — Significant decomposition misalignment")
    print("=" * 70)


if __name__ == '__main__':
    main()
