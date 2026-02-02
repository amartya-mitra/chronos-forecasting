# Reasoning Mode Finetuning for Chronos-2

This document describes the **Reasoning Mode Finetuning** exercise for the Chronos-2 time series forecasting model. The goal is to extend the base Chronos model with a "reasoning mode" that can generate interpretable decomposition components (trend, seasonality, volatility) before producing forecasts.

## Overview

### Objective

Train Chronos-2-Small to support two operating modes:

| Mode | Control Token | Output |
|------|---------------|--------|
| **Fast Mode** | `<\|fast_mode\|>` (token 4096) | Direct forecast (64 tokens) |
| **Reasoning Mode** | `<\|reasoning_mode\|>` (token 4097) | Decomposition (192 tokens) + Forecast (64 tokens) = 256 tokens total |

### Architecture Changes

1. **Vocabulary Extension**: Append 2 control tokens to the original 4096-token vocabulary (→ 4098 total)
2. **Max Generation Length**: Extended from 20 to 300 tokens for reasoning mode
3. **Output Format** (Reasoning Mode):
   ```
   [Trend: 64 tokens] [Seasonality: 64 tokens] [Volatility: 64 tokens] [Forecast: 64 tokens]
   ```

## Dataset Options

Two dataset sources are supported:

### Option 1: GiftEval (STL Decomposition)

```bash
python reasoning-finetuning/prepare_dataset.py --dataset-source gifteval
```

- **Source**: GiftEval time series benchmark
- **Decomposition**: Approximated via STL (`statsmodels.tsa.seasonal.STL`)
- **Output**: `data/gifteval-reasoning.arrow`

### Option 2: SarSim0 (Synthetic with Known Decomposition)

```bash
python reasoning-finetuning/prepare_dataset.py --dataset-source sarsim0 --num-samples 1000
```

- **Source**: SarSim0 synthetic generator (`synth-data/SarSim0-main/`)
- **Decomposition**: Exact components from generation (trend, seasonal, volatility)
- **Output**: `data/sarsim0-reasoning.arrow`

> **Advantage**: SarSim0 provides *exact* decomposition, not approximated. This should improve reasoning mode training.

### SarSim0 Validation

The synthetic data was validated against the base Chronos model:

| Metric | Value |
|--------|-------|
| Samples Tested | 6 |
| Average Correlation | **+0.68** |

![SarSim0 Validation](figures/sarsim0_validation.png)

*Green: Target from SarSim0 | Blue dashed: Base model prediction*

### Sample Structure

## Training Configuration

```yaml
# Base Model
model_id: amazon/chronos-t5-small

# Training Parameters
max_steps: 1000
learning_rate: 1e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
per_device_train_batch_size: 16
gradient_accumulation_steps: 2

# Sequence Parameters
context_length: 512
prediction_length: 64
max_new_tokens: 300  # For reasoning mode

# Evaluation
eval_steps: 200
save_steps: 200
```

## Project Structure

```
chronos-forecasting/
├── reasoning-finetuning/         # Reasoning mode extension
│   ├── README.md                 # This file
│   ├── model_utils.py            # Model availability & download
│   ├── prepare_dataset.py        # Generate training data
│   ├── sarsim0_adapter.py        # SarSim0 → reasoning format
│   ├── validate_sarsim0.py       # Validate synthetic data
│   ├── train.py                  # Training wrapper
│   ├── verify_fast_mode.py       # Fast mode verification
│   ├── verify_reasoning.py       # Reasoning mode verification
│   ├── configs/default.yaml      # Training config
│   ├── data/                     # Dataset files (gitignored)
│   └── figures/                  # Plots (gitignored)
├── synth-data/SarSim0-main/      # Synthetic data generator
├── output/                       # Model outputs (gitignored)
└── .gitignore                    
```

## Quick Start

### 1. Setup Environment
```bash
cd chronos-forecasting
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install statsmodels fire datasets
```

### 2. Check/Download Models
```bash
python reasoning-finetuning/model_utils.py --check-base
python reasoning-finetuning/model_utils.py --check-finetuned
```

### 3. Prepare Training Data
```bash
# Option A: GiftEval (requires gifteval-subset.arrow)
python reasoning-finetuning/prepare_dataset.py --dataset-source gifteval

# Option B: SarSim0 synthetic data (recommended)
python reasoning-finetuning/prepare_dataset.py --dataset-source sarsim0 --num-samples 1000
```

### 4. Validate Synthetic Data (optional)
```bash
python reasoning-finetuning/validate_sarsim0.py
```

### 5. Train the Model
```bash
python reasoning-finetuning/train.py reasoning-finetuning/configs/default.yaml
```

### 6. Verify Fast Mode
```bash
python reasoning-finetuning/verify_fast_mode.py
```
Expected: Correlation ≥ 0.99 between finetuned (fast mode) and base model.

### 7. Verify Reasoning Mode
```bash
python reasoning-finetuning/verify_reasoning.py
```
Compares model's decomposition against STL ground truth.

## Results

### Fast Mode Verification ✅
| Metric | Finetuned | Base Model |
|--------|-----------|------------|
| MSE | 0.054 | 0.048 |
| Correlation | +0.998 | +0.998 |
| **Between Models** | **+0.999** | - |

Fast mode preserves the original forecasting quality.

### Reasoning Mode Verification ⚠️
| Component | Correlation with STL |
|-----------|---------------------|
| Trend | +0.08 |
| Seasonality | -0.25 |
| Volatility | -0.01 |

Reasoning mode needs more training data and steps to learn accurate decomposition.

## Files to Exclude from Git

The following are automatically generated and should not be committed:
- `output/` - Model checkpoints (~185MB each)
- `*.arrow` - Dataset files (up to 200MB)
- `*.log` - Training logs
- `*.png` - Verification plots
- `.venv/` - Virtual environment

See `.gitignore` for the complete list.

## Regenerating from Scratch

If you clone this repository without the model files:

```bash
# 1. Download base model (automatic on first use)
python -c "from chronos import ChronosPipeline; ChronosPipeline.from_pretrained('amazon/chronos-t5-small')"

# 2. Generate training data
python reasoning-finetuning/prepare_dataset.py

# 3. Train finetuned model
python reasoning-finetuning/train.py reasoning-finetuning/configs/default.yaml
```

## Known Issues & Future Work

1. **Reasoning accuracy**: Current decomposition correlation is low; needs more training data
2. **Training data diversity**: Consider using multiple STL decomposition methods
3. **Hyperparameter tuning**: Learning rate and batch size may need adjustment
4. **Evaluation metrics**: Add MASE, CRPS for forecast quality assessment

## References

- [Chronos Paper](https://arxiv.org/abs/2403.07815) - Language Models are Zero-Shot Time Series Forecasters
- [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [STL Decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html)
