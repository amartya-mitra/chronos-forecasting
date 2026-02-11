# Reasoning Mode Finetuning for Chronos-2

This project extends the **Chronos-2-Small** forecasting model with a "reasoning mode" that generates interpretable decomposition components (trend, seasonality, volatility) before producing forecasts.

## 🚀 Project Status & Handover

**Current State**: ✅ Code Complete & Verified | ⚠️ Model Needs Retraining

We have successfully implemented and verified the entire pipeline for fine-tuning Chronos with reasoning capabilities.

### Key Achievements
1.  **Dataset Preparation**:
    *   **Goal**: Create a dataset with 50% "fast mode" (standard forecasting) and 50% "reasoning mode" (decomposition + forecasting) samples.
    *   **Implementation**: `prepare_dataset.py` handles both real-world data (GiftEval via STL decomposition) and synthetic data (SarSim0 with ground truth components).
    *   **Logic**:
        *   **Fast Mode**: `<|fast_mode|>` token → Forecast (64 tokens).
        *   **Reasoning Mode**: `<|reasoning_mode|>` token → Trend (64) → Seasonality (64) → Volatility (64) → Forecast (64).
        *   **Scaling**: All components are tokenized using the **context scale** to ensure consistency.
        *   **Amplification**: Small-magnitude components are amplified before tokenization (Seasonality ×10, Volatility ×50) to preserve resolution.

2.  **Training Pipeline (Critical Fixes)**:
    *   **Issue**: Initial training attempts revealed a scale mismatch. The `InstanceSplitter` in GluonTS randomly samples context windows, which could have a different scale than the pre-computed reasoning tokens in the dataset.
    *   **Fix**: We modified `train_reasoning.py` to:
        *   **Bypass `InstanceSplitter`** for pre-tokenized data.
        *   Use a **deterministic split** matching the dataset preparation logic.
        *   **Force the encoder to use the stored `context_scale`** from the dataset, ensuring perfect alignment between the input context and the target reasoning tokens.
    *   **Verification**: A smoke test confirmed the correct tensor shapes and token alignment.

3.  **Verification Suite**:
    *   **Fast Mode**: `verify_fast_mode.py` confirms the finetuned model matches the base model's performance (Correlation > 0.99).
    *   **Reasoning Mode**: `verify_reasoning.py` checks decomposition accuracy.
    *   **Dataset Sanity Check**: `sanity_check_decomposition.py` (NEW) rigorously verifies that the tokenized decomposition components in the dataset match a fresh STL decomposition of the input context. **Result: 99.4% Pass** (only minor edge cases in volatility).

### Next Steps (Handover)
The code is fixed and verified. The *model checkpoint* currently saved was trained **before** the final scale consistency fix.
**Immediate Action**:
1.  **Retrain the model** using `train_reasoning.py`.
    ```bash
    python reasoning-finetuning/train.py reasoning-finetuning/configs/default.yaml
    ```
2.  **Verify Reasoning Performance**:
    ```bash
    python reasoning-finetuning/verify_reasoning.py
    ```
    Expect improved correlation for Seasonality and Volatility due to the fixed scale consistency.

---

## Architecture Overview

### Modes
| Mode | Control Token | Output Structure |
|------|---------------|------------------|
| **Fast Mode** | `<|fast_mode|>` (4096) | `[Forecast: 64]` |
| **Reasoning** | `<|reasoning_mode|>` (4097) | `[Trend: 64] [Seasonal: 64] [Volatility: 64] [Forecast: 64]` |

### Tokenization Strategy
To ensure the model can learn effectively:
*   **Context Scale**: All inputs and targets are scaled by the mean absolute value of the context.
*   **Amplification**:
    *   **Trend**: ×1 (No change)
    *   **Seasonality**: ×10 (Boosts signal strength in token space)
    *   **Volatility**: ×50 (Boosts signal strength)
    *   *Note*: Inference scripts must divide by these factors to recover original values.

## Quick Start Guide

### 1. Setup
```bash
cd chronos-forecasting
pip install -e ".[dev]"
pip install statsmodels fire datasets
```

### 2. Prepare Data
Choose one source. **SarSim0** is recommended for cleaner ground truth decomposition.
```bash
# Option A: GiftEval (Real-world, STL approximation)
python reasoning-finetuning/prepare_dataset.py --dataset-source gifteval

# Option B: SarSim0 (Synthetic, Exact decomposition)
python reasoning-finetuning/prepare_dataset.py --dataset-source sarsim0 --num-samples 1000
```

### 3. Verify Data Integrity (Recommended)
Run the rigorous sanity check to ensure decomposition tokens align with context.
```bash
python reasoning-finetuning/sanity_check_decomposition.py
```
*Expected: >99% Pass rate.*

### 4. Train
```bash
python reasoning-finetuning/train.py reasoning-finetuning/configs/default.yaml
```

### 5. Verify & Visualize
Generate comparison plots for Fast Mode and Reasoning Mode.
```bash
# Verify Fast Mode quality matches base model
python reasoning-finetuning/verify_fast_mode.py

# Verify Reasoning Decomposition accuracy
python reasoning-finetuning/verify_reasoning.py

# Generate Visual Plots (Comparison & Decomposition Overlay)
python reasoning-finetuning/plot_verification.py
```
*Outputs saved to `reasoning-finetuning/figures/`*

## Key Scripts

| Script | Purpose |
|--------|---------|
| `prepare_dataset.py` | Generates training data (GiftEval/SarSim0) with control tokens & decomposition. |
| `train_reasoning.py` | Main training logic. Handles pre-tokenized data, checks for scale consistency. |
| `sanity_check_decomposition.py` | **Crucial**: Verifies that dataset tokens match input context decomposition. |
| `verify_fast_mode.py` | Compares finetuned model forecast vs base model (should be identical). |
| `verify_reasoning.py` | Checks if model generates valid decomposition components. |
| `plot_verification.py` | Generates visual verification plots. |

## known Issues
- **Reasoning Accuracy**: The previous model checkpoint had low correlation for Seasonality/Volatility. This was likely due to the scale inconsistency we fixed. Retraining is required to confirm the improvement.
