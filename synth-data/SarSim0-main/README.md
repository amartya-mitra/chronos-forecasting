# SarSim0 - SARIMA Simulator for Zero-Shot Forecasting

A PyTorch-based synthetic time series generator implementing the SarSim0 pipeline. Generates diverse, realistic time series for training foundation models like Chronos.

## Pipeline Architecture

```
y = N ∘ I ∘ S(ε)
```

| Stage | Module | Description |
|-------|--------|-------------|
| **S** | `sarima.py` | SARIMA base signal with pole-based stability |
| **I** | `sarima2.py` | SARIMA-2 bi-seasonal interaction layer |
| **N** | `noisers.py` | Heavy-tailed noise (Poisson, Gamma, Lognormal) |

## Quick Start

```python
from sarsim0 import SarSimConfig, SarSim0Generator

# Create generator
config = SarSimConfig(context_window=512, prediction_window=64)
generator = SarSim0Generator(config=config, seed=42)

# Generate batch of time series
context, target = generator.generate_batch(batch_size=256)
# context: (256, 512), target: (256, 64)

# Generate with known decomposition
series = generator.generate_series(batch_size=32, length=1000)
```

## Features

- **Vectorized generation**: Efficient batch processing via PyTorch
- **SARIMA foundation**: Autoregressive models with seasonal components
- **Heavy-tailed noise**: Poisson, Gamma, Lognormal distributions
- **Multivariate support**: Correlated multi-variate series
- **Covariates**: Past-only and future-known covariates
- **DataLoader integration**: Ready for PyTorch training loops

## Configuration

```python
SarSimConfig(
    # SARIMA orders
    p_range=(0, 10),      # AR order
    q_range=(0, 3),       # MA order
    s_range=(0, 52),      # Seasonal period
    
    # Generation
    context_window=4096,
    prediction_window=512,
    burn_in=200,
    
    # Noise
    poisson_lambda_range=(0.1, 100.0),
    gamma_lambda_range=(0.1, 100.0),
    
    # Multivariate
    multivariate_prob=0.3,
    n_variates_range=(2, 5),
)
```

## Modules

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters from paper (Table 9) |
| `sarima.py` | SARIMA generator with pole constraints |
| `sarima2.py` | Bi-seasonal additive/multiplicative composition |
| `noisers.py` | Heavy-tailed noise distributions |
| `pipeline.py` | Full pipeline + DataLoader |
| `multivariate.py` | Multivariate series with correlations |
| `covariates.py` | Past/future covariates |

## Integration with Chronos

### Training DataLoader

```python
from sarsim0 import create_dataloader, SarSimConfig

config = SarSimConfig()
dataloader = create_dataloader(
    batch_size=256, 
    config=config, 
    num_workers=4
)

for context, target in dataloader:
    # Train your model
    pass
```

### Chronos-2 Format

```python
from sarsim0 import generate_mixed_sarsim0_chronos2

# Generate data compatible with Chronos-2 fit()
data = generate_mixed_sarsim0_chronos2(
    num_series=1000,
    context_length=512,
    prediction_length=64,
)
```

## DroPE Fine-tuning

The `chronos-drope/` directory contains scripts for fine-tuning Chronos-2 with DroPE (Dropping Positional Encoding) for better length generalization.

```bash
python chronos-drope/train_chronos2.py \
    --model_id amazon/chronos-bolt-small \
    --num_training_steps 10000 \
    --batch_size 32
```

## References

- [Chronos Paper](https://arxiv.org/abs/2403.07815)
- [DroPE Paper](https://pub.sakana.ai/DroPE/)
