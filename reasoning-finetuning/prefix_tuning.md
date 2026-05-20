Overview

This task documents the design, implementation steps, and failure modes for Option C: Decomposition-Structured Prefix Tuning with Chronos — a method to improve TSFM forecasting by injecting an explicit, decomposition-aware prefix into frozen Chronos attention layers, without requiring the model to generate decomposition tokens as output.

Core idea

Instead of asking Chronos to output a structured decomposition (trend → seasonal → noise → forecast), the decomposition is computed externally via STL and compressed into the prefix KV vectors. Chronos then forecasts as usual, enriched by a prefix that encodes the decomposition content of the input context.

Pipeline:

[INPUT x]
    |
    v
STL decomposition
    |
    +---> trend component
    |
    +---> seasonal component
    |
    +---> residual / noise component
    |
    v
Three separate learned projection heads
    |
    v
P_trend || P_seasonal || P_noise  (concatenated prefix KVs)
    |
    v
Injected into each attention layer of frozen Chronos encoder + decoder
    |
    v
[FORECAST]  (same output format as zero-shot Chronos)

Implementation Steps

Step 1 — STL decomposition of training corpus

Run STL decomposition over the full training dataset. Store trend, seasonal, and residual components alongside the raw series.

Period must be specified per dataset / frequency. Use FFT-based dominant frequency detection for automatic period estimation.
For multi-frequency series (e.g. hourly with daily + weekly seasonality), use MSTL.
Inspect residual variance across a sample of series before proceeding. If residual variance ≈ raw signal variance, STL has failed and must be replaced.
```
from statsmodels.tsa.seasonal import STL
import numpy as np

def decompose(series, period):
    result = STL(series, period=period).fit()
    return result.trend, result.seasonal, result.resid
```
Step 2 — Implement the prefix generator g(x)

Build a small module with a shared encoder and three separate projection heads — one per decomposition component. Each head projects its component into the KV space of Chronos's frozen attention layers.

class PrefixGenerator(nn.Module):
    def __init__(self, context_len, d_model, num_layers, prefix_len_per_component):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(32 * 32, d_model)
        )
        self.proj_trend    = nn.ModuleList([nn.Linear(d_model, 2 * prefix_len_per_component * d_model) for _ in range(num_layers)])
        self.proj_seasonal = nn.ModuleList([nn.Linear(d_model, 2 * prefix_len_per_component * d_model) for _ in range(num_layers)])
        self.proj_noise    = nn.ModuleList([nn.Linear(d_model, 2 * prefix_len_per_component * d_model) for _ in range(num_layers)])
        self.m = prefix_len_per_component
        self.d = d_model

    def forward(self, trend, seasonal, noise):
        h_t = self.encoder(trend.unsqueeze(1))
        h_s = self.encoder(seasonal.unsqueeze(1))
        h_n = self.encoder(noise.unsqueeze(1))

        prefix_kvs = []
        for l in range(len(self.proj_trend)):
            Kt, Vt = self.proj_trend[l](h_t).view(-1, 2, self.m, self.d).unbind(1)
            Ks, Vs = self.proj_seasonal[l](h_s).view(-1, 2, self.m, self.d).unbind(1)
            Kn, Vn = self.proj_noise[l](h_n).view(-1, 2, self.m, self.d).unbind(1)
            K = torch.cat([Kt, Ks, Kn], dim=1)   # (batch, 3m, d_model)
            V = torch.cat([Vt, Vs, Vn], dim=1)
            prefix_kvs.append((K, V))
        return prefix_kvs

Step 3 — Inject prefix KVs into frozen Chronos attention layers

Hook into each attention layer's key and value computation to prepend the generated prefix vectors. Chronos backbone weights remain frozen throughout.

def prepend_prefix(layer_idx, K_orig, V_orig, prefix_kvs):
    P_K, P_V = prefix_kvs[layer_idx]
    K = torch.cat([P_K, K_orig], dim=1)
    V = torch.cat([P_V, V_orig], dim=1)
    return K, V

Step 4 — Training objective

Train only the projection heads (and shared encoder) in g(x). All Chronos parameters are frozen. Use forecasting loss as the primary objective.

# Freeze Chronos
for param in chronos_model.parameters():
    param.requires_grad = False

# Only g(x) parameters are trainable
optimizer = torch.optim.AdamW(prefix_generator.parameters(), lr=1e-4)

# Training step
prefix_kvs = prefix_generator(trend, seasonal, noise)
forecast = chronos_model(input_tokens, prefix_kvs=prefix_kvs)
loss = nll_loss(forecast, target)

# Optional: orthogonality regularization (see Failure Mode 3)
# loss += lambda_ortho * ortho_penalty(prefix_kvs)

loss.backward()
optimizer.step()

Step 5 — Ablations to run before scaling

Prefix length per component: m ∈ {8, 16, 32, 64}
With vs without shared encoder (separate encoders per component)
Forecasting loss only vs forecasting + orthogonality penalty
Prefix injection in encoder only vs encoder + decoder

Failure Modes

Failure 1 — STL decomposition quality (upstream)

What goes wrong: Poor period estimation or non-stationary structure causes STL to return components that don't reflect the true signal. Garbage input → garbage prefix.

How to detect: Compute residual variance / raw signal variance across training series. If ratio > 0.5 on average, STL is underperforming.

Recovery:

Use MSTL for multi-frequency series
Use FFT to auto-detect dominant period
For non-stationary series: use rolling STL over sub-windows and average representations

Failure 2 — KV space mismatch (prefix ignored by attention)

What goes wrong: The projected STL embeddings land outside the distribution of KV vectors seen during Chronos pretraining. The frozen attention layers assign near-zero weight to prefix tokens and effectively ignore them. Forecast improvement is zero.

How to detect: Log mean attention weight on prefix tokens vs input tokens across all layers during training. If prefix attention weight stays near zero and does not increase, this failure is occurring.

Recovery:

Initialize projection head weights from Chronos's own frozen K/V projection matrices (warm start in the right space)
Add a prefix attention regularization term penalizing near-zero prefix attention weights

Failure 3 — Trivial collapse across the three heads

What goes wrong: P_trend, P_seasonal, and P_noise converge to nearly identical representations. The forecasting loss does not require them to be distinct, so they collapse. The structured decomposition in the prefix is lost.

How to detect: Compute pairwise cosine similarity between the three sub-prefix vectors on a held-out batch after each epoch. If consistently > 0.9, collapse is occurring.

Recovery: Add an orthogonality penalty to training loss:

def ortho_penalty(P_t, P_s, P_n):
    def cos_sim(a, b):
        a, b = a.flatten(1), b.flatten(1)
        return F.cosine_similarity(a, b, dim=1).mean()
    return cos_sim(P_t, P_s)**2 + cos_sim(P_t, P_n)**2 + cos_sim(P_s, P_n)**2

Failure 4 — Prefix length too short (information bottleneck)

What goes wrong: m is too small to faithfully encode the trend or seasonal curve. The prefix stores coarse summary statistics only ("trend is increasing") rather than the actual curve content — equivalent to Option A's failure mode.

How to detect: Train a small reconstruction decoder on top of each sub-prefix vector and measure reconstruction MSE against the STL components. High reconstruction error = insufficient prefix capacity.

Recovery: Ablate over m ∈ {8, 16, 32, 64} per component. A useful heuristic: m should scale with the dominant frequency complexity of the component being encoded.

Failure 5 — Projection overfitting (poor generalization)

What goes wrong: Projection heads overfit to the decomposition patterns in the training corpus. Strong in-sample improvement but poor performance on held-out series or different datasets.

How to detect: Evaluate on a dataset not seen during training (e.g. train on M4 Hourly, evaluate on ETT). A large train/eval gap indicates overfitting.

Recovery:

Apply dropout and weight decay to projection heads
Diversify training corpus (mixed frequencies, domains, decomposition profiles)

Failure 6 — Non-stationarity within the context window

What goes wrong: STL assumes a stable decomposition structure across the entire context window. For series with structural breaks or time-varying seasonality, STL returns a blurred, averaged decomposition that misrepresents the local signal dynamics.

How to detect: Inspect STL outputs on series with known regime changes. If trend component shows large discontinuities or seasonal component shows phase drift, this is occurring.

Recovery:

Use rolling STL over sub-windows (e.g. 3 overlapping windows) and concatenate or average the resulting representations
Switch to a learnable decomposition filter (moving average bank) as the upstream decomposer

Summary Table

## Handover Context (for Claude Code)

- Model: `amazon/chronos-t5-small` (HuggingFace T5)
  - d_model = 512, num_encoder_layers = 6, num_decoder_layers = 6
- KV injection mechanism: use `register_forward_hook` on each
  `T5Attention` block to prepend prefix KVs before the attention op
- Data paths:
  - SarSim0: `reasoning-finetuning/data/sarsim0-reasoning.arrow`
  - GiftEval: `reasoning-finetuning/data/gifteval-reasoning.arrow`
- Existing relevant file: `reasoning-finetuning/prepare_dataset.py`
  (has `compute_decomposition()` — reuse, don't rewrite STL)
- Today's scope (May 12): Steps 1–2 only. Do not attempt injection yet.

Last updated: 2026-05-12. Based on design discussions with Claude (claude.ai).