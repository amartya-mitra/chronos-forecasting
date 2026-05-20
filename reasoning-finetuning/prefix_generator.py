#!/usr/bin/env python3
"""
PrefixGenerator — Step 2 of decomposition-structured prefix tuning.

Maps three STL decomposition components (trend, seasonal, noise) into a
list of per-layer prefix KV tuples for injection into frozen Chronos
attention layers.

Architecture
------------
  encoder (shared):
    Conv1d(1 → 32, kernel=7, padding=3)
    GELU
    AdaptiveAvgPool1d(32)   ← context-length-agnostic: any input length → 32
    Flatten
    Linear(32 × 32 → d_model)

  proj_trend / proj_seasonal / proj_noise:
    ModuleList of num_layers × Sequential(
        Linear(d_model → rank),
        Linear(rank → 2 × m × d_model)
    )  ← low-rank factorisation; rank=64 by default (~20M params total)

  forward output:
    list of num_layers (K, V) tuples, each (batch, 3 × m, d_model)

Reference: prefix_tuning.md § Step 2
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn


class PrefixGenerator(nn.Module):
    """
    Compress STL decomposition components into prefix KV vectors.

    Args:
        d_model:                  Hidden dimension of the target model
                                  (512 for Chronos T5-Small).
        num_layers:               Number of attention layers to generate
                                  prefix KVs for (6 for T5-Small encoder).
        prefix_len_per_component: Number of prefix tokens per component per
                                  layer. Total prefix length = 3 × this value.
        rank:                     Bottleneck rank for the low-rank factored
                                  projection heads. Default 64 gives ~20M
                                  total params vs ~152M for the full-rank
                                  equivalent.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        prefix_len_per_component: int = 16,
        rank: int = 64,
    ) -> None:
        super().__init__()

        self.m = prefix_len_per_component
        self.d = d_model
        self.num_layers = num_layers

        # Shared encoder: AdaptiveAvgPool1d(32) makes this context-length-agnostic —
        # any input length is pooled to a fixed 32-step representation.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(32 * 32, d_model),
        )

        def _factored_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, rank),
                nn.Linear(rank, 2 * prefix_len_per_component * d_model),
            )

        # Per-component projection heads: one factored (d_model→rank→2md) block
        # per attention layer.  Low-rank bottleneck keeps total params ~20M.
        self.proj_trend    = nn.ModuleList([_factored_head() for _ in range(num_layers)])
        self.proj_seasonal = nn.ModuleList([_factored_head() for _ in range(num_layers)])
        self.proj_noise    = nn.ModuleList([_factored_head() for _ in range(num_layers)])

    # ── warm-start ───────────────────────────────────────────────────────────

    def warm_start_from_chronos(self, chronos_model, scale_warm: bool = True) -> None:
        """
        Initialise the output linear of each factored projection head by
        projecting m orthonormal template hidden states through Chronos's
        frozen K and V weight matrices.

        This places prefix KVs inside T5's natural KV subspace from the
        first forward pass, mitigating Failure Mode 2 (prefix ignored by
        attention due to KV space mismatch).

        Root cause of earlier failure
        ------------------------------
        Raw Wk rows are weight matrix rows — they are NOT valid K projection
        outputs.  Natural K vectors are k(h) = h @ Wk.T (a hidden state
        projected through Wk).  Q vectors align with these projections; they
        do not align with raw rows of Wk.  Setting the bias to raw Wk rows
        produced Q·K/√d std ≈ 0.05 (prefix) vs ≈ 0.77 (input) — a 16×
        gap that leaves prefix tokens effectively invisible to softmax.

        Corrected strategy (asymmetric heads to prevent Failure Mode 3)
        ------------------------------------------------------------------
        For each encoder layer i:
          1. Draw m orthonormal template hidden states H ∈ R^(d_model × m).
          2. Compute K_warm = H.T @ Wk.T  and  V_warm = H.T @ Wv.T.
          3. Apply per-head initialisation to break symmetry at init:

             Trend    head: K slot ← K_warm,  V slot ← V_warm
                            (Wk-based, same as before)
             Seasonal head: K slot ← V_warm,  V slot ← K_warm
                            (swapped source: Wv-based K slot, Wk-based V slot)
             Residual head: weight ← Kaiming uniform,  bias ← 0
                            (fully independent from both Wk and Wv)

          All three heads scale their weights to 0.01 first so the
          bias dominates at init, except the residual head which uses
          fresh Kaiming weights (already in the right magnitude range).

        This ensures the three heads start from orthogonal subspaces:
        trend (Wk subspace), seasonal (Wv subspace), residual (random).
        Without this, all three heads share the same bias (Failure Mode 3:
        cos=1.0 at init, may not diverge if loss gradient is small).

        H uses a fixed per-layer seed for full determinism.

        Args:
            chronos_model: pipeline.model (ChronosModel); must have
                           model.encoder.block[i].layer[0].SelfAttention
                           accessible (standard Chronos T5-Small layout).
            scale_warm:   If True (default), multiply the outer linear weight
                          by 0.01 for trend and seasonal heads so the copied
                          bias dominates at init.  Set False (Option A) to
                          keep full-magnitude weights alongside the bias.
        """
        with torch.no_grad():
            for i in range(self.num_layers):
                attn = chronos_model.model.encoder.block[i].layer[0].SelfAttention
                Wk = attn.k.weight.data   # (inner_dim=512, d_model=512)
                Wv = attn.v.weight.data   # (inner_dim=512, d_model=512)

                # m orthonormal template hidden states (reproducible per layer)
                gen = torch.Generator().manual_seed(42 + i)
                H_raw = torch.randn(self.d, self.m, generator=gen, dtype=Wk.dtype)
                H, _ = torch.linalg.qr(H_raw)   # (d_model, m), columns orthonormal
                H = H[:, : self.m]               # ensure exactly m columns

                # Valid K/V projections: K_warm[p] = H[:, p] @ Wk.T
                K_warm = H.T @ Wk.T   # (m, inner_dim) — Wk subspace
                V_warm = H.T @ Wv.T   # (m, inner_dim) — Wv subspace

                b_K = K_warm.reshape(-1)   # (m · d_model,)
                b_V = V_warm.reshape(-1)

                # ── Trend: Wk-based K slot, Wv-based V slot ─────────────────
                out_trend = self.proj_trend[i][1]
                if scale_warm:
                    out_trend.weight.data.mul_(0.01)
                out_trend.bias.data[: self.m * self.d].copy_(b_K)
                out_trend.bias.data[self.m * self.d :].copy_(b_V)

                # ── Seasonal: Wv-based K slot, Wk-based V slot (swapped) ────
                out_seasonal = self.proj_seasonal[i][1]
                if scale_warm:
                    out_seasonal.weight.data.mul_(0.01)
                out_seasonal.bias.data[: self.m * self.d].copy_(b_V)
                out_seasonal.bias.data[self.m * self.d :].copy_(b_K)

                # ── Residual: Kaiming uniform — fully independent of Wk/Wv ──
                out_noise = self.proj_noise[i][1]
                nn.init.kaiming_uniform_(out_noise.weight.data)
                nn.init.zeros_(out_noise.bias.data)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Accept a numpy array or tensor; return a float32 tensor with a batch
        dimension.  A 1-D input (single series) is unsqueezed to (1, T).
        A 2-D input (batch, T) is returned as-is.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)      # (T,) → (1, T)
        return x                    # (batch, T)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        trend:    Union[np.ndarray, torch.Tensor],
        seasonal: Union[np.ndarray, torch.Tensor],
        noise:    Union[np.ndarray, torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            trend:    (context_length,) or (batch, context_length)
            seasonal: same shape as trend
            noise:    same shape as trend

        Returns:
            List of num_layers (K, V) tuples.
            Each K and V has shape (batch, 3 × m, d_model).
        """
        # Add channel dim for Conv1d: (batch, 1, context_length)
        t = self._to_tensor(trend).unsqueeze(1)
        s = self._to_tensor(seasonal).unsqueeze(1)
        n = self._to_tensor(noise).unsqueeze(1)

        # Shared encoder → (batch, d_model)
        h_t = self.encoder(t)
        h_s = self.encoder(s)
        h_n = self.encoder(n)

        prefix_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.num_layers):
            # Project and split into K, V: each (batch, m, d_model)
            Kt, Vt = self.proj_trend[l](h_t).view(-1, 2, self.m, self.d).unbind(1)
            Ks, Vs = self.proj_seasonal[l](h_s).view(-1, 2, self.m, self.d).unbind(1)
            Kn, Vn = self.proj_noise[l](h_n).view(-1, 2, self.m, self.d).unbind(1)

            K = torch.cat([Kt, Ks, Kn], dim=1)   # (batch, 3m, d_model)
            V = torch.cat([Vt, Vs, Vn], dim=1)
            prefix_kvs.append((K, V))

        return prefix_kvs


# ── unit test ─────────────────────────────────────────────────────────────────

def _run_unit_test() -> None:
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(REPO_ROOT / "src"))

    D_MODEL     = 512
    NUM_LAYERS  = 6
    PREFIX_LEN  = 16
    BATCH       = 2
    CONTEXT_LEN = 512   # SarSim0 context length; encoder handles any length

    print("=" * 60)
    print("PrefixGenerator — Unit Test")
    print("=" * 60)
    RANK = 64

    print(f"  d_model={D_MODEL}  num_layers={NUM_LAYERS}  "
          f"prefix_len_per_component={PREFIX_LEN}  rank={RANK}  batch={BATCH}")
    print()

    pg = PrefixGenerator(
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        prefix_len_per_component=PREFIX_LEN,
        rank=RANK,
    )
    pg.train()

    # ── assertion e: trainable parameter count ────────────────────────────
    n_trainable = sum(p.numel() for p in pg.parameters() if p.requires_grad)
    print(f"[e] Total trainable parameters: {n_trainable:,}")

    # ── assertion c: simulate frozen Chronos; verify gradient isolation ───
    print("\n[c] Loading frozen Chronos T5-Small …")
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32,
    )
    chronos_model = pipeline.model
    for param in chronos_model.parameters():
        param.requires_grad_(False)

    n_chronos_grad = sum(1 for p in chronos_model.parameters() if p.requires_grad)
    assert n_chronos_grad == 0, \
        f"Expected 0 trainable Chronos params; got {n_chronos_grad}"
    print(f"    Chronos trainable params after freeze : {n_chronos_grad}  ✓")

    all_pg_have_grad = all(p.requires_grad for p in pg.parameters())
    assert all_pg_have_grad, "All PrefixGenerator params must have requires_grad=True"
    print(f"    All PrefixGenerator params require grad : {all_pg_have_grad}  ✓")

    # ── assertions a, b: output shape ────────────────────────────────────
    print("\n[a,b] Forward pass with random numpy inputs …")
    rng = np.random.default_rng(0)
    trend_np    = rng.standard_normal((BATCH, CONTEXT_LEN)).astype(np.float32)
    seasonal_np = rng.standard_normal((BATCH, CONTEXT_LEN)).astype(np.float32)
    noise_np    = rng.standard_normal((BATCH, CONTEXT_LEN)).astype(np.float32)

    prefix_kvs = pg(trend_np, seasonal_np, noise_np)

    # (a) list of num_layers (K, V) tuples
    assert isinstance(prefix_kvs, list), "Output must be a list"
    assert len(prefix_kvs) == NUM_LAYERS, \
        f"Expected {NUM_LAYERS} (K,V) pairs; got {len(prefix_kvs)}"
    print(f"    Output list length = {len(prefix_kvs)}  (== num_layers)  ✓")

    # (b) K and V shape: (batch, 3 * prefix_len, d_model) = (2, 48, 512)
    expected_seq = 3 * PREFIX_LEN   # 48
    for i, (K, V) in enumerate(prefix_kvs):
        assert K.shape == (BATCH, expected_seq, D_MODEL), \
            f"Layer {i} K shape mismatch: expected {(BATCH, expected_seq, D_MODEL)}, got {tuple(K.shape)}"
        assert V.shape == (BATCH, expected_seq, D_MODEL), \
            f"Layer {i} V shape mismatch: expected {(BATCH, expected_seq, D_MODEL)}, got {tuple(V.shape)}"
    K0, V0 = prefix_kvs[0]
    print(f"    K shape = {tuple(K0.shape)}  (batch, 3×m, d_model)  ✓")
    print(f"    V shape = {tuple(V0.shape)}  (batch, 3×m, d_model)  ✓")

    # ── assertion d: forward + backward without error ─────────────────────
    print("\n[d] Backward pass …")
    # Re-run forward to get a fresh graph (previous forward was outside autograd)
    pg.zero_grad()
    prefix_kvs2 = pg(trend_np, seasonal_np, noise_np)
    loss: torch.Tensor = sum(K.sum() + V.sum() for K, V in prefix_kvs2)  # type: ignore[assignment]
    loss.backward()

    # Verify PrefixGenerator grads are populated
    n_pg_grads = sum(1 for p in pg.parameters() if p.grad is not None)
    assert n_pg_grads > 0, "Expected PrefixGenerator params to have gradients after backward"
    print(f"    Backward completed without error  ✓")
    print(f"    PrefixGenerator params with non-None grad : {n_pg_grads}  ✓")

    # Verify Chronos params have no gradients
    chronos_grads = [n for n, p in chronos_model.named_parameters() if p.grad is not None]
    assert len(chronos_grads) == 0, \
        f"Chronos params should have no gradients; found: {chronos_grads[:3]}"
    print(f"    Chronos params with non-None grad : {len(chronos_grads)} (correctly excluded)  ✓")

    print()
    print("=" * 60)
    print("ALL ASSERTIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_unit_test()
