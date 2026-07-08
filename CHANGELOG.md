# Changelog

## [Unreleased]

### Phase 1 — Internal Decomposition Probe (2026-07-08)

**Question**: Does frozen Chronos-T5-Small already encode trend/seasonal/residual
in its encoder hidden states, without any training?

**Scripts**:
- `reasoning-finetuning/phase1_internal_decomposition_probe.py` — baseline probe (job 7405)
- `reasoning-finetuning/phase1_recheck.py` — corrected probe, FIX A + FIX B (job 7408)
- `reasoning-finetuning/phase1_random_ctrl_all.py` — random backbone control on all 5 datasets (job 7410)

**Method**: Ridge probes (`alpha=1.0`, series-level 80/20 split) on frozen encoder
hidden states (`output_hidden_states=True`, 6 T5 blocks × 512-dim) against STL
ground-truth labels. Datasets: M4 Hourly, Monthly, Daily, Weekly, Electricity.

**FIX A — Trend normalization**: The initial run (job 7405) used raw STL trend as
label, giving aggregate R²≈0. Raw trend correlates with `ctx_scale = mean(|context|)`,
not with trend *shape*. Dividing labels by `ctx_scale` (same normalization as Chronos
tokenizer) reveals genuine backbone encoding: trend R² jumps from ≈0 to 0.61–0.83
per dataset.

**FIX B — STL period audit**: M4 Daily FFT detects dominant period=16 (fortnightly/
monthly business patterns), not benchmark 7 (weekly). Electricity detects period=12
(half-day), not benchmark 24 (daily). Both FFT-detected periods yield higher or
equivalent seasonal R² compared to benchmark; FFT periods are correct for this data.

**Per-dataset verdict** (gap threshold = 0.15 above own random control):

| Dataset       | Trend pre→rand (gap) | Seasonal pre→rand (gap) | Verdict   |
|---------------|----------------------|-------------------------|-----------|
| M4 Hourly     | 0.58→0.02 (+0.56) ✓  | 0.92→0.77 (+0.16) ✓     | OUTCOME A |
| M4 Monthly    | 0.83→0.45 (+0.38) ✓  | 0.69→0.33 (+0.36) ✓     | OUTCOME A |
| M4 Daily      | 0.81→0.70 (+0.11) ✗  | 0.45→0.14 (+0.31) ✓     | MIXED     |
| M4 Weekly     | 0.80→0.45 (+0.35) ✓  | 0.54→0.24 (+0.30) ✓     | OUTCOME A |
| Electricity   | 0.66→−0.09 (+0.74) ✓ | 0.82→0.54 (+0.27) ✓     | OUTCOME A |

**Overall**: MIXED — 4/5 OUTCOME A; M4 Daily trend gap (+0.11) just below threshold.
Backbone clearly encodes trend shape and dominant seasonality; M4 Daily trend is the
one borderline case (random R²=0.70 is already high, suggesting trend is trivially
recoverable from the raw tokenization of short daily series).

**Phase 2 direction**: surfacing path (Phase 2a) — lightweight readout heads on
frozen encoder layers 1–3 to surface the already-encoded decomposition structure.
Await explicit user sign-off before starting Phase 2 implementation.

---

### Stage 2 — 5-Dataset Round-Robin Joint Training (prior)

5000-step round-robin over M4H, M4M, M4D, M4W, Electricity resulted in
**NEGATIVE TRANSFER** (Electricity MASE 1.2561 vs zero-shot 0.9002, +39.5%).
M4 sub-domains collapsed into a single cluster. Stage 3 NOT cleared.

### B3a/B3b — Corrected Verdict: PARTIAL_FIX (prior)

3-axis evaluation: AXIS1 STRONG (Δ=1.6%), AXIS2 INSUFFICIENT (Elec 1.3173),
AXIS3 PARTIAL_FIX (82% gap recovery). Transfer with minimal exposure confirmed;
M4 Monthly MASE fix (nm==0.0) still required.

### Option C — delta_past Recoverability: PATHWAY_SOUND (prior)

Job 6276 (Conv1d, normalized): val R²=+0.73. Pipeline capable; delta_future has
no predictable signal from context. Option C negative result stands with positive
control confirmed.
