# Joint Training Pilot — M4 Hourly + M4 Monthly

**Checkpoint:** `pilot-joint-m4h-m4m-3000.pt` (step 3000)
**Date:** 2026-06-12

## Verdict: FAIL

D6=0.9939 > 0.97 — prefix collapsed to a single representation

## Comparison Table

| Dataset     | MASE_base | MASE_solo | MASE_joint | Δ(solo→joint) | Retain% |
|-------------|-----------|-----------|------------|---------------|---------|
| M4 Hourly   | 1.6565    | 1.2181    | 1.1585     | -0.0596       | 113.6%    |
| M4 Monthly  | 1.2105    | 0.7274    | 0.7453     | +0.0179       | 96.3%    |

## D6 Trajectory (prefix differentiation)

| Step | Cosine Similarity | Assessment |
|------|-------------------|------------|
|    0 | 0.9981 | ⚠⚠ COLLAPSED |
| 1500 | 0.9955 | ⚠⚠ COLLAPSED |
| 3000 | 0.9939 | ⚠⚠ COLLAPSED |

## EMA fc_loss Trajectories

**M4 Hourly:**  [(0, 2.6246), (500, 2.4702), (1000, 2.2767), (1499, 2.2031), (1999, 2.1816), (2499, 2.1175), (2999, 2.053)]

**M4 Monthly:** [(0, None), (500, 3.0624), (1000, 3.1561), (1499, 2.9797), (1999, 3.0967), (2499, 2.9491), (2999, 3.0712)]

## PASS Criteria Applied

- MASE_joint ≤ MASE_solo + 0.03 on **both** datasets: `✓`
- D6_final < 0.90 (prefix differentiation): `✗`
- No regression beyond zero-shot baseline: `✓`

## Next Steps

**Do not proceed to Stage 2.** Review results above and address failure mode before scaling to full joint run.
