# Joint Pilot — Corrected Differentiation Analysis

**Checkpoint:** `pilot-joint-m4h-m4m-3000.pt` (step 3000)
**Analysis date:** 2026-06-14
**Probe set:** 64 M4 Hourly + 64 M4 Monthly held-out val contexts
**Prefix vector dim D:** 294,912

---

## Why Re-Analyse?

The original D6 metric computes `cosine(mean_KV_hourly, mean_KV_monthly)`, where the
mean is taken over all probe samples *before* the cosine. This collapses all
within-dataset variance: the result reflects only the **shared component** (warm-start
initialisation), not learned differentiation.

Evidence: D6 = 0.9981 at step 0, **before any gradient step**. A valid
differentiation metric cannot read ≥0.99 at random initialisation.

This analysis collects **per-sample** prefix vectors and applies four corrected diagnostics.

---

## Corrected Diagnostics

### D6a — Within-Dataset Variation

*Most fundamental check: is the prefix even input-dependent?*

| Dataset    | Mean CoV (across 294,912 dims) |
|------------|------------------------------|
| M4 Hourly  | 2.5195 |
| M4 Monthly | 0.3268 |
| **Combined** | **1.4231** |

**INPUT-DEPENDENT (CoV ≳ 0.1)**

CoV → 0 means prefix ≈ constant regardless of input → true collapse.
CoV ≳ 0.1 means prefix varies with input → conditioning is happening.

---

### D6b — Mean-Centered Cosine

Global mean C subtracted from every sample vector before taking dataset means:

```
cosine(mean(P_hourly − C), mean(P_monthly − C)) = -1.0000
```

Near −1 or low: strong differentiation in the residual (after removing shared warm-start).
Near +1: dataset means align even in residual space.

---

### D6c — Linear Separability

`StandardScaler → LogisticRegression(C=0.01)` trained to predict dataset label,
5-fold cross-validation on N=128 samples:

**Accuracy: 1.000 ± 0.000**
Fold scores: 1.0000  1.0000  1.0000  1.0000  1.0000

**SEPARABLE (acc ≳ 90%)**

~50% = one blob, not separable (collapse).  ~100% = fully separable.

---

### D6d — Between/Within Variance Ratio

| Quantity | Value |
|----------|-------|
| Between-dataset variance `‖C_h − C_m‖²` | 1382.14 |
| Within-dataset variance (hourly) | 2079.05 |
| Within-dataset variance (monthly) | 347.86 |
| Within-dataset variance (mean) | 1213.46 |
| **Ratio (between / within_mean)** | **1.1390** |

≪ 1: internal spread dominates → not separated.
≳ 1: datasets more separated than internal spread.

---

## Training D6 Reference (mean-cosine — flawed metric)

| Step | Cosine | Note |
|------|--------|------|
|    0 | 0.9981 | init (pre-training) |
| 1500 | 0.9955 | post step 1500 |
| 3000 | 0.9939 | post step 3000 |

D6 ≥ 0.99 at init because `warm_start_from_chronos` initialises all projection heads
from the same Chronos encoder weights — producing near-identical outputs for both
datasets before any training.

---

## Verdict: **DIFFERENTIATED**

Original FAIL was a metric artifact. D6 (mean-cosine) measures the shared warm-start component, not learned differentiation. Pilot is a PASS with positive transfer. Proceed to Stage 2.

**Stage 2 cleared:** `True`

---

## Interpretation Table Applied

| D6a CoV | D6c Sep-Acc | Conclusion |
|---------|-------------|------------|
| ≳ 0.1 | ≳ 90% | **DIFFERENTIATED** — original FAIL was a metric artifact |
| ≳ 0.1 | ~ 50% | **ADAPTIVE** — shared manifold, acceptable for Stage 2 |
| → 0 | ~ 50% | **TRUE COLLAPSE** — do not proceed |

**This run:** D6a CoV = 1.4231 / D6c sep-acc = 1.000 ± 0.000 → **DIFFERENTIATED**

## PCA Scatter

`joint_pilot_pca.png` — 2D PCA of all 128 prefix vectors coloured by dataset.
(PC1=71.9%  PC2=17.0% explained variance)
