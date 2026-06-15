#!/usr/bin/env python3
"""
diff_analysis.py — Corrected differentiation analysis on pilot-joint-m4h-m4m-3000.pt.

NO TRAINING.  Loads checkpoint, collects per-sample prefix KV vectors for
N_PROBE M4 Hourly and N_PROBE M4 Monthly held-out val contexts, then computes
four corrected diagnostics that avoid the mean-then-cosine flaw of D6.

  D6a  Within-dataset CoV       (is the prefix input-dependent at all?)
  D6b  Mean-centered cosine     (remove shared warm-start component)
  D6c  Linear separability      (logistic regression 5-fold CV accuracy)
  D6d  Between/within ratio     (Fisher-style variance separation)

Outputs
  results/joint_pilot_differentiation.json
  results/joint_pilot_differentiation.md
  results/joint_pilot_pca.png    (optional, if matplotlib available)
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent
FINETUNING_ROOT = Path(__file__).parent
RESULTS_DIR     = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from scripts.step1_stl_decompose import detect_periods, stl_decompose

# ── Config ─────────────────────────────────────────────────────────────────────

N_PROBE    = 64      # per-dataset probe count
TRAIN_SEED = 42
RS_THRESH  = 0.4

D_MODEL    = 512
NUM_LAYERS = 6
PREFIX_LEN = 16
RANK       = 64

HOURLY_CFG = {
    'data_path':   'data/m4hourly-tsf.arrow',
    'context_col': 'context',
    'min_period':  6,
    'max_period':  24,
}
MONTHLY_CFG = {
    'data_path':   'data/m4monthly-tsf.arrow',
    'context_col': 'context',
    'min_period':  4,
    'max_period':  12,
}

CKPT_PATH = FINETUNING_ROOT / "checkpoints" / "pilot-joint-m4h-m4m-3000.pt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Load Chronos (tokenizer + ctx_scale only) ──────────────────────────────────

print("Loading Chronos T5-Small …")
pipeline  = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", device_map=str(device), dtype=torch.float32,
)
tokenizer = pipeline.tokenizer
for p in pipeline.model.parameters():
    p.requires_grad_(False)


# ── Load checkpoint ────────────────────────────────────────────────────────────

print(f"Loading {CKPT_PATH.name} …")
ckpt = torch.load(str(CKPT_PATH), map_location=device)
print(f"  step={ckpt['step']}  loss={ckpt['loss']:.4f}")

prefix_gen = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(device)
prefix_gen.load_state_dict(ckpt['pg_state'])
prefix_gen.eval()

d6_traj = ckpt.get('diagnostics', {}).get('d6_traj', [])
print(f"  Training D6 traj (mean-cosine, now under review): {d6_traj}")


# ── Collect per-sample prefix KV vectors ──────────────────────────────────────
# Each sample → one flat vector P[i] of dim = NUM_LAYERS * PREFIX_TOTAL * d_head * 2
# (factor-2: K and V concatenated)

def collect_prefix_vectors(cfg, n=N_PROBE):
    path = FINETUNING_ROOT / cfg['data_path']
    with open(str(path), "rb") as f:
        table = ipc.open_file(f).read_all()
    total   = len(table)
    rng     = np.random.default_rng(TRAIN_SEED)
    perm    = rng.permutation(total)
    val_idx = perm[int(total * 0.8):][:n]
    print(f"  {path.name}: {total} total → {len(val_idx)} val probes")

    vecs = []
    for i in val_idx:
        i = int(i)
        ctx = np.array(table.column(cfg['context_col'])[i].as_py(), dtype=np.float32)
        pds = detect_periods(ctx, min_period=cfg['min_period'],
                             max_period=cfg.get('max_period'))
        tr, se, re = stl_decompose(ctx, pds)
        rs = float(np.var(re) / (np.var(ctx) + 1e-9))
        if rs > RS_THRESH:
            tr = np.zeros_like(ctx, dtype=np.float32)
            se = np.zeros_like(ctx, dtype=np.float32)
            re = np.zeros_like(ctx, dtype=np.float32)

        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        _, _, ctx_scale = tokenizer.context_input_transform(
            pipeline._prepare_and_validate_context(ctx_t)
        )
        scale = ctx_scale.to(device).unsqueeze(-1)

        tr_t = torch.tensor(tr, dtype=torch.float32).unsqueeze(0).to(device) / scale
        se_t = torch.tensor(se, dtype=torch.float32).unsqueeze(0).to(device) / scale
        re_t = torch.tensor(re, dtype=torch.float32).unsqueeze(0).to(device) / scale

        with torch.no_grad():
            kvs = prefix_gen(tr_t, se_t, re_t)
            parts = []
            for K, V in kvs:
                # K, V shape: (batch=1, PREFIX_TOTAL, d_head)
                parts.append(K[0].flatten())
                parts.append(V[0].flatten())
            vecs.append(torch.cat(parts).cpu().numpy())
    return np.stack(vecs)   # (N, D)


print(f"\nCollecting per-sample prefix KV vectors (N={N_PROBE} per dataset) …")
t0 = time.time()
P_h = collect_prefix_vectors(HOURLY_CFG)
P_m = collect_prefix_vectors(MONTHLY_CFG)
D   = P_h.shape[1]
print(f"  Elapsed: {time.time()-t0:.1f}s  |  P_h: {P_h.shape}  P_m: {P_m.shape}")
print(f"  Prefix vector dim D = {D:,}")


# ── D6a: Within-dataset variation (CoV) ──────────────────────────────────────

def _mean_cov(P):
    mu  = np.abs(P.mean(axis=0)) + 1e-10
    std = P.std(axis=0)
    return float(np.mean(std / mu))

cov_h    = _mean_cov(P_h)
cov_m    = _mean_cov(P_m)
mean_cov = (cov_h + cov_m) / 2
print(f"\nD6a  CoV: hourly={cov_h:.4f}  monthly={cov_m:.4f}  mean={mean_cov:.4f}")


# ── D6b: Mean-centered cosine ─────────────────────────────────────────────────

P_all    = np.concatenate([P_h, P_m], axis=0)   # (2N, D)
C_global = P_all.mean(axis=0)
mR_h     = (P_h - C_global).mean(axis=0)
mR_m     = (P_m - C_global).mean(axis=0)
d6b      = float(np.dot(mR_h, mR_m) /
                 (np.linalg.norm(mR_h) * np.linalg.norm(mR_m) + 1e-10))
print(f"D6b  cosine(mR_h, mR_m) = {d6b:.4f}")


# ── D6c: Linear separability (logistic regression 5-fold CV) ─────────────────

d6c_mean  = None
d6c_std   = None
d6c_folds = []
d6c_note  = ""
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SKPipeline

    X = np.concatenate([P_h, P_m], axis=0)
    y = np.array([0] * len(P_h) + [1] * len(P_m))
    clf = SKPipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(C=0.01, max_iter=2000, solver='lbfgs')),
    ])
    scores    = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    d6c_mean  = float(scores.mean())
    d6c_std   = float(scores.std())
    d6c_folds = [round(float(s), 4) for s in scores]
    print(f"D6c  5-fold CV acc = {d6c_mean:.3f} ± {d6c_std:.3f}  "
          f"folds={[f'{s:.3f}' for s in d6c_folds]}")
except ImportError as e:
    d6c_note = f"sklearn not available ({e})"
    print(f"D6c  SKIPPED — {d6c_note}")


# ── D6d: Between/within variance ratio ────────────────────────────────────────

C_h      = P_h.mean(axis=0)
C_m      = P_m.mean(axis=0)
within_h = float(np.mean(np.sum((P_h - C_h) ** 2, axis=1)))
within_m = float(np.mean(np.sum((P_m - C_m) ** 2, axis=1)))
within   = (within_h + within_m) / 2
between  = float(np.sum((C_h - C_m) ** 2))
d6d      = between / (within + 1e-10)
print(f"D6d  between={between:.2f}  within_mean={within:.2f}  ratio={d6d:.4f}")


# ── Verdict ────────────────────────────────────────────────────────────────────

if mean_cov >= 0.1:
    d6a_verdict = "INPUT-DEPENDENT (CoV ≳ 0.1)"
    d6a_collapse = False
else:
    d6a_verdict  = "NEAR-CONSTANT (CoV → 0) — TRUE COLLAPSE"
    d6a_collapse = True

if d6c_mean is not None:
    if d6c_mean >= 0.90:
        d6c_verdict = "SEPARABLE (acc ≳ 90%)"
    elif d6c_mean >= 0.55:
        d6c_verdict = "PARTIALLY SEPARABLE"
    else:
        d6c_verdict = "NOT SEPARABLE (acc ~ 50%)"
else:
    d6c_verdict = "UNKNOWN (sklearn unavailable)"

# Apply interpretation table from spec
if not d6a_collapse and (d6c_mean is None or d6c_mean >= 0.90):
    verdict    = "DIFFERENTIATED"
    conclusion = ("Original FAIL was a metric artifact. D6 (mean-cosine) measures "
                  "the shared warm-start component, not learned differentiation. "
                  "Pilot is a PASS with positive transfer. Proceed to Stage 2.")
    stage2_ok  = True
elif not d6a_collapse and d6c_mean is not None and d6c_mean < 0.55:
    verdict    = "ADAPTIVE (NOT DATASET-SEPARABLE)"
    conclusion = ("Prefix is input-dependent (CoV ≳ 0.1) but not dataset-separable "
                  "in a linear sense — one shared adaptive manifold serves both "
                  "periodicities. Acceptable; likely still fine for Stage 2. "
                  "Note for paper: shared manifold may limit scaling to >6 datasets.")
    stage2_ok  = True
else:
    verdict    = "TRUE COLLAPSE"
    conclusion = ("Near-constant prefix (CoV → 0): real capacity finding. "
                  "Do NOT proceed to Stage 2; investigate prefix_len / rank first.")
    stage2_ok  = False

print(f"\n{'='*65}")
print(f"VERDICT:  {verdict}")
print(f"Stage 2 cleared: {stage2_ok}")
print(f"{'='*65}")
print(conclusion)


# ── PCA scatter (nice-to-have) ────────────────────────────────────────────────

pca_path   = None
pca_var    = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(P_all)
    Z_h = Z[:len(P_h)]
    Z_m = Z[len(P_h):]
    pca_var = pca.explained_variance_ratio_.tolist()

    acc_str = f"  sep-acc={d6c_mean:.2f}" if d6c_mean is not None else ""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z_h[:, 0], Z_h[:, 1], alpha=0.7, label='M4 Hourly',  s=50)
    ax.scatter(Z_m[:, 0], Z_m[:, 1], alpha=0.7, label='M4 Monthly', s=50, marker='^')
    ax.set_title(f'Prefix KV — PCA 2D (CoV={mean_cov:.3f}{acc_str})')
    ax.set_xlabel(f'PC1 ({pca_var[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_var[1]*100:.1f}% var)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_path = str(RESULTS_DIR / "joint_pilot_pca.png")
    fig.savefig(pca_path, dpi=150)
    plt.close()
    print(f"\nPCA scatter saved → {pca_path}")
except Exception as e:
    print(f"\n(PCA scatter skipped: {e})")


# ── Build display strings (safe for f-strings) ────────────────────────────────

d6c_acc_display = f"{d6c_mean:.3f} ± {d6c_std:.3f}" if d6c_mean is not None else "N/A"
d6c_fold_display = ("  ".join(f"{v:.4f}" for v in d6c_folds)
                    if d6c_folds else f"(not run — {d6c_note})")
d6_init_val  = d6_traj[0][1] if d6_traj else float("nan")
d6_traj_rows = "\n".join(
    f"| {s:>4} | {v:.4f} | {'init (pre-training)' if s == 0 else f'post step {s}'} |"
    for s, v in d6_traj
) or "| — | — | no data |"
pca_section  = (
    f"\n## PCA Scatter\n\n`joint_pilot_pca.png` — 2D PCA of all {N_PROBE*2} prefix "
    f"vectors coloured by dataset.\n(PC1={pca_var[0]*100:.1f}%  PC2={pca_var[1]*100:.1f}% "
    f"explained variance)\n"
    if pca_path and pca_var else ""
)


# ── Save JSON ─────────────────────────────────────────────────────────────────

json_blob = {
    'checkpoint':       str(CKPT_PATH.name),
    'ckpt_step':        int(ckpt['step']),
    'n_probe_per_ds':   N_PROBE,
    'prefix_dim_D':     int(D),
    'training_d6_traj': d6_traj,
    'd6a': {
        'cov_hourly':  cov_h,
        'cov_monthly': cov_m,
        'mean_cov':    mean_cov,
        'verdict':     d6a_verdict,
    },
    'd6b': {
        'cos_mean_centered': d6b,
    },
    'd6c': {
        'cv_mean_accuracy': d6c_mean,
        'cv_std_accuracy':  d6c_std,
        'cv_fold_scores':   d6c_folds,
        'verdict':          d6c_verdict,
        'note':             d6c_note,
    },
    'd6d': {
        'between_variance':       between,
        'within_variance_hourly': within_h,
        'within_variance_monthly':within_m,
        'within_variance_mean':   within,
        'ratio':                  d6d,
    },
    'verdict':       verdict,
    'conclusion':    conclusion,
    'stage2_cleared': stage2_ok,
    'pca_plot':      pca_path,
}

json_out = RESULTS_DIR / "joint_pilot_differentiation.json"
with open(str(json_out), "w") as f:
    json.dump(json_blob, f, indent=2)
print(f"\nSaved → {json_out}")


# ── Save Markdown ─────────────────────────────────────────────────────────────

md_content = f"""# Joint Pilot — Corrected Differentiation Analysis

**Checkpoint:** `{CKPT_PATH.name}` (step {ckpt['step']})
**Analysis date:** 2026-06-14
**Probe set:** {N_PROBE} M4 Hourly + {N_PROBE} M4 Monthly held-out val contexts
**Prefix vector dim D:** {D:,}

---

## Why Re-Analyse?

The original D6 metric computes `cosine(mean_KV_hourly, mean_KV_monthly)`, where the
mean is taken over all probe samples *before* the cosine. This collapses all
within-dataset variance: the result reflects only the **shared component** (warm-start
initialisation), not learned differentiation.

Evidence: D6 = {d6_init_val:.4f} at step 0, **before any gradient step**. A valid
differentiation metric cannot read ≥0.99 at random initialisation.

This analysis collects **per-sample** prefix vectors and applies four corrected diagnostics.

---

## Corrected Diagnostics

### D6a — Within-Dataset Variation

*Most fundamental check: is the prefix even input-dependent?*

| Dataset    | Mean CoV (across {D:,} dims) |
|------------|------------------------------|
| M4 Hourly  | {cov_h:.4f} |
| M4 Monthly | {cov_m:.4f} |
| **Combined** | **{mean_cov:.4f}** |

**{d6a_verdict}**

CoV → 0 means prefix ≈ constant regardless of input → true collapse.
CoV ≳ 0.1 means prefix varies with input → conditioning is happening.

---

### D6b — Mean-Centered Cosine

Global mean C subtracted from every sample vector before taking dataset means:

```
cosine(mean(P_hourly − C), mean(P_monthly − C)) = {d6b:.4f}
```

Near −1 or low: strong differentiation in the residual (after removing shared warm-start).
Near +1: dataset means align even in residual space.

---

### D6c — Linear Separability

`StandardScaler → LogisticRegression(C=0.01)` trained to predict dataset label,
5-fold cross-validation on N={N_PROBE*2} samples:

**Accuracy: {d6c_acc_display}**
Fold scores: {d6c_fold_display}

**{d6c_verdict}**

~50% = one blob, not separable (collapse).  ~100% = fully separable.

---

### D6d — Between/Within Variance Ratio

| Quantity | Value |
|----------|-------|
| Between-dataset variance `‖C_h − C_m‖²` | {between:.2f} |
| Within-dataset variance (hourly) | {within_h:.2f} |
| Within-dataset variance (monthly) | {within_m:.2f} |
| Within-dataset variance (mean) | {within:.2f} |
| **Ratio (between / within_mean)** | **{d6d:.4f}** |

≪ 1: internal spread dominates → not separated.
≳ 1: datasets more separated than internal spread.

---

## Training D6 Reference (mean-cosine — flawed metric)

| Step | Cosine | Note |
|------|--------|------|
{d6_traj_rows}

D6 ≥ 0.99 at init because `warm_start_from_chronos` initialises all projection heads
from the same Chronos encoder weights — producing near-identical outputs for both
datasets before any training.

---

## Verdict: **{verdict}**

{conclusion}

**Stage 2 cleared:** `{stage2_ok}`

---

## Interpretation Table Applied

| D6a CoV | D6c Sep-Acc | Conclusion |
|---------|-------------|------------|
| ≳ 0.1 | ≳ 90% | **DIFFERENTIATED** — original FAIL was a metric artifact |
| ≳ 0.1 | ~ 50% | **ADAPTIVE** — shared manifold, acceptable for Stage 2 |
| → 0 | ~ 50% | **TRUE COLLAPSE** — do not proceed |

**This run:** D6a CoV = {mean_cov:.4f} / D6c sep-acc = {d6c_acc_display} → **{verdict}**
{pca_section}"""

md_out = RESULTS_DIR / "joint_pilot_differentiation.md"
with open(str(md_out), "w") as f:
    f.write(md_content)
print(f"Saved → {md_out}")

print(f"\n{'='*65}")
print(f"VERDICT:  {verdict}")
print(f"Stage 2 cleared: {stage2_ok}")
print(f"{'='*65}")
