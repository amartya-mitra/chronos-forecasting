#!/usr/bin/env python3
"""
diff_stage2.py — Part C differentiation analysis for Stage 2 checkpoint.

Probe set: 32 held-out contexts per dataset × 5 datasets = 160 total vectors.
Flatten per-sample prefix KV tensors across all 6 layers → D-dim vector P[i].

C1 — D6a: Within-dataset variation (CoV) per dataset
C2 — D6c: 5-class linear separability (LogisticRegression 5-fold CV, chance=20%)
C3 — D6d: Multiclass Fisher ratio = trace(S_B) / trace(S_W)
C4 — Pairwise centroid similarity matrix (5×5)

Outputs:
  results/joint_stage2_differentiation.json
  results/joint_stage2_pca.png    (optional)
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

N_PROBE    = 32     # per-dataset probe count
TRAIN_SEED = 42
RS_THRESH  = 0.4

D_MODEL    = 512
NUM_LAYERS = 6
PREFIX_LEN = 16
RANK       = 64

DATASET_ORDER  = ['sarsim0', 'm4hourly', 'm4monthly', 'm4daily', 'm4weekly']
DATASET_LABELS = {
    'sarsim0':   'SarSim0',
    'm4hourly':  'M4 Hourly',
    'm4monthly': 'M4 Monthly',
    'm4daily':   'M4 Daily',
    'm4weekly':  'M4 Weekly',
}

DATASET_CONFIG = {
    'sarsim0': {
        'data_path':   'data/sarsim0-10k.arrow',
        'context_col': 'context',
        'min_period':  6,
        'max_period':  None,
    },
    'm4hourly': {
        'data_path':   'data/m4hourly-tsf.arrow',
        'context_col': 'context',
        'min_period':  6,
        'max_period':  24,
    },
    'm4monthly': {
        'data_path':   'data/m4monthly-tsf.arrow',
        'context_col': 'context',
        'min_period':  4,
        'max_period':  12,
    },
    'm4daily': {
        'data_path':   'data/m4daily-tsf.arrow',
        'context_col': 'context',
        'min_period':  4,
        'max_period':  31,
    },
    'm4weekly': {
        'data_path':   'data/m4weekly-tsf.arrow',
        'context_col': 'context',
        'min_period':  4,
        'max_period':  80,
    },
}

CKPT_PATH = FINETUNING_ROOT / "checkpoints" / "joint-stage2-5ds-5000.pt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Load Chronos ───────────────────────────────────────────────────────────────

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
print(f"  step={ckpt['step']}")

prefix_gen = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(device)
prefix_gen.load_state_dict(ckpt['pg_state'])
prefix_gen.eval()


# ── Collect per-sample prefix KV vectors ──────────────────────────────────────

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
                parts.append(K[0].flatten())
                parts.append(V[0].flatten())
            vecs.append(torch.cat(parts).cpu().numpy())
    return np.stack(vecs)   # (N, D)


print(f"\nCollecting per-sample prefix KV vectors (N={N_PROBE} per dataset) …")
t0 = time.time()
P = {}
for ds in DATASET_ORDER:
    lbl   = DATASET_LABELS[ds]
    print(f"  [{lbl}]")
    P[ds] = collect_prefix_vectors(DATASET_CONFIG[ds])

D    = P[DATASET_ORDER[0]].shape[1]
N_K  = len(DATASET_ORDER)
N_DS = N_PROBE
print(f"  Elapsed: {time.time()-t0:.1f}s  |  dim D = {D:,}  |  total vectors = {N_K*N_DS}")


# ── C1: D6a — Within-dataset CoV (per dataset) ────────────────────────────────

def _mean_cov(M):
    mu  = np.abs(M.mean(axis=0)) + 1e-10
    std = M.std(axis=0)
    return float(np.mean(std / mu))

print(f"\nC1 (D6a) — Within-dataset variation (CoV):")
cov = {}
for ds in DATASET_ORDER:
    lbl   = DATASET_LABELS[ds]
    cov_v = _mean_cov(P[ds])
    cov[ds] = cov_v
    flag = "" if cov_v >= 0.1 else "  ⚠ COLLAPSED"
    print(f"  {lbl:<12}  CoV = {cov_v:.4f}{flag}")

collapsed_datasets = [ds for ds in DATASET_ORDER if cov[ds] < 0.1]
if collapsed_datasets:
    lbls = ", ".join(DATASET_LABELS[ds] for ds in collapsed_datasets)
    print(f"  ⚠ Collapsed datasets (CoV < 0.1): {lbls}")
else:
    print(f"  All datasets input-dependent (CoV ≥ 0.1)  ✓")


# ── C2: D6c — 5-class linear separability ─────────────────────────────────────

P_all = np.concatenate([P[ds] for ds in DATASET_ORDER], axis=0)   # (5*N_PROBE, D)
y_all = np.array([k for k, ds in enumerate(DATASET_ORDER) for _ in range(N_PROBE)])

c2_mean  = None
c2_std   = None
c2_folds = []
c2_note  = ""
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SKPipeline

    clf = SKPipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(C=0.01, max_iter=2000, solver='lbfgs')),
    ])
    scores   = cross_val_score(clf, P_all, y_all, cv=5, scoring='accuracy')
    c2_mean  = float(scores.mean())
    c2_std   = float(scores.std())
    c2_folds = [round(float(s), 4) for s in scores]
    print(f"\nC2 (D6c) — 5-class linear separability (chance = 20%):")
    print(f"  5-fold CV acc = {c2_mean:.3f} ± {c2_std:.3f}  "
          f"folds={[f'{s:.3f}' for s in c2_folds]}")
except ImportError as e:
    c2_note = f"sklearn not available ({e})"
    print(f"\nC2 (D6c) — SKIPPED — {c2_note}")


# ── C3: D6d — Multiclass Fisher ratio ─────────────────────────────────────────

C_global = P_all.mean(axis=0)
centroids = {ds: P[ds].mean(axis=0) for ds in DATASET_ORDER}

# trace(S_B) = Σ_k n_k * ||C_k - C_global||^2
trace_SB = sum(N_PROBE * float(np.sum((centroids[ds] - C_global) ** 2))
               for ds in DATASET_ORDER)

# trace(S_W) = Σ_k Σ_i∈k ||P_i - C_k||^2
trace_SW = sum(float(np.sum((P[ds] - centroids[ds]) ** 2))
               for ds in DATASET_ORDER)

fisher_ratio = trace_SB / (trace_SW + 1e-10)

print(f"\nC3 (D6d) — Multiclass Fisher ratio:")
print(f"  trace(S_B) = {trace_SB:.2f}")
print(f"  trace(S_W) = {trace_SW:.2f}")
print(f"  trace(S_B) / trace(S_W) = {fisher_ratio:.4f}")
flag = "✓ separated" if fisher_ratio > 1.0 else "⚠ not separated"
print(f"  Target > 1.0:  {flag}")


# ── C4: Pairwise centroid similarity matrix ────────────────────────────────────

residuals = {ds: centroids[ds] - C_global for ds in DATASET_ORDER}

n_ds = len(DATASET_ORDER)
sim_matrix = np.zeros((n_ds, n_ds))
for i, ds_i in enumerate(DATASET_ORDER):
    for j, ds_j in enumerate(DATASET_ORDER):
        ri = residuals[ds_i]
        rj = residuals[ds_j]
        sim_matrix[i, j] = float(
            np.dot(ri, rj) / (np.linalg.norm(ri) * np.linalg.norm(rj) + 1e-10)
        )

print(f"\nC4 — Pairwise centroid similarity matrix (mean-centered):")
short = ['SarSim', 'M4H', 'M4M', 'M4D', 'M4W']
hdr   = "             " + "  ".join(f"{s:>7}" for s in short)
print(hdr)
for i, ds_i in enumerate(DATASET_ORDER):
    lbl = DATASET_LABELS[ds_i][:12]
    row = "  ".join(f"{sim_matrix[i, j]:>+7.3f}" for j in range(n_ds))
    print(f"{lbl:<12}  {row}")

# Flag notable values
print()
for i, ds_i in enumerate(DATASET_ORDER):
    for j, ds_j in enumerate(DATASET_ORDER):
        if i >= j:
            continue
        s = sim_matrix[i, j]
        li = DATASET_LABELS[ds_i]
        lj = DATASET_LABELS[ds_j]
        if s > 0.5:
            print(f"  ⚠ HIGH similarity: {li} vs {lj} = {s:+.3f}")
        elif s < -0.5:
            print(f"  ✓ ANTIPODAL: {li} vs {lj} = {s:+.3f}")


# ── Save JSON ─────────────────────────────────────────────────────────────────

c2_acc_display  = f"{c2_mean:.3f} ± {c2_std:.3f}" if c2_mean is not None else "N/A"
c2_fold_display = ("  ".join(f"{v:.4f}" for v in c2_folds)
                   if c2_folds else f"(not run — {c2_note})")

json_blob = {
    'checkpoint':       str(CKPT_PATH.name),
    'ckpt_step':        int(ckpt['step']),
    'n_probe_per_ds':   N_PROBE,
    'prefix_dim_D':     int(D),
    'c1_d6a_cov': {ds: cov[ds] for ds in DATASET_ORDER},
    'c1_collapsed_datasets': collapsed_datasets,
    'c2_d6c': {
        'cv_mean_accuracy': c2_mean,
        'cv_std_accuracy':  c2_std,
        'cv_fold_scores':   c2_folds,
        'chance_level':     0.2,
        'target':           0.7,
        'note':             c2_note,
    },
    'c3_fisher': {
        'trace_SB':    trace_SB,
        'trace_SW':    trace_SW,
        'ratio':       fisher_ratio,
    },
    'c4_pairwise_sim': {
        'datasets': DATASET_ORDER,
        'labels':   [DATASET_LABELS[ds] for ds in DATASET_ORDER],
        'matrix':   sim_matrix.tolist(),
    },
}

json_out = RESULTS_DIR / "joint_stage2_differentiation.json"
with open(str(json_out), "w") as f:
    json.dump(json_blob, f, indent=2)
print(f"\nSaved → {json_out}")


# ── Optional PCA scatter (5 datasets) ─────────────────────────────────────────

pca_path = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca     = PCA(n_components=2, random_state=42)
    Z_all   = pca.fit_transform(P_all)
    pca_var = pca.explained_variance_ratio_.tolist()

    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', '^', 's', 'D', 'v']

    fig, ax = plt.subplots(figsize=(8, 7))
    offset_idx = 0
    for k, ds in enumerate(DATASET_ORDER):
        lbl = DATASET_LABELS[ds]
        Z_k = Z_all[offset_idx:offset_idx + N_PROBE]
        ax.scatter(Z_k[:, 0], Z_k[:, 1], alpha=0.7, label=lbl,
                   s=50, color=colors[k], marker=markers[k])
        offset_idx += N_PROBE

    acc_str = f"  sep-acc={c2_mean:.2f}" if c2_mean is not None else ""
    ax.set_title(f'Stage 2 Prefix KV — PCA 2D{acc_str}')
    ax.set_xlabel(f'PC1 ({pca_var[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_var[1]*100:.1f}% var)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_path = str(RESULTS_DIR / "joint_stage2_pca.png")
    fig.savefig(pca_path, dpi=150)
    plt.close()
    print(f"PCA scatter saved → {pca_path}")
    json_blob['pca_plot'] = pca_path
    with open(str(json_out), "w") as f:
        json.dump(json_blob, f, indent=2)
except Exception as e:
    print(f"(PCA skipped: {e})")


# ── Print summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("STAGE 2 DIFFERENTIATION SUMMARY")
print(f"{'='*65}")
print(f"C1 (D6a) CoV per dataset:")
for ds in DATASET_ORDER:
    ok = "✓" if cov[ds] >= 0.1 else "⚠"
    print(f"  {DATASET_LABELS[ds]:<12}  {cov[ds]:.4f}  {ok}")
print(f"C2 (D6c) 5-class sep-acc:  {c2_acc_display}  (chance=20%)")
print(f"C3 Fisher ratio:           {fisher_ratio:.4f}  {'(> 1.0 ✓)' if fisher_ratio > 1.0 else '(≤ 1.0 ⚠)'}")
print(f"C4 pairwise matrix:  see JSON / PCA plot")
print(f"{'='*65}")
