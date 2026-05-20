#!/usr/bin/env python3
"""
prefix_train_scaffold.py — Step 4: Locked configuration smoke test.

Locked configuration (Option A):
  - warm_start_from_chronos(scale_warm=False): full-magnitude Wk/Wv, Kaiming residual
  - Optimizer: single AdamW, lr=1e-4, weight_decay=0.01
  - Lambda_ortho: 0.01

Runs 3 batches and checks 5 diagnostics:
  D1. Bottleneck grad norm ratio max/min < 20× (expect ~1.7×)
  D2. Prefix K / Chronos K magnitude ratio in 0.1–2.0 (expect ~0.41)
  D3. Forecast loss in 3–8, stable (no jump > 1.0)
  D4. Max pairwise |cos sim| < 0.3 (expect ~0.023)
  D5. 5/6 encoder layers ≥ 5% total prefix attention

Prints "Configuration locked." if all 5 pass.

TODO: evaluation loop on held-out val set
TODO: checkpoint saving (best WTS + config)
TODO: cosine LR scheduler with warmup
"""

import sys
from pathlib import Path

REPO_ROOT       = Path(__file__).parent.parent.parent
FINETUNING_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNING_ROOT))

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F

from chronos import ChronosPipeline
from prefix_generator import PrefixGenerator
from prefix_injection import inject_prefix, remove_prefix_hooks
from scripts.step1_stl_decompose import detect_periods, stl_decompose


# ── Config ─────────────────────────────────────────────────────────────────────

ARROW_PATH   = FINETUNING_ROOT / "data" / "sarsim0-10k.arrow"
CONTEXT_LEN  = 512
PRED_LEN     = 64
BATCH_SIZE   = 4
TRAIN_SEED   = 42
NUM_BATCHES  = 3

D_MODEL      = 512
NUM_LAYERS   = 6
PREFIX_LEN   = 16
PREFIX_TOTAL = 3 * PREFIX_LEN   # 48
RANK         = 64
LAMBDA_ORTHO = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diagnostic thresholds
D1_MAX_RATIO  = 20.0
D2_MIN_RATIO  = 0.1
D2_MAX_RATIO  = 2.0
D3_FC_MIN     = 3.0
D3_FC_MAX     = 8.0
D3_MAX_JUMP   = 2.0
D4_MAX_COS    = 0.3
D5_MIN_LAYERS = 5          # out of 6
ATTN_THRESH   = 0.05


# ── Load dataset ───────────────────────────────────────────────────────────────

print("Loading sarsim0-10k.arrow …")
with open(str(ARROW_PATH), "rb") as f:
    table = ipc.open_file(f).read_all()

n_total   = len(table)
rng       = np.random.default_rng(TRAIN_SEED)
indices   = rng.permutation(n_total)
train_idx = indices[: int(n_total * 0.8)]
print(f"  {n_total} rows  →  {len(train_idx)} train")


# ── Load Chronos ───────────────────────────────────────────────────────────────

print("Loading Chronos T5-Small …")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=str(DEVICE),
    dtype=torch.float32,
)
chronos_model = pipeline.model
tokenizer     = pipeline.tokenizer
for p in chronos_model.parameters():
    p.requires_grad_(False)
print(f"  {sum(p.numel() for p in chronos_model.parameters()):,} params frozen\n")


# ── Batch builder ──────────────────────────────────────────────────────────────

def build_batch(row_indices: list[int]) -> dict:
    contexts, futures, trends, seasonals, residuals = [], [], [], [], []
    for idx in row_indices:
        ctx = np.array(table.column("context")[idx].as_py(), dtype=np.float32)
        ftr = np.array(table.column("forecast")[idx].as_py(), dtype=np.float32)
        periods = detect_periods(ctx)
        tr, se, re = stl_decompose(ctx, periods)
        contexts.append(ctx); futures.append(ftr)
        trends.append(tr);    seasonals.append(se);  residuals.append(re)

    ctx_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32).to(DEVICE)
    input_ids, attn_mask, ctx_scale = tokenizer.context_input_transform(
        pipeline._prepare_and_validate_context(ctx_tensor)
    )
    future_tensor = torch.tensor(np.stack(futures), dtype=torch.float32).to(DEVICE)
    label_ids, label_mask, _ = tokenizer._input_transform(
        context=future_tensor, scale=ctx_scale
    )
    label_ids[~label_mask.bool()] = -100

    return dict(
        input_ids = input_ids.to(DEVICE),
        attn_mask = attn_mask.to(DEVICE),
        label_ids = label_ids.to(DEVICE),
        trend     = torch.tensor(np.stack(trends),    dtype=torch.float32).to(DEVICE),
        seasonal  = torch.tensor(np.stack(seasonals), dtype=torch.float32).to(DEVICE),
        residual  = torch.tensor(np.stack(residuals), dtype=torch.float32).to(DEVICE),
    )


# ── Ortho penalty ──────────────────────────────────────────────────────────────

def ortho_penalty(prefix_kvs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    total = torch.zeros(1, device=prefix_kvs[0][0].device)
    for K, _V in prefix_kvs:
        m  = K.shape[1] // 3
        Kt = K[:, 0:m,   :].flatten(1)
        Ks = K[:, m:2*m, :].flatten(1)
        Kr = K[:, 2*m:,  :].flatten(1)
        total = (
            total
            + F.cosine_similarity(Kt, Ks, dim=1).mean() ** 2
            + F.cosine_similarity(Kt, Kr, dim=1).mean() ** 2
            + F.cosine_similarity(Ks, Kr, dim=1).mean() ** 2
        )
    return total / len(prefix_kvs)


# ── Smoke test ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("LOCKED CONFIGURATION SMOKE TEST  (Option A)")
print("  scale_warm=False | AdamW lr=1e-4 | weight_decay=0.01")
print("=" * 60)

torch.manual_seed(0)
pg = PrefixGenerator(
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    prefix_len_per_component=PREFIX_LEN, rank=RANK,
).to(DEVICE)
pg.train()
pg.warm_start_from_chronos(chronos_model, scale_warm=False)

optimizer = torch.optim.AdamW(pg.parameters(), lr=1e-4, weight_decay=0.01)

losses       = []
d1_gnorms    = {}
d2_pfx_k     = None
d2_chr_k     = None
d4_cos       = {}
d5_attn      = []

chr_k_storage: list = []

def _ck_hook(module, inp, out):
    chr_k_storage.append(out.detach().cpu())

for batch_idx in range(NUM_BATCHES):
    row_idx = train_idx[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE].tolist()
    batch   = build_batch(row_idx)

    ck_handle = None
    if batch_idx == 0:
        ck_handle = (
            chronos_model.model.encoder.block[0]
            .layer[0].SelfAttention.k
            .register_forward_hook(_ck_hook)
        )

    optimizer.zero_grad()
    try:
        prefix_kvs = pg(batch["trend"], batch["seasonal"], batch["residual"])
        inject_prefix(chronos_model, prefix_generator=pg, prefix_kvs=prefix_kvs)

        if batch_idx == 0:
            with torch.no_grad():
                enc_attn = chronos_model.model.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attn_mask"],
                    output_attentions=True,
                )
            for w in enc_attn.attentions:
                pfx_w = w[:, :, :, :PREFIX_TOTAL].mean().item()
                d5_attn.append(pfx_w * PREFIX_TOTAL)

        output = chronos_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attn_mask"],
            labels=batch["label_ids"],
        )
        fc_loss    = output.loss
        o_loss     = ortho_penalty(prefix_kvs)
        total_loss = fc_loss + LAMBDA_ORTHO * o_loss
        total_loss.backward()
        optimizer.step()

    finally:
        remove_prefix_hooks(chronos_model)
        if ck_handle is not None:
            ck_handle.remove()

    losses.append((fc_loss.item(), o_loss.item(), total_loss.item()))
    print(f"  Batch {batch_idx}: fc={fc_loss.item():.4f}  "
          f"ortho={o_loss.item():.4f}  total={total_loss.item():.4f}")

    if batch_idx == 0:
        for head_name, mod_list in [
            ("trend",    pg.proj_trend),
            ("seasonal", pg.proj_seasonal),
            ("noise",    pg.proj_noise),
        ]:
            p = mod_list[0][0].weight
            d1_gnorms[head_name] = (
                p.grad.norm().item() if p.grad is not None else float("nan")
            )

        P_K = prefix_kvs[0][0].detach().cpu()
        d2_pfx_k = P_K.abs().mean().item()
        d2_chr_k = chr_k_storage[0].abs().mean().item() if chr_k_storage else float("nan")

        m  = PREFIX_LEN
        Kt = P_K[:, 0:m,   :].flatten(1)
        Ks = P_K[:, m:2*m, :].flatten(1)
        Kr = P_K[:, 2*m:,  :].flatten(1)
        d4_cos["ts"] = F.cosine_similarity(Kt, Ks, dim=1).mean().item()
        d4_cos["tr"] = F.cosine_similarity(Kt, Kr, dim=1).mean().item()
        d4_cos["sr"] = F.cosine_similarity(Ks, Kr, dim=1).mean().item()


# ── Diagnostics ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("DIAGNOSTICS")
print(f"{'='*60}")

# D1
gnorm_vals = [v for v in d1_gnorms.values() if not np.isnan(v) and v > 0]
d1_ratio   = max(gnorm_vals) / min(gnorm_vals) if gnorm_vals else float("nan")
d1_pass    = d1_ratio < D1_MAX_RATIO
print(f"\nD1. Bottleneck grad norms (inner linear, layer 0):")
for h in ("trend", "seasonal", "noise"):
    print(f"    {h:<12} {d1_gnorms.get(h, float('nan')):.6f}")
print(f"    ratio max/min  {d1_ratio:.1f}×  {'PASS' if d1_pass else 'FAIL ⚠  (target < 20×)'}")

# D2
d2_ratio = (d2_pfx_k / (d2_chr_k + 1e-9)) if (d2_pfx_k is not None and d2_chr_k is not None) else float("nan")
d2_pass  = D2_MIN_RATIO <= d2_ratio <= D2_MAX_RATIO
print(f"\nD2. KV magnitude (layer 0):")
print(f"    Prefix K mean|·|  {d2_pfx_k:.4f}")
print(f"    Chronos K mean|·| {d2_chr_k:.4f}")
print(f"    ratio pfx/chr     {d2_ratio:.3f}  {'PASS' if d2_pass else 'FAIL ⚠  (target 0.1–2.0)'}")

# D3
fc_losses = [fc for fc, _, _ in losses]
d3_range  = all(D3_FC_MIN <= fc <= D3_FC_MAX for fc in fc_losses)
d3_stable = max(abs(fc_losses[i+1] - fc_losses[i]) for i in range(len(fc_losses)-1)) < D3_MAX_JUMP
d3_pass   = d3_range and d3_stable
print(f"\nD3. Loss stability:")
for i, (fc, o, tot) in enumerate(losses):
    flag = "" if D3_FC_MIN <= fc <= D3_FC_MAX else "  ⚠"
    print(f"    Batch {i}: fc={fc:.4f}  ortho={o:.4f}  total={tot:.4f}{flag}")
max_jump = max(abs(fc_losses[i+1] - fc_losses[i]) for i in range(len(fc_losses)-1))
print(f"    range OK={d3_range}  max step jump={max_jump:.4f}  {'PASS' if d3_pass else 'FAIL ⚠'}")

# D4
max_cos = max(abs(v) for v in d4_cos.values())
d4_pass = max_cos < D4_MAX_COS
print(f"\nD4. Pairwise cosine similarities (layer 0 prefix K):")
for key, label in [("ts", "trend–seasonal"), ("tr", "trend–noise"), ("sr", "seasonal–noise")]:
    print(f"    {label:<20} {d4_cos.get(key, float('nan')):+.4f}")
print(f"    max |cos sim|  {max_cos:.4f}  {'PASS' if d4_pass else 'FAIL ⚠  (target < 0.3)'}")

# D5
layers_above = sum(1 for v in d5_attn if v >= ATTN_THRESH)
d5_pass      = layers_above >= D5_MIN_LAYERS
print(f"\nD5. Per-layer total prefix attention (batch 0):")
for i, v in enumerate(d5_attn):
    status = "OK" if v >= ATTN_THRESH else "⚠ LOW"
    print(f"    Layer {i}: {v:.4f}  {status}")
print(f"    {layers_above}/6 layers ≥ 5%  {'PASS' if d5_pass else 'FAIL ⚠  (target ≥ 5/6)'}")


# ── Final verdict ──────────────────────────────────────────────────────────────

all_pass = d1_pass and d2_pass and d3_pass and d4_pass and d5_pass
print(f"\n{'='*60}")
if all_pass:
    print("Configuration locked.")
else:
    failures = []
    if not d1_pass: failures.append(f"D1 ratio={d1_ratio:.1f}× (target <20×)")
    if not d2_pass: failures.append(f"D2 ratio={d2_ratio:.3f} (target 0.1–2.0)")
    if not d3_pass: failures.append(f"D3 range={d3_range} stable={d3_stable}")
    if not d4_pass: failures.append(f"D4 max_cos={max_cos:.4f} (target <0.3)")
    if not d5_pass: failures.append(f"D5 layers={layers_above}/6 (target ≥5)")
    print(f"FAILED diagnostics: {'; '.join(failures)}")
print(f"{'='*60}")
