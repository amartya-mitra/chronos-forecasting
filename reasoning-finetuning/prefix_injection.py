#!/usr/bin/env python3
"""
prefix_injection.py — Step 3: inject prefix KVs into frozen Chronos T5 encoder.

Strategy: monkey-patch T5Attention.forward() on each of the 6 encoder
self-attention layers to prepend PrefixGenerator KVs AFTER T5's own
Q/K/V projections and BEFORE the dot-product attention computation.

Key shape conventions (T5-Small):
  n_heads = 8,  d_kv = 64,  d_model = 512
  After T5 projection: K,V are (batch, n_heads, seq_len, d_kv)
  PrefixGenerator K,V:  (batch, prefix_total, d_model)
    → reshape to (batch, n_heads, prefix_total, d_kv) before cat.

position_bias flow:
  Layer 0  (has_relative_attention_bias=True): position_bias is None on entry.
           Computed here for extended key_length (prefix + original).
           Returned with shape (batch, n_heads, q_len, prefix+k_len).
  Layers 1–5: receive the extended position_bias from layer 0 as-is —
              no further extension needed (shapes already match extended K).

Known simplification: relative position biases for prefix token positions
are taken from T5's learned table using the tokens' natural relative offsets
to the query. These offsets are arbitrary for prefix tokens (they land in
"far in the past" buckets), but this is acceptable for an initial prototype.
See prefix_tuning.md § Step 5 for planned warm-start mitigation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── internal: per-layer patched forward factory ───────────────────────────────

def _make_patched_forward(attn, P_K: torch.Tensor, P_V: torch.Tensor):
    """
    Return a replacement forward() for one T5Attention module.

    P_K, P_V: (orig_batch, prefix_total, d_model) — detached tensors from
              PrefixGenerator output for this layer.

    The replacement is stored as a plain instance attribute on `attn`,
    so nn.Module.__call__ routes through it without any special descriptor
    magic; `attn` is captured in the closure for attribute access.
    """
    prefix_total = P_K.shape[1]   # 3 * prefix_len_per_component

    def patched_forward(
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        from transformers.cache_utils import EncoderDecoderCache

        batch_size, seq_length = hidden_states.shape[:2]
        n_heads = attn.n_heads
        d_kv    = attn.key_value_proj_dim
        is_cross_attention = key_value_states is not None

        # ── Q projection ───────────────────────────────────────────────────
        query_states = attn.q(hidden_states)
        query_states = query_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)

        # ── K, V projections (replicates original T5Attention logic) ───────
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(attn.layer_idx)
            curr_pkv = (
                past_key_values.cross_attention_cache
                if is_cross_attention
                else past_key_values.self_attention_cache
            )
        else:
            curr_pkv = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # Reuse cached cross-attention KVs — not applicable to encoder self-attn
            key_states   = curr_pkv.layers[attn.layer_idx].keys
            value_states = curr_pkv.layers[attn.layer_idx].values
        else:
            key_states   = attn.k(current_states)
            value_states = attn.v(current_states)
            key_states   = key_states.view(  batch_size, -1, n_heads, d_kv).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, n_heads, d_kv).transpose(1, 2)
            if past_key_values is not None:
                cache_pos = cache_position if not is_cross_attention else None
                key_states, value_states = curr_pkv.update(
                    key_states, value_states, attn.layer_idx, {"cache_position": cache_pos}
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[attn.layer_idx] = True

        # ── Expand prefix to match effective batch (pipeline repeats for num_samples) ──
        orig_batch = P_K.shape[0]
        pk = P_K.to(hidden_states.device)
        pv = P_V.to(hidden_states.device)
        if batch_size != orig_batch:
            if batch_size % orig_batch != 0:
                raise RuntimeError(
                    f"Prefix batch mismatch: model sees batch={batch_size}, "
                    f"prefix was generated for batch={orig_batch}"
                )
            factor = batch_size // orig_batch
            pk = pk.repeat(factor, 1, 1)
            pv = pv.repeat(factor, 1, 1)

        # ── Reshape: (batch, prefix_total, d_model) → (batch, n_heads, prefix_total, d_kv)
        P_K_r = pk.view(batch_size, prefix_total, n_heads, d_kv).transpose(1, 2)
        P_V_r = pv.view(batch_size, prefix_total, n_heads, d_kv).transpose(1, 2)

        # ── Prepend prefix KVs ─────────────────────────────────────────────
        key_states   = torch.cat([P_K_r, key_states],   dim=2)   # (b, h, prefix+seq, d)
        value_states = torch.cat([P_V_r, value_states], dim=2)

        # ── Attention scores ───────────────────────────────────────────────
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # ── Position bias ──────────────────────────────────────────────────
        if position_bias is None:
            # Layer 0: compute for extended key_length.
            key_length     = key_states.shape[-2]           # prefix_total + original_k
            orig_key_len   = key_length - prefix_total

            if cache_position is not None:
                real_seq_length = int(cache_position[-1].item()) + 1
            elif query_length is not None:
                real_seq_length = query_length
            else:
                real_seq_length = seq_length

            if not attn.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, n_heads, seq_length, key_length),
                    device=scores.device, dtype=scores.dtype,
                )
                if attn.gradient_checkpointing and attn.training:
                    position_bias.requires_grad = True
            else:
                position_bias = attn.compute_bias(
                    real_seq_length, key_length,
                    device=scores.device, cache_position=cache_position,
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                # mask shape: (batch, 1, 1-or-seq, orig_k_len), additive (0=attend, -inf=mask).
                # Prefix positions are always attended; prepend zeros.
                prefix_zeros = torch.zeros(
                    (batch_size, 1, seq_length, prefix_total),
                    device=mask.device, dtype=mask.dtype,
                )
                orig_mask_slice = mask[:, :, :, :orig_key_len]
                if orig_mask_slice.shape[2] == 1 and seq_length > 1:
                    orig_mask_slice = orig_mask_slice.expand(batch_size, 1, seq_length, -1)
                extended_mask = torch.cat([prefix_zeros, orig_mask_slice], dim=3)
                position_bias = position_bias + extended_mask

        # else: position_bias is from layer 0's patched forward and already
        # has shape (batch, n_heads, q_len, prefix+k_len) — use as-is.

        if attn.pruned_heads:
            head_mask = torch.ones(position_bias.shape[1], device=position_bias.device)
            head_mask[list(attn.pruned_heads)] = 0
            position_bias_masked = position_bias[:, head_mask.bool()]
        else:
            position_bias_masked = position_bias

        scores = scores + position_bias_masked

        # ── Softmax + dropout ──────────────────────────────────────────────
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=attn.dropout, training=attn.training
        )
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # ── Output projection ──────────────────────────────────────────────
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, attn.inner_dim)
        attn_output = attn.o(attn_output)

        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    return patched_forward


# ── public API ────────────────────────────────────────────────────────────────

def inject_prefix(
    chronos_model,
    prefix_generator,
    trend=None,
    seasonal=None,
    residual=None,
    prefix_kvs=None,
):
    """
    Monkey-patch the 6 encoder T5Attention layers to prepend prefix KVs.

    Two calling modes:

    Inference mode (gradient-free):
        inject_prefix(chronos_model, prefix_generator, trend, seasonal, residual)
        PrefixGenerator is called under torch.no_grad(); KVs are detached.

    Training mode (gradient-tracked):
        prefix_kvs = prefix_generator(trend, seasonal, residual)   # outside no_grad
        inject_prefix(chronos_model, prefix_generator, prefix_kvs=prefix_kvs)
        KVs are NOT detached; gradients flow back to PrefixGenerator during backward.

    Idempotent: calling again replaces the previous patch without double-saving
    the original forward.

    Args:
        chronos_model:    pipeline.model  (ChronosModel)
        prefix_generator: PrefixGenerator instance
        trend, seasonal, residual: numpy arrays or tensors (context_length,) or
                                   (batch, context_length)  [inference mode only]
        prefix_kvs:       pre-computed list of (P_K, P_V) tensors with grad_fn
                          [training mode only; mutually exclusive with trend/seasonal/residual]

    Returns:
        prefix_kvs: list of 6 (P_K, P_V) tensors
    """
    if prefix_kvs is not None:
        # Training mode: caller supplies gradient-tracked KVs
        detach = False
    else:
        # Inference mode: generate here under no_grad
        if trend is None or seasonal is None or residual is None:
            raise ValueError(
                "Provide either prefix_kvs (training) or "
                "trend + seasonal + residual (inference)"
            )
        with torch.no_grad():
            prefix_kvs = prefix_generator(trend, seasonal, residual)
        detach = True

    num_enc = len(chronos_model.model.encoder.block)
    if len(prefix_kvs) < num_enc:
        raise ValueError(
            f"PrefixGenerator returned {len(prefix_kvs)} KV pairs "
            f"but encoder has {num_enc} layers"
        )

    for i in range(num_enc):
        attn = chronos_model.model.encoder.block[i].layer[0].SelfAttention
        P_K, P_V = prefix_kvs[i]

        # Save original once (idempotent: skip if already saved)
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward

        if detach:
            P_K, P_V = P_K.detach(), P_V.detach()

        attn.forward = _make_patched_forward(attn, P_K, P_V)

    return prefix_kvs


def remove_prefix_hooks(chronos_model):
    """
    Restore original T5Attention.forward() on all encoder layers.
    Safe to call even if inject_prefix() was never called.
    """
    for i in range(len(chronos_model.model.encoder.block)):
        attn = chronos_model.model.encoder.block[i].layer[0].SelfAttention
        if hasattr(attn, "_original_forward"):
            attn.forward = attn._original_forward
            del attn._original_forward


# ── unit test ─────────────────────────────────────────────────────────────────

def _run_unit_test() -> None:
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "reasoning-finetuning"))

    from chronos import ChronosPipeline
    from prefix_generator import PrefixGenerator

    BATCH          = 2
    CONTEXT_LEN    = 512
    NUM_SAMPLES    = 4
    PRED_LEN       = 64
    D_MODEL        = 512
    NUM_LAYERS     = 6
    PREFIX_LEN     = 16
    PREFIX_TOTAL   = 3 * PREFIX_LEN   # 48

    print("=" * 60)
    print("prefix_injection — Unit Test")
    print("=" * 60)
    print(f"  batch={BATCH}  context_len={CONTEXT_LEN}  num_samples={NUM_SAMPLES}")
    print(f"  prefix_len_per_component={PREFIX_LEN}  prefix_total={PREFIX_TOTAL}")
    print()

    # ── Setup ─────────────────────────────────────────────────────────────
    print("Loading Chronos T5-Small …")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=torch.float32,
    )
    chronos_model = pipeline.model
    for p in chronos_model.parameters():
        p.requires_grad_(False)

    pg = PrefixGenerator(
        d_model=D_MODEL, num_layers=NUM_LAYERS, prefix_len_per_component=PREFIX_LEN
    )
    pg.eval()

    rng = np.random.default_rng(0)
    trend_np    = rng.standard_normal((BATCH, CONTEXT_LEN)).astype(np.float32)
    seasonal_np = rng.standard_normal((BATCH, CONTEXT_LEN)).astype(np.float32)
    residual_np = rng.standard_normal((BATCH, CONTEXT_LEN)).astype(np.float32)

    context = torch.randn(BATCH, CONTEXT_LEN)

    # Tokenise input once — reused for all direct encoder calls
    ctx_tensor = pipeline._prepare_and_validate_context(context)
    input_tokens, enc_mask, _ = pipeline.tokenizer.context_input_transform(ctx_tensor)

    # ── [e] Chronos params frozen ─────────────────────────────────────────
    n_trainable = sum(1 for p in chronos_model.parameters() if p.requires_grad)
    assert n_trainable == 0, f"Chronos should have 0 trainable params; got {n_trainable}"
    print(f"[e] Chronos trainable params (before injection): {n_trainable}  ✓")

    # ── Baseline encoder output (no prefix) ───────────────────────────────
    print("\nBaseline encoder forward pass …")
    with torch.no_grad():
        enc_baseline = chronos_model.model.encoder(
            input_ids=input_tokens, attention_mask=enc_mask
        )
    hidden_baseline = enc_baseline.last_hidden_state    # (batch, seq, d_model)

    # ── [a] inject_prefix() runs without error ────────────────────────────
    print("\n[a] inject_prefix() …")
    inject_prefix(chronos_model, pg, trend_np, seasonal_np, residual_np)
    print("    inject_prefix() completed without error  ✓")

    # ── [b] Output shape unchanged via full pipeline.predict() ───────────
    print("\n[b] pipeline.predict() shape check …")
    with torch.no_grad():
        samples = pipeline.predict(context, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
    expected_shape = (BATCH, NUM_SAMPLES, PRED_LEN)
    assert tuple(samples.shape) == expected_shape, \
        f"Expected shape {expected_shape}, got {tuple(samples.shape)}"
    print(f"    samples.shape = {tuple(samples.shape)}  ✓")

    # ── [c] Encoder hidden states differ with prefix ───────────────────────
    print("\n[c] Encoder hidden states change with prefix …")
    with torch.no_grad():
        enc_injected = chronos_model.model.encoder(
            input_ids=input_tokens, attention_mask=enc_mask
        )
    hidden_injected = enc_injected.last_hidden_state

    assert hidden_injected.shape == hidden_baseline.shape, (
        f"Shape changed: {hidden_injected.shape} vs {hidden_baseline.shape}"
    )
    max_diff = (hidden_injected - hidden_baseline).abs().max().item()
    differs  = max_diff > 1e-5
    assert differs, (
        "Prefix injection had NO effect on encoder hidden states "
        "(max diff = {:.2e}). Prefix is silently ignored.".format(max_diff)
    )
    print(f"    Hidden state shape unchanged: {tuple(hidden_injected.shape)}  ✓")
    print(f"    Max |Δhidden|  = {max_diff:.4e}  (prefix is being attended to)  ✓")

    # ── Per-layer attention-weight report ─────────────────────────────────
    # Threshold: total prefix attention = per-token prefix_w × PREFIX_TOTAL ≥ 0.05
    # (prefix tokens are getting at least 5% of total attention mass).
    # Per-token prefix_w ≥ 0.05 is physically impossible (max ≈ 1/560 ≈ 0.002).
    ATTN_THRESHOLD = 0.05

    print("\nPer-layer attention-weight report  (prefix vs input tokens):")
    print(f"  {'Layer':<6} {'Pfx/tok':>9}  {'Inp/tok':>9}  {'Ratio':>7}  {'TotalPfx':>9}  Status")
    print(f"  {'-'*5:<6} {'-'*9:>9}  {'-'*9:>9}  {'-'*7:>7}  {'-'*9:>9}  ------")

    with torch.no_grad():
        enc_attn = chronos_model.model.encoder(
            input_ids=input_tokens, attention_mask=enc_mask, output_attentions=True
        )

    prefix_attns = []
    for i, w in enumerate(enc_attn.attentions):
        # w: (batch, n_heads, q_len, PREFIX_TOTAL + k_len)
        prefix_w      = w[:, :, :, :PREFIX_TOTAL].mean().item()
        input_w       = w[:, :, :, PREFIX_TOTAL:].mean().item()
        ratio         = prefix_w / (input_w + 1e-9)
        total_pfx     = prefix_w * PREFIX_TOTAL   # fraction of attention on prefix
        low           = total_pfx < ATTN_THRESHOLD
        flag          = "  ⚠ LOW" if low else "  OK"
        print(f"  {i:<6} {prefix_w:>9.5f}  {input_w:>9.5f}  {ratio:>7.4f}  {total_pfx:>9.4f} {flag}")
        prefix_attns.append(prefix_w)

    cold_start_attns = prefix_attns[:]
    if all((a * PREFIX_TOTAL) < ATTN_THRESHOLD for a in prefix_attns):
        print(
            "\n  ⚠  Failure Mode 2 (cold start): prefix attention < 0.05 uniformly.\n"
            "     Now applying warm_start_from_chronos() …"
        )
    print()

    # ── warm-start: re-inject with Chronos-initialised prefix generator ───
    print("─" * 60)
    print("Warm-start: pg.warm_start_from_chronos(chronos_model)")
    print("─" * 60)
    remove_prefix_hooks(chronos_model)   # clean up previous patch
    pg.warm_start_from_chronos(chronos_model)

    inject_prefix(chronos_model, pg, trend_np, seasonal_np, residual_np)

    print("\nPer-layer attention-weight report (AFTER warm-start):")
    print(f"  {'Layer':<6} {'Pfx/tok':>9}  {'Inp/tok':>9}  {'Ratio':>7}  {'TotalPfx':>9}  Status")
    print(f"  {'-'*5:<6} {'-'*9:>9}  {'-'*9:>9}  {'-'*7:>7}  {'-'*9:>9}  ------")

    with torch.no_grad():
        enc_warm = chronos_model.model.encoder(
            input_ids=input_tokens, attention_mask=enc_mask, output_attentions=True
        )

    warm_prefix_attns = []
    for i, w in enumerate(enc_warm.attentions):
        prefix_w  = w[:, :, :, :PREFIX_TOTAL].mean().item()
        input_w   = w[:, :, :, PREFIX_TOTAL:].mean().item()
        ratio     = prefix_w / (input_w + 1e-9)
        total_pfx = prefix_w * PREFIX_TOTAL
        low       = total_pfx < ATTN_THRESHOLD
        delta_tok = prefix_w - cold_start_attns[i]
        flag      = "  ⚠ LOW" if low else "  OK"
        print(
            f"  {i:<6} {prefix_w:>9.5f}  {input_w:>9.5f}  {ratio:>7.4f}  "
            f"{total_pfx:>9.4f} {flag}  (Δtok {delta_tok:+.5f})"
        )
        warm_prefix_attns.append(prefix_w)

    layers_above = sum(1 for a in warm_prefix_attns if (a * PREFIX_TOTAL) >= ATTN_THRESHOLD)
    print(
        f"\n  Layers with total prefix attention ≥ {ATTN_THRESHOLD:.0%}: "
        f"{layers_above} / {NUM_LAYERS}"
    )
    if layers_above < 2:
        print(
            f"  ⚠  Fewer than 2 layers above {ATTN_THRESHOLD:.0%} total prefix attention.\n"
            "     Check warm_start_from_chronos: K_warm must be H.T @ Wk.T\n"
            "     (valid K projections), not raw Wk rows.\n"
            "     Also verify reshape order: P_K_r = pk.view(b, pfx, n_heads, d_kv).transpose(1,2)"
        )
    else:
        print("  Warm-start successful — proceed to training loop.")
    print()

    # ── [d] remove_prefix_hooks() restores exact baseline ─────────────────
    print("[d] remove_prefix_hooks() …")
    remove_prefix_hooks(chronos_model)

    with torch.no_grad():
        enc_restored = chronos_model.model.encoder(
            input_ids=input_tokens, attention_mask=enc_mask
        )
    hidden_restored = enc_restored.last_hidden_state

    max_restore_err = (hidden_restored - hidden_baseline).abs().max().item()
    assert max_restore_err < 1e-6, (
        f"Restored hidden states differ from baseline! "
        f"max |Δ| = {max_restore_err:.2e}"
    )
    print(f"    max |restored − baseline| = {max_restore_err:.2e}  ✓")

    # ── [e] re-check Chronos params still frozen ──────────────────────────
    n_trainable_post = sum(1 for p in chronos_model.parameters() if p.requires_grad)
    assert n_trainable_post == 0, f"Chronos params unfrozen during test! {n_trainable_post}"
    print(f"\n[e] Chronos trainable params (after removal): {n_trainable_post}  ✓")

    print()
    print("=" * 60)
    print("ALL ASSERTIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_unit_test()
