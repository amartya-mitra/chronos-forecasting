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

def _make_patched_forward(attn, P_K: torch.Tensor, P_V: torch.Tensor,
                          training_state=None, reset_position_bias: bool = False):
    """
    Return a replacement forward() for one T5Attention module.

    P_K, P_V: (orig_batch, prefix_total, d_model) — detached tensors from
              PrefixGenerator output for this layer.

    The replacement is stored as a plain instance attribute on `attn`,
    so nn.Module.__call__ routes through it without any special descriptor
    magic; `attn` is captured in the closure for attribute access.

    training_state (optional dict):
        When provided, applies an annealing prefix-attention ceiling to
        prevent attention collapse during training.  Not applied during
        inference (training_state=None).

        Required keys:
            'current_step'  (int)  : current training step
            'total_steps'   (int)  : total planned steps (for annealing)

        Optional keys populated by this closure:
            'ceiling_fired_accumulator' (list): appended with per-call
                firing rate (fraction of query positions where ceiling fired)
            'pfx_total_pre_ceil_accumulator' (list): appended with per-call
                mean prefix attention mass BEFORE clamping (diagnostic)
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

        # ── Position-bias reset (layer-split two-stage injection) ──────────
        # When Stage-2 prefix (16 tokens) follows Stage-1 prefix (48 tokens),
        # the position_bias arriving from layer 2 has the wrong last dimension.
        # Forcing None causes this layer to re-initialise it as zeros, which
        # is correct for layers that lack has_relative_attention_bias.
        if reset_position_bias:
            position_bias = None

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

        # ── Annealing prefix-attention ceiling (training only) ─────────────
        # Prevents attention collapse (pfx_total→1.0) seen in outlier series.
        # Only active when training_state is provided; inference is unchanged.
        #
        # Unified mode (training_state without 'stage1_len'):
        #   CEILING(step) = min(0.5 + 0.3 × (step / total_steps), 0.8)
        #
        # Split mode (training_state with 'stage1_len' — C1-v3):
        #   Stage-1 (first stage1_len tokens): unified ceiling above
        #   Stage-2 (remaining tokens):
        #     CEIL_2(step) = min(uniform_share × 1.2 × (step/total_steps),
        #                        uniform_share × 1.5)
        #     where uniform_share = stage2_len / total_attn_keys
        if training_state is not None:
            cur_step  = training_state.get('current_step', 0)
            tot_steps = max(training_state.get('total_steps', 2000), 1)
            stage1_len = training_state.get('stage1_len', None)

            if stage1_len is not None and prefix_total > stage1_len:
                # ── Split ceiling mode (C1-v3) ─────────────────────────────
                stage2_len = prefix_total - stage1_len  # e.g. 16

                pfx1_w = attn_weights[..., :stage1_len]
                pfx2_w = attn_weights[..., stage1_len:prefix_total]
                inp_w  = attn_weights[..., prefix_total:]

                sum1 = pfx1_w.sum(dim=-1, keepdim=True)  # (b, h, q, 1)
                sum2 = pfx2_w.sum(dim=-1, keepdim=True)

                # Stage-1 ceiling: 0.5 → 0.8 annealing
                ceil_1 = min(0.5 + 0.3 * (cur_step / tot_steps), 0.8)

                # Stage-2 ceiling: proportional share, slowly opening
                total_attn_keys = float(attn_weights.shape[-1])  # pfx_total + seq_len
                uniform_share   = stage2_len / total_attn_keys
                ceil_2 = min(
                    uniform_share * 1.2 * (cur_step / tot_steps),
                    uniform_share * 1.5,
                )

                # Track pre-ceiling values
                s1_pre = training_state.get('stage1_pre_ceil_accumulator')
                s2_pre = training_state.get('stage2_pre_ceil_accumulator')
                if s1_pre is not None:
                    s1_pre.append(sum1.detach().mean().item())
                if s2_pre is not None:
                    s2_pre.append(sum2.detach().mean().item())

                s2_us = training_state.get('stage2_uniform_share_accumulator')
                if s2_us is not None:
                    s2_us.append(uniform_share)

                # Clamp Stage-1
                scale1       = torch.where(sum1 > ceil_1,
                                           ceil_1 / (sum1 + 1e-9),
                                           torch.ones_like(sum1))
                pfx1_clamped = pfx1_w * scale1

                # Clamp Stage-2
                scale2       = torch.where(sum2 > ceil_2,
                                           ceil_2 / (sum2 + 1e-9),
                                           torch.ones_like(sum2))
                pfx2_clamped = pfx2_w * scale2

                # Redistribute remaining attention mass to input tokens
                inp_sum     = inp_w.sum(dim=-1, keepdim=True) + 1e-9
                remaining   = (1.0
                               - pfx1_clamped.sum(dim=-1, keepdim=True)
                               - pfx2_clamped.sum(dim=-1, keepdim=True))
                inp_boosted = inp_w * (remaining / inp_sum)

                # Track firing rates
                s1_fired = training_state.get('stage1_ceil_fired_accumulator')
                s2_fired = training_state.get('stage2_ceil_fired_accumulator')
                if s1_fired is not None:
                    s1_fired.append((sum1.detach() > ceil_1).float().mean().item())
                if s2_fired is not None:
                    s2_fired.append((sum2.detach() > ceil_2).float().mean().item())

                # Backward-compat combined accumulators (sum of both prefix groups)
                pfx_all_clamped = torch.cat([pfx1_clamped, pfx2_clamped], dim=-1)
                pfx_mass_comb   = pfx_all_clamped.sum(dim=-1, keepdim=True)
                pre_ceil_acc = training_state.get('pfx_total_pre_ceil_accumulator')
                if pre_ceil_acc is not None:
                    pre_ceil_acc.append(
                        (sum1 + sum2).detach().mean().item()
                    )
                fired_acc = training_state.get('ceiling_fired_accumulator')
                if fired_acc is not None:
                    fired_acc.append(
                        ((sum1.detach() > ceil_1) | (sum2.detach() > ceil_2))
                        .float().mean().item()
                    )

                attn_weights = torch.cat([pfx1_clamped, pfx2_clamped, inp_boosted], dim=-1)

            else:
                # ── Unified ceiling mode (original / backward-compatible) ──
                ceiling  = min(0.5 + 0.3 * (cur_step / tot_steps), 0.8)

                pfx_w    = attn_weights[..., :prefix_total]
                inp_w    = attn_weights[..., prefix_total:]
                pfx_mass = pfx_w.sum(dim=-1, keepdim=True)

                pre_ceil_acc = training_state.get('pfx_total_pre_ceil_accumulator')
                if pre_ceil_acc is not None:
                    pre_ceil_acc.append(pfx_mass.detach().mean().item())

                scale = torch.where(
                    pfx_mass > ceiling,
                    ceiling / (pfx_mass + 1e-9),
                    torch.ones_like(pfx_mass),
                )
                pfx_w_clamped = pfx_w * scale

                inp_sum       = inp_w.sum(dim=-1, keepdim=True) + 1e-9
                remaining     = 1.0 - pfx_w_clamped.sum(dim=-1, keepdim=True)
                inp_w_boosted = inp_w * (remaining / inp_sum)

                fired_acc = training_state.get('ceiling_fired_accumulator')
                if fired_acc is not None:
                    fired_acc.append(
                        (pfx_mass.detach() > ceiling).float().mean().item()
                    )

                attn_weights = torch.cat([pfx_w_clamped, inp_w_boosted], dim=-1)

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
    training_state=None,
):
    """
    Monkey-patch the 6 encoder T5Attention layers to prepend prefix KVs.

    Two calling modes:

    Inference mode (gradient-free):
        inject_prefix(chronos_model, prefix_generator, trend, seasonal, residual)
        PrefixGenerator is called under torch.no_grad(); KVs are detached.
        No ceiling applied (training_state=None).

    Training mode (gradient-tracked):
        prefix_kvs = prefix_generator(trend, seasonal, residual)   # outside no_grad
        inject_prefix(chronos_model, prefix_generator,
                      prefix_kvs=prefix_kvs, training_state=training_state)
        KVs are NOT detached; gradients flow back to PrefixGenerator during backward.
        Annealing ceiling active when training_state is provided.

    Idempotent: calling again replaces the previous patch without double-saving
    the original forward.

    Args:
        chronos_model:    pipeline.model  (ChronosModel)
        prefix_generator: PrefixGenerator instance
        trend, seasonal, residual: numpy arrays or tensors (context_length,) or
                                   (batch, context_length)  [inference mode only]
        prefix_kvs:       pre-computed list of (P_K, P_V) tensors with grad_fn
                          [training mode only; mutually exclusive with trend/seasonal/residual]
        training_state:   optional mutable dict enabling the annealing prefix-
                          attention ceiling (see _make_patched_forward docstring).
                          Must contain 'current_step' and 'total_steps'.

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

        attn.forward = _make_patched_forward(attn, P_K, P_V,
                                             training_state=training_state)

    return prefix_kvs


def _make_bias_reset_forward(attn):
    """
    Wrap T5Attention.forward to force position_bias=None on entry.

    Used on layer 3 when inject_prefix_first3 is active: layer 2's patched
    forward returns position_bias with shape (b, h, q, 48+k) but layer 3's
    unpatched attention scores have shape (b, h, q, k), causing a broadcast
    failure.  Resetting to None lets layer 3 re-initialise the bias as zeros
    of the correct shape (1, n_heads, seq, seq) — safe because layer 3 lacks
    has_relative_attention_bias.
    """
    def patched_forward(*args, **kwargs):
        kwargs['position_bias'] = None
        return attn._original_forward(*args, **kwargs)
    return patched_forward


def inject_prefix_first3(chronos_model, kv_list, training_state=None):
    """
    Inject prefix only on encoder layers 0-2.  Layers 3-5 are left unpatched
    except layer 3 which gets a position_bias reset to handle the shape
    mismatch caused by layer 2's 48-token prefix.

    Used as Pass-1 in two-stage training to capture encoder hidden states
    after layer 2 without yet having a Stage-2 prefix.

    Args:
        chronos_model: pipeline.model (ChronosModel)
        kv_list:       list of ≥3 (K, V) tuples from PrefixGenerator.forward()
        training_state: optional ceiling state dict (see _make_patched_forward)
    """
    for i in range(3):
        attn = chronos_model.model.encoder.block[i].layer[0].SelfAttention
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward
        P_K, P_V = kv_list[i]
        attn.forward = _make_patched_forward(
            attn, P_K.detach(), P_V.detach(), training_state=training_state
        )
    # Layer 3 receives position_bias of wrong shape from layer 2; reset it.
    attn3 = chronos_model.model.encoder.block[3].layer[0].SelfAttention
    if not hasattr(attn3, "_original_forward"):
        attn3._original_forward = attn3.forward
    attn3.forward = _make_bias_reset_forward(attn3)


def inject_prefix_split(
    chronos_model,
    kv1_list,
    K2: torch.Tensor,
    V2: torch.Tensor,
    training_state_1=None,
    training_state_2=None,
):
    """
    Layer-split two-stage prefix injection for Option C.

    Layers 0-2 receive Stage-1 prefix from kv1_list (48 tokens, 3×16).
    Layers 3-5 receive Stage-2 prefix (K2, V2) — the same 16-token prefix
    is shared across all three Stage-2 layers.

    Position bias is reset at layer 3 (zero-initialised) to handle the
    change in prefix_total from 48 (Stage-1) to 16 (Stage-2).  Layers 4-5
    then inherit the correctly-sized bias from layer 3.

    Args:
        chronos_model:    pipeline.model (ChronosModel)
        kv1_list:         list of ≥6 (K1, V1) tuples from Stage-1 PrefixGenerator
                          (only indices 0-2 are used for layers 0-2)
        K2, V2:           (batch, 16, d_model) Stage-2 prefix tensors; may carry
                          grad_fn when called from the training pass
        training_state_1: ceiling/monitoring state for Stage-1 layers (0-2)
        training_state_2: ceiling/monitoring state for Stage-2 layers (3-5)
    """
    if len(kv1_list) < 3:
        raise ValueError(f"kv1_list needs ≥3 entries; got {len(kv1_list)}")
    num_enc = len(chronos_model.model.encoder.block)
    if num_enc != 6:
        raise ValueError(f"Expected 6 encoder layers; got {num_enc}")

    for i in range(num_enc):
        attn = chronos_model.model.encoder.block[i].layer[0].SelfAttention
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward

        if i < 3:
            P_K, P_V = kv1_list[i]
            attn.forward = _make_patched_forward(
                attn, P_K.detach(), P_V.detach(),
                training_state=training_state_1,
                reset_position_bias=False,
            )
        else:
            # Same (K2, V2) applied to layers 3, 4, 5.
            # reset_position_bias=True only on layer 3 to re-initialise the
            # position_bias that arrived (with wrong shape) from layer 2.
            attn.forward = _make_patched_forward(
                attn, K2, V2,
                training_state=training_state_2,
                reset_position_bias=(i == 3),
            )


def inject_prefix_all_combined(
    chronos_model,
    kv1_list,
    K2: torch.Tensor,
    V2: torch.Tensor,
    training_state=None,
):
    """
    C1-v2: inject Stage-1 prefix (48-tok) + Stage-2 prefix (16-tok) on ALL 6 layers.

    Each layer i receives cat([kv1[i], K2], dim=1) — 64 tokens total.
    kv1 tokens are detached (Stage-1 frozen); K2/V2 may carry grad_fn (training).

    Ablation / E0 path: call inject_prefix(chronos_model, prefix_kvs=kv1_list)
    instead — injects only the 48-token Stage-1 prefix, identical to E0.
    SC1 passes by construction.
    """
    num_enc = len(chronos_model.model.encoder.block)
    if num_enc != 6:
        raise ValueError(f"Expected 6 encoder layers; got {num_enc}")
    if len(kv1_list) < 6:
        raise ValueError(f"kv1_list needs ≥6 entries; got {len(kv1_list)}")

    for i in range(num_enc):
        attn = chronos_model.model.encoder.block[i].layer[0].SelfAttention
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward
        K1i, V1i = kv1_list[i]
        # K2/V2 gradient preserved through cat (Stage-1 tokens detached)
        K_comb = torch.cat([K1i.detach(), K2], dim=1)   # (batch, 64, d_model)
        V_comb = torch.cat([V1i.detach(), V2], dim=1)
        attn.forward = _make_patched_forward(
            attn, K_comb, V_comb, training_state=training_state
        )


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
