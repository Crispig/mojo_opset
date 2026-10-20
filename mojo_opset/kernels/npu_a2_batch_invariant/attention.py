from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import get_num_cores
from .utils import require_deterministic_mode

_BLOCK_M = 64
_BLOCK_N = 64
_HEAD_DIM = 128
_PAGE_SIZE = 256


@triton.jit
def _attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_q_lens_ptr,
    cu_k_lens_ptr,
    seq_lens_ptr,
    page_table_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kp,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vp,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_ptb,
    stride_ptp,
    softmax_scale: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUPED_DECODE: tl.constexpr = False,
):
    """Causal packed/paged forward with an identical logical reduction chain."""

    batch_id = tl.program_id(0)
    core_id = tl.program_id(1)
    core_count = tl.num_programs(1)

    q_start = tl.load(cu_q_lens_ptr + batch_id).to(tl.int32)
    q_end = tl.load(cu_q_lens_ptr + batch_id + 1).to(tl.int32)
    q_len = q_end - q_start
    if PAGED:
        k_start = 0
        kv_len = tl.load(seq_lens_ptr + batch_id).to(tl.int32)
    else:
        k_start = tl.load(cu_k_lens_ptr + batch_id).to(tl.int32)
        k_end = tl.load(cu_k_lens_ptr + batch_id + 1).to(tl.int32)
        kv_len = k_end - k_start

    # Prefill tiles are anchored to absolute KV positions. Grouped decode packs
    # independent heads into the same Cube tile, retaining absolute causal
    # positions and the original per-head QK/PV reduction shapes and KV chain.
    q_abs_start = kv_len - q_len
    first_q_block = q_abs_start // BLOCK_M
    q_block_count = tl.cdiv(kv_len, BLOCK_M) - first_q_block
    if GROUPED_DECODE:
        tl.static_assert(PAGED and NUM_Q_HEADS // NUM_KV_HEADS <= BLOCK_M)
        task_count = tl.where(kv_len > 0, NUM_KV_HEADS, 0)
    else:
        task_count = q_block_count * NUM_Q_HEADS

    offs_d = tl.arange(0, BLOCK_D)
    qk_scale = softmax_scale * 1.4426950408889634

    for task_id in range(core_id, task_count, core_count):
        if GROUPED_DECODE:
            kv_head_id = task_id
            q_heads = kv_head_id * (NUM_Q_HEADS // NUM_KV_HEADS) + tl.arange(0, BLOCK_M)
            offs_m = tl.zeros((BLOCK_M,), tl.int32) + q_abs_start
            valid_q = tl.arange(0, BLOCK_M) < NUM_Q_HEADS // NUM_KV_HEADS
            q_ptrs = (
                q_ptr + q_start.to(tl.int64) * stride_qt + q_heads[:, None] * stride_qh + offs_d[None, :] * stride_qd
            )
        else:
            q_block_id = first_q_block + task_id // NUM_Q_HEADS
            q_head_id = task_id % NUM_Q_HEADS
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)
            offs_m = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
            valid_q = (offs_m >= q_abs_start) & (offs_m < kv_len)
            q_local = tl.where(valid_q, offs_m - q_abs_start, 0)
            q_token = (q_start + q_local).to(tl.int64)
            q_ptrs = q_ptr + q_token[:, None] * stride_qt + q_head_id * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(
            q_ptrs,
            mask=valid_q[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for kv_block_id in range(0, tl.cdiv(kv_len, BLOCK_N)):
            offs_n = kv_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
            valid_n = offs_n < kv_len

            if PAGED:
                tl.static_assert(PAGE_SIZE % BLOCK_N == 0)
                logical_page = kv_block_id * BLOCK_N // PAGE_SIZE
                page_offset = kv_block_id * BLOCK_N % PAGE_SIZE
                physical_page = tl.load(
                    page_table_ptr + batch_id * stride_ptb + logical_page * stride_ptp,
                ).to(tl.int64)
                k_base = k_ptr + physical_page * stride_kt + kv_head_id * stride_kh + page_offset * stride_kp
                v_base = v_ptr + physical_page * stride_vt + kv_head_id * stride_vh + page_offset * stride_vp
                k_token_stride, v_token_stride = stride_kp, stride_vp
            else:
                k_token = (k_start + kv_block_id * BLOCK_N).to(tl.int64)
                k_base = k_ptr + k_token * stride_kt + kv_head_id * stride_kh
                v_base = v_ptr + k_token * stride_vt + kv_head_id * stride_vh
                k_token_stride, v_token_stride = stride_kt, stride_vt

            k_block = tl.make_block_ptr(
                base=k_base,
                shape=(tl.minimum(BLOCK_N, kv_len - kv_block_id * BLOCK_N), HEAD_DIM),
                strides=(k_token_stride, stride_kd),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            v_block = tl.make_block_ptr(
                base=v_base,
                shape=(tl.minimum(BLOCK_N, kv_len - kv_block_id * BLOCK_N), HEAD_DIM),
                strides=(v_token_stride, stride_vd),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )

            kv_mask = valid_n[:, None] & (offs_d[None, :] < HEAD_DIM)
            k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
            # On Ascend, block-pointer padding alone does not keep dummy QK
            # columns zero through dot lowering. The finite mask needs them.
            k = tl.where(kv_mask, k, 0.0)
            v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, tl.trans(k))
            causal_mask = valid_q[:, None] & valid_n[None, :] & (offs_m[:, None] >= offs_n[None, :])
            qk = qk * qk_scale + tl.where(causal_mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            p_cast = p.to(k.dtype)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = tl.dot(p_cast, v, acc * alpha[:, None])
            m_i = m_ij

        output = acc / l_i[:, None]
        if GROUPED_DECODE:
            out_ptrs = (
                out_ptr + q_start.to(tl.int64) * stride_ot + q_heads[:, None] * stride_oh + offs_d[None, :] * stride_od
            )
        else:
            out_ptrs = out_ptr + q_token[:, None] * stride_ot + q_head_id * stride_oh + offs_d[None, :] * stride_od
        tl.store(
            out_ptrs,
            output.to(out_ptr.type.element_ty),
            mask=valid_q[:, None] & (offs_d[None, :] < HEAD_DIM),
        )


def _validate_tnd(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be an NPU tensor, got {tensor.device}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must use torch.bfloat16, got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[-1] != _HEAD_DIM:
        raise ValueError(f"{name} must have shape [tokens, heads, {_HEAD_DIM}], got {tuple(tensor.shape)}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in the head dimension")


def _validate_cumulative_lengths(
    name: str,
    cumulative_lengths: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    if cumulative_lengths.device != device:
        raise ValueError(f"{name} must be on {device}, got {cumulative_lengths.device}")
    if cumulative_lengths.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32, got {cumulative_lengths.dtype}")
    if cumulative_lengths.ndim != 1 or cumulative_lengths.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional cumulative-length tensor")
    if cumulative_lengths.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")


def _validate_common(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    require_deterministic_mode()
    _validate_tnd("q", q)
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same NPU device")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must use the same dtype")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same number of heads")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")


def attention_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Batch-invariant packed causal self-attention for Qwen3 BF16/D128.

    The caller supplies matching, strictly increasing cumulative lengths.
    Metadata contents stay on device so this path supports graph replay.
    """

    _validate_common(q, k, v)
    _validate_tnd("k", k)
    _validate_tnd("v", v)
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same token count")
    _validate_cumulative_lengths("cu_q_lens", cu_q_lens, device=q.device)
    _validate_cumulative_lengths("cu_k_lens", cu_k_lens, device=q.device)
    if cu_q_lens.shape != cu_k_lens.shape or q.shape[0] != k.shape[0]:
        raise NotImplementedError("The initial kernel supports causal self-attention only")

    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    output = torch.empty_like(q)
    batch_size = cu_q_lens.numel() - 1
    _attention_fwd_kernel[(batch_size, get_num_cores("cube"))](
        q,
        k,
        v,
        output,
        cu_q_lens,
        cu_k_lens,
        cu_k_lens,
        cu_k_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        0,
        k.stride(2),
        v.stride(0),
        v.stride(1),
        0,
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        0,
        0,
        scale,
        q.shape[1],
        k.shape[1],
        _HEAD_DIM,
        _PAGE_SIZE,
        PAGED=False,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_HEAD_DIM,
        multibuffer=False,
    )
    return output


def attention_paged(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: Optional[float] = None,
    *,
    grouped_decode: bool = False,
) -> torch.Tensor:
    """Batch-invariant paged causal attention sharing the packed arithmetic.

    The caller validates cumulative/query lengths and active physical pages
    before capture; replay reads their updated contents directly on device.
    """

    require_deterministic_mode()
    _validate_tnd("q", q)
    if q.requires_grad or key_cache.requires_grad or value_cache.requires_grad:
        raise RuntimeError("Paged attention is inference-only")
    if key_cache.device != q.device or value_cache.device != q.device:
        raise ValueError("q and both caches must be on the same NPU device")
    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        raise TypeError("q and both caches must use the same dtype")
    expected_rank = 4
    if key_cache.ndim != expected_rank or value_cache.ndim != expected_rank:
        raise ValueError("key_cache and value_cache must use [pages, heads, page, dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have identical shapes")
    if key_cache.shape[2:] != (_PAGE_SIZE, _HEAD_DIM):
        raise ValueError(
            f"cache must use page_size={_PAGE_SIZE} and head_dim={_HEAD_DIM}, got {tuple(key_cache.shape)}"
        )
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError("KV caches must be contiguous in the head dimension")
    if q.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if cu_q_lens.device != q.device or seq_lens.device != q.device or block_tables.device != q.device:
        raise ValueError("attention metadata must be on the same NPU device as q")
    if cu_q_lens.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("cu_q_lens and seq_lens must use torch.int32")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must use torch.int32")
    batch_size = cu_q_lens.numel() - 1
    if cu_q_lens.ndim != 1 or batch_size < 1:
        raise ValueError("cu_q_lens must be a one-dimensional cumulative-length tensor")
    if seq_lens.shape != (batch_size,) or block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        raise ValueError("cu_q_lens, seq_lens, and block_tables must describe the same batch")
    if seq_lens.stride(0) != 1:
        raise ValueError("seq_lens must be contiguous")
    _validate_cumulative_lengths("cu_q_lens", cu_q_lens, device=q.device)

    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    output = torch.empty_like(q)
    # Small-batch decode packs only independent heads; KV reductions stay fixed.
    block_m = 16 if grouped_decode and batch_size <= 2 and q.shape[1] // key_cache.shape[1] <= 16 else _BLOCK_M
    _attention_fwd_kernel[(batch_size, get_num_cores("cube"))](
        q,
        key_cache,
        value_cache,
        output,
        cu_q_lens,
        cu_q_lens,
        seq_lens,
        block_tables,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        scale,
        q.shape[1],
        key_cache.shape[1],
        _HEAD_DIM,
        _PAGE_SIZE,
        PAGED=True,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_HEAD_DIM,
        GROUPED_DECODE=grouped_decode,
        multibuffer=False,
    )
    return output


def flash_attention_infer_fwd(
    q,
    k,
    v,
    attention_mask,
    cu_q_lens,
    cu_k_lens,
    max_q_len,
    max_k_len,
    dropout_p,
    softmax_scale,
    causal,
    deterministic,
):
    if attention_mask is not None or dropout_p != 0.0 or not causal:
        raise NotImplementedError("Batch-invariant flash attention requires causal attention without mask/dropout")
    return attention_packed(q, k, v, cu_q_lens, cu_k_lens, softmax_scale)


def _validate_attention(is_causal, gqa_layout, mask):
    if not is_causal or gqa_layout != "AABB" or mask is not None:
        raise NotImplementedError("Batch-invariant attention requires causal AABB GQA without a custom mask")


def paged_prefill_gqa_infer_fwd(
    query,
    key_cache,
    value_cache,
    cu_q_lens,
    block_tables,
    softmax_scale,
    cu_total_seq_lens,
    mask,
    max_q_len,
    max_total_seq_len,
    is_causal,
    gqa_layout,
    scheduler_metadata,
):
    _validate_attention(is_causal, gqa_layout, mask)
    if scheduler_metadata is not None:
        raise NotImplementedError("Batch-invariant attention does not use external scheduler metadata")
    cumulative = cu_q_lens if cu_total_seq_lens is None else cu_total_seq_lens
    return attention_paged(
        query,
        key_cache,
        value_cache,
        cu_q_lens,
        (cumulative[1:] - cumulative[:-1]).contiguous(),
        block_tables,
        softmax_scale,
    )


def paged_decode_gqa_infer_fwd(
    query,
    key_cache,
    value_cache,
    total_seq_lens,
    block_tables,
    softmax_scale,
    mask,
    max_total_seq_len,
    is_causal,
    gqa_layout,
):
    _validate_attention(is_causal, gqa_layout, mask)
    cumulative = torch.arange(query.shape[0] + 1, dtype=torch.int32, device=query.device)
    return attention_paged(
        query,
        key_cache,
        value_cache,
        cumulative,
        total_seq_lens,
        block_tables,
        softmax_scale,
        grouped_decode=True,
    )


__all__ = [
    "attention_packed",
    "attention_paged",
    "flash_attention_infer_fwd",
    "paged_prefill_gqa_infer_fwd",
    "paged_decode_gqa_infer_fwd",
]
