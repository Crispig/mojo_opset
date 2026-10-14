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

    # Anchor the query tile to its absolute position in the logical KV stream.
    # A decode token therefore occupies the same matrix row as it did in the
    # packed training/prefill call, instead of switching to a q_len=1 kernel.
    q_abs_start = kv_len - q_len
    first_q_block = q_abs_start // BLOCK_M
    q_block_count = tl.cdiv(kv_len, BLOCK_M) - first_q_block
    task_count = q_block_count * NUM_Q_HEADS

    offs_d = tl.arange(0, BLOCK_D)
    qk_scale = softmax_scale * 1.4426950408889634

    for task_id in range(core_id, task_count, core_count):
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
            safe_n = tl.where(valid_n, offs_n, 0)

            if PAGED:
                logical_page = safe_n // PAGE_SIZE
                page_offset = safe_n - logical_page * PAGE_SIZE
                physical_page = tl.load(
                    page_table_ptr + batch_id * stride_ptb + logical_page * stride_ptp,
                    mask=valid_n,
                    other=0,
                ).to(tl.int64)
                k_ptrs = (
                    k_ptr
                    + physical_page[:, None] * stride_kt
                    + kv_head_id * stride_kh
                    + page_offset[:, None] * stride_kp
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + physical_page[:, None] * stride_vt
                    + kv_head_id * stride_vh
                    + page_offset[:, None] * stride_vp
                    + offs_d[None, :] * stride_vd
                )
            else:
                k_token = (k_start + safe_n).to(tl.int64)
                k_ptrs = k_ptr + k_token[:, None] * stride_kt + kv_head_id * stride_kh + offs_d[None, :] * stride_kd
                v_ptrs = v_ptr + k_token[:, None] * stride_vt + kv_head_id * stride_vh + offs_d[None, :] * stride_vd

            kv_mask = valid_n[:, None] & (offs_d[None, :] < HEAD_DIM)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

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
    total_tokens: int,
) -> tuple[int, ...]:
    if cumulative_lengths.device != device:
        raise ValueError(f"{name} must be on {device}, got {cumulative_lengths.device}")
    if cumulative_lengths.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32, got {cumulative_lengths.dtype}")
    if cumulative_lengths.ndim != 1 or cumulative_lengths.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional cumulative-length tensor")
    if cumulative_lengths.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")
    boundaries = tuple(int(value) for value in cumulative_lengths.detach().cpu().tolist())
    if boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError(f"{name} must start at 0 and end at {total_tokens}, got {boundaries}")
    if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError(f"{name} must be strictly increasing, got {boundaries}")
    return boundaries


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
    """Batch-invariant packed causal attention for Qwen3 BF16/D128."""

    _validate_common(q, k, v)
    _validate_tnd("k", k)
    _validate_tnd("v", v)
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same token count")
    q_boundaries = _validate_cumulative_lengths("cu_q_lens", cu_q_lens, device=q.device, total_tokens=q.shape[0])
    k_boundaries = _validate_cumulative_lengths("cu_k_lens", cu_k_lens, device=q.device, total_tokens=k.shape[0])
    if q_boundaries != k_boundaries:
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
) -> torch.Tensor:
    """Batch-invariant paged causal attention sharing the packed arithmetic."""

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
    q_boundaries = _validate_cumulative_lengths("cu_q_lens", cu_q_lens, device=q.device, total_tokens=q.shape[0])
    sequence_lengths = tuple(int(value) for value in seq_lens.detach().cpu().tolist())
    query_lengths = tuple(end - start for start, end in zip(q_boundaries, q_boundaries[1:]))
    if any(q_len > kv_len or kv_len <= 0 for q_len, kv_len in zip(query_lengths, sequence_lengths)):
        raise ValueError("Each query length must be positive and no greater than its KV length")
    pages_needed = tuple((length + _PAGE_SIZE - 1) // _PAGE_SIZE for length in sequence_lengths)
    if any(count > block_tables.shape[1] for count in pages_needed):
        raise ValueError("block_tables does not contain enough logical pages")
    active_pages = block_tables.detach().cpu()
    for batch_id, count in enumerate(pages_needed):
        pages = active_pages[batch_id, :count]
        if bool(((pages < 0) | (pages >= key_cache.shape[0])).any().item()):
            raise ValueError("block_tables contains an invalid physical page")

    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    output = torch.empty_like(q)
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
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_HEAD_DIM,
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
    return attention_paged(query, key_cache, value_cache, cumulative, total_seq_lens, block_tables, softmax_scale)


__all__ = [
    "attention_packed",
    "attention_paged",
    "flash_attention_infer_fwd",
    "paged_prefill_gqa_infer_fwd",
    "paged_decode_gqa_infer_fwd",
]
