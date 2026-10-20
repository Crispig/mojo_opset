from typing import Literal

import torch
import triton
import triton.language as tl

from mojo_opset.utils.kv_cache_metadata import assert_paged_kv_store_contract

from .utils import get_num_cores
from .utils import libentry
from .utils import require_deterministic_mode

_COPY_BLOCK_SIZE = 256


@triton.jit
def _store_paged_chunks_kernel(
    key,
    value,
    key_cache,
    value_cache,
    chunks,
    stride_cm_row,
    stride_cm_col,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kb,
    stride_kch,
    stride_kp,
    stride_kcd,
    stride_vb,
    stride_vch,
    stride_vp,
    stride_vcd,
    NUM_CHUNKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    dims = tl.arange(0, BLOCK_D)
    for task in range(tl.program_id(0), NUM_CHUNKS * NUM_HEADS, tl.num_programs(0)):
        chunk = chunks + task // NUM_HEADS * stride_cm_row
        head = task % NUM_HEADS
        first = tl.load(chunk).to(tl.int64)
        block = tl.load(chunk + stride_cm_col).to(tl.int64)
        offset = tl.load(chunk + 2 * stride_cm_col)
        length = tl.load(chunk + 3 * stride_cm_col)
        if block >= 0:
            for start in range(0, length, BLOCK_T):
                positions = start + tl.arange(0, BLOCK_T)
                tokens = first + positions
                destination = offset + positions
                k_mask = (positions[:, None] < length) & (dims[None, :] < KEY_DIM)
                k_values = tl.load(
                    key + tokens[:, None] * stride_kt + head * stride_kh + dims[None, :] * stride_kd,
                    mask=k_mask,
                    other=0.0,
                )
                tl.store(
                    key_cache
                    + block * stride_kb
                    + head * stride_kch
                    + destination[:, None] * stride_kp
                    + dims[None, :] * stride_kcd,
                    k_values,
                    mask=k_mask,
                )
                v_mask = (positions[:, None] < length) & (dims[None, :] < VALUE_DIM)
                v_values = tl.load(
                    value + tokens[:, None] * stride_vt + head * stride_vh + dims[None, :] * stride_vd,
                    mask=v_mask,
                    other=0.0,
                )
                tl.store(
                    value_cache
                    + block * stride_vb
                    + head * stride_vch
                    + destination[:, None] * stride_vp
                    + dims[None, :] * stride_vcd,
                    v_values,
                    mask=v_mask,
                )


@triton.jit
def _chunks_from_blocks_kernel(
    block_table,
    cumulative_lengths,
    context_lengths,
    chunks,
    stride_batch,
    stride_page,
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    row = tl.program_id(0)
    if IS_DECODE:
        batch = row
        first = batch
        context = tl.load(context_lengths + batch)
        page = context // PAGE_SIZE
        valid = (context >= 0) & (page >= 0) & (page < NUM_PAGES)
        physical = tl.load(block_table + batch * stride_batch + page * stride_page, mask=valid, other=-1)
        source_start = first
        offset = context % PAGE_SIZE
        length = tl.where(valid & (physical >= 0), 1, 0)
    else:
        batch = row // NUM_PAGES
        page = row % NUM_PAGES
        first = tl.load(cumulative_lengths + batch)
        last = tl.load(cumulative_lengths + batch + 1)
        context = tl.load(context_lengths + batch)
        start = tl.maximum(context, page * PAGE_SIZE)
        end = tl.minimum(context + last - first, (page + 1) * PAGE_SIZE)
        physical = tl.load(block_table + batch * stride_batch + page * stride_page)
        source_start = first + start - context
        offset = start - page * PAGE_SIZE
        length = tl.where((context >= 0) & (physical >= 0), tl.maximum(end - start, 0), 0)
    tl.store(chunks + row * 4, source_start)
    tl.store(chunks + row * 4 + 1, physical)
    tl.store(chunks + row * 4 + 2, offset)
    tl.store(chunks + row * 4 + 3, length)


@libentry()
@triton.jit
def _store_cache_kernel(
    source,
    cache,
    slot_mapping,
    source_stride_token,
    source_stride_head,
    source_stride_dim,
    cache_stride_block,
    cache_stride_page,
    cache_stride_head,
    cache_stride_dim,
    page_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping + token_idx).to(tl.int64)
    valid_slot = slot >= 0
    block_idx = tl.maximum(slot, 0) // page_size
    page_offset = tl.maximum(slot, 0) % page_size

    feature_count: tl.constexpr = num_heads * head_dim
    for feature_start in range(0, feature_count, BLOCK_SIZE):
        feature_offsets = feature_start + tl.arange(0, BLOCK_SIZE)
        feature_mask = feature_offsets < feature_count
        head_offsets = feature_offsets // head_dim
        dim_offsets = feature_offsets % head_dim
        source_offsets = (
            token_idx * source_stride_token + head_offsets * source_stride_head + dim_offsets * source_stride_dim
        )
        cache_offsets = (
            block_idx * cache_stride_block
            + page_offset * cache_stride_page
            + head_offsets * cache_stride_head
            + dim_offsets * cache_stride_dim
        )
        values = tl.load(source + source_offsets, mask=feature_mask, other=0.0)
        tl.store(cache + cache_offsets, values, mask=valid_slot & feature_mask)


def _cache_strides(
    cache: torch.Tensor,
    layout: Literal["blocks_page_heads_dim", "blocks_heads_page_dim"],
) -> tuple[int, int, int, int, int]:
    if cache.ndim != 4:
        raise ValueError(f"cache must be 4D, got shape {tuple(cache.shape)}")
    if layout == "blocks_page_heads_dim":
        _, page_size, _, _ = cache.shape
        stride_block, stride_page, stride_head, stride_dim = cache.stride()
    elif layout == "blocks_heads_page_dim":
        _, _, page_size, _ = cache.shape
        stride_block, stride_head, stride_page, stride_dim = cache.stride()
    else:
        raise ValueError(f"Unsupported cache layout: {layout!r}")
    return stride_block, stride_page, stride_head, stride_dim, page_size


def _validate_slot_mapping(slot_mapping: torch.Tensor, num_slots: int) -> None:
    if slot_mapping.ndim != 1:
        raise ValueError(f"slot_mapping must be 1D, got shape {tuple(slot_mapping.shape)}")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"slot_mapping must use int32 or int64, got {slot_mapping.dtype}")

    slots = slot_mapping.detach().to(device="cpu", dtype=torch.int64)
    valid = slots[slots >= 0]
    if bool((slots < -1).any()):
        raise ValueError("slot_mapping only accepts -1 as the invalid slot sentinel")
    if valid.numel() and int(valid.max()) >= num_slots:
        raise ValueError(f"slot_mapping contains a slot outside [0, {num_slots})")
    if valid.numel() != torch.unique(valid).numel():
        raise ValueError("valid cache slots must be unique to avoid write races")


def _store_single_cache(
    source: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    layout: Literal["blocks_page_heads_dim", "blocks_heads_page_dim"],
    validate_slots: bool,
) -> None:
    if source.ndim != 3:
        raise ValueError(f"source must have shape [tokens, heads, dim], got {tuple(source.shape)}")
    if not source.is_contiguous():
        source = source.contiguous()

    num_tokens, num_heads, head_dim = source.shape
    stride_block, stride_page, stride_head, stride_dim, page_size = _cache_strides(cache, layout)
    cache_heads = cache.shape[2] if layout == "blocks_page_heads_dim" else cache.shape[1]
    if cache_heads != num_heads or cache.shape[-1] != head_dim:
        raise ValueError(
            "source/cache head shape mismatch: "
            f"source={tuple(source.shape)}, cache={tuple(cache.shape)}, layout={layout}"
        )
    if slot_mapping.numel() != num_tokens:
        raise ValueError(f"slot_mapping has {slot_mapping.numel()} entries for {num_tokens} source tokens")
    if slot_mapping.device != source.device or cache.device != source.device:
        raise ValueError("source, cache, and slot_mapping must be on the same device")

    if validate_slots:
        _validate_slot_mapping(slot_mapping, cache.shape[0] * page_size)
    if num_tokens == 0:
        return

    _store_cache_kernel[(num_tokens,)](
        source,
        cache,
        slot_mapping,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        stride_block,
        stride_page,
        stride_head,
        stride_dim,
        page_size=page_size,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE=_COPY_BLOCK_SIZE,
    )


def _store_cache_slots(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    layout: Literal["blocks_page_heads_dim", "blocks_heads_page_dim"] = "blocks_page_heads_dim",
    validate_slots: bool = True,
) -> None:
    """Scatter key/value rows into unique physical cache slots.

    ``-1`` slots are ignored. Valid slots must be unique; otherwise parallel
    writers would make the result scheduling-dependent.
    """

    require_deterministic_mode()
    if key.shape[:2] != value.shape[:2]:
        raise ValueError(
            f"key and value must have the same token/head dimensions, got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    _store_single_cache(
        key,
        key_cache,
        slot_mapping,
        layout=layout,
        validate_slots=validate_slots,
    )
    _store_single_cache(
        value,
        value_cache,
        slot_mapping,
        layout=layout,
        validate_slots=validate_slots,
    )


def store_paged_kv_cache_fwd(
    key_states, value_states, key_cache, value_cache, block_table, cu_q_lens, context_kv_lens, chunk_metadata
):
    """Write caller-validated cache metadata without reading device contents on the host."""

    require_deterministic_mode()
    if key_states.shape[:2] != value_states.shape[:2]:
        raise ValueError("key and value must have matching token/head dimensions")
    for source, cache in ((key_states, key_cache), (value_states, value_cache)):
        if source.ndim != 3 or cache.ndim != 4:
            raise ValueError("source/cache must have shape [tokens, heads, dim]/[pages, heads, page, dim]")
        if cache.shape[1] != source.shape[1] or cache.shape[3] != source.shape[2]:
            raise ValueError("source/cache head shape mismatch")
        if source.device != key_states.device or cache.device != key_states.device:
            raise ValueError("source, cache, and metadata must be on the same device")
        if source.dtype != cache.dtype:
            raise TypeError("source and cache must use the same dtype")
    if key_cache.shape[:3] != value_cache.shape[:3]:
        raise ValueError("key/value caches must have matching page/head dimensions")
    if not key_states.shape[0]:
        return key_cache, value_cache
    page_size = key_cache.shape[2]
    if chunk_metadata is not None:
        assert_paged_kv_store_contract(chunk_metadata)
        if chunk_metadata.device != key_states.device:
            raise ValueError("chunk_metadata and key_states must be on the same device")
    else:
        if block_table is None or context_kv_lens is None:
            raise ValueError("block_table and context_kv_lens are required")
        if block_table.dtype != torch.int32 or block_table.ndim != 2:
            raise ValueError("block_table must have shape [batch, pages] and use int32")
        if context_kv_lens.dtype != torch.int32 or context_kv_lens.shape != (block_table.shape[0],):
            raise ValueError("context_kv_lens must have shape [batch] and use int32")
        if block_table.device != key_states.device or context_kv_lens.device != key_states.device:
            raise ValueError("cache metadata and key_states must be on the same device")
        if context_kv_lens.stride(0) != 1:
            raise ValueError("context_kv_lens must be contiguous")
        if cu_q_lens is not None:
            if cu_q_lens.dtype != torch.int32 or cu_q_lens.shape != (block_table.shape[0] + 1,):
                raise ValueError("cu_q_lens must have shape [batch + 1] and use int32")
            if cu_q_lens.device != key_states.device or cu_q_lens.stride(0) != 1:
                raise ValueError("cu_q_lens must be contiguous and on the same device as key_states")
        elif key_states.shape[0] != block_table.shape[0]:
            raise ValueError("decode requires one source token per sequence")
        chunk_count = block_table.shape[0] * (1 if cu_q_lens is None else block_table.shape[1])
        chunk_metadata = torch.empty((chunk_count, 4), device=key_states.device, dtype=torch.int32)
        if chunk_count:
            _chunks_from_blocks_kernel[(chunk_count,)](
                block_table,
                cu_q_lens,
                context_kv_lens,
                chunk_metadata,
                block_table.stride(0),
                block_table.stride(1),
                NUM_PAGES=block_table.shape[1],
                PAGE_SIZE=page_size,
                IS_DECODE=cu_q_lens is None,
            )
    tasks = chunk_metadata.shape[0] * key_states.shape[1]
    if tasks:
        _store_paged_chunks_kernel[(min(tasks, get_num_cores("vector")),)](
            key_states,
            value_states,
            key_cache,
            value_cache,
            chunk_metadata,
            chunk_metadata.stride(0),
            chunk_metadata.stride(1),
            *key_states.stride(),
            *value_states.stride(),
            *key_cache.stride(),
            *value_cache.stride(),
            NUM_CHUNKS=chunk_metadata.shape[0],
            NUM_HEADS=key_states.shape[1],
            KEY_DIM=key_states.shape[2],
            VALUE_DIM=value_states.shape[2],
            BLOCK_T=64,
            BLOCK_D=triton.next_power_of_2(max(key_states.shape[2], value_states.shape[2])),
            multibuffer=False,
        )
    return key_cache, value_cache


store_paged_kv_cache_fwd.supports_separate_head_dims = True

__all__ = ["store_paged_kv_cache_fwd"]
