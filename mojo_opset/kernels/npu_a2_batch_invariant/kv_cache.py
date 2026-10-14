from typing import Literal

import torch
import triton
import triton.language as tl

from mojo_opset.utils.kv_cache_metadata import assert_paged_kv_store_contract
from mojo_opset.utils.kv_cache_metadata import build_paged_kv_chunk_metadata

from .utils import libentry
from .utils import require_deterministic_mode

_COPY_BLOCK_SIZE = 256


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
    if chunk_metadata is None:
        if block_table is None or context_kv_lens is None:
            raise ValueError("block_table and context_kv_lens are required")
        chunk_metadata = build_paged_kv_chunk_metadata(block_table, cu_q_lens, context_kv_lens, key_cache.shape[2])
    assert_paged_kv_store_contract(chunk_metadata)
    slots = torch.full((key_states.shape[0],), -1, dtype=torch.int64, device=key_states.device)
    for source_start, block_id, block_offset, chunk_len in chunk_metadata.cpu().tolist():
        start = block_id * key_cache.shape[2] + block_offset
        slots[source_start : source_start + chunk_len] = torch.arange(
            start, start + chunk_len, dtype=torch.int64, device=key_states.device
        )
    _store_cache_slots(key_states, value_states, key_cache, value_cache, slots, layout="blocks_heads_page_dim")
    return key_cache, value_cache


store_paged_kv_cache_fwd.supports_separate_head_dims = True

__all__ = ["store_paged_kv_cache_fwd"]
