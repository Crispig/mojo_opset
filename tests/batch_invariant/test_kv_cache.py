import pytest
import torch

from mojo_opset.kernels.npu_a2_batch_invariant.kv_cache import _store_cache_slots as store_kvcache

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_store_kvcache_matches_reference_and_is_order_invariant(dtype):
    torch.manual_seed(17)
    device = torch.device("npu")
    tokens, blocks, page_size, heads = 7, 4, 4, 3
    key = torch.randn(tokens, heads, 5, dtype=dtype, device=device)
    value = torch.randn(tokens, heads, 7, dtype=dtype, device=device)
    slots = torch.tensor([5, -1, 0, 14, 3, 9, 7], dtype=torch.int64, device=device)

    key_cache = torch.zeros(blocks, page_size, heads, 5, dtype=dtype, device=device)
    value_cache = torch.zeros(blocks, page_size, heads, 7, dtype=dtype, device=device)
    store_kvcache(key, value, key_cache, value_cache, slots)

    key_ref = torch.zeros_like(key_cache)
    value_ref = torch.zeros_like(value_cache)
    for token_idx, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            block_idx, page_offset = divmod(slot, page_size)
            key_ref[block_idx, page_offset] = key[token_idx]
            value_ref[block_idx, page_offset] = value[token_idx]
    assert torch.equal(key_cache, key_ref)
    assert torch.equal(value_cache, value_ref)

    permutation = torch.tensor([6, 2, 4, 0, 5, 1, 3], device=device)
    key_permuted = torch.zeros_like(key_cache)
    value_permuted = torch.zeros_like(value_cache)
    store_kvcache(
        key[permutation],
        value[permutation],
        key_permuted,
        value_permuted,
        slots[permutation],
    )
    assert torch.equal(key_permuted, key_cache)
    assert torch.equal(value_permuted, value_cache)


def test_store_kvcache_rejects_duplicate_slots():
    device = torch.device("npu")
    key = torch.randn(2, 1, 4, device=device)
    value = torch.randn(2, 1, 4, device=device)
    cache = torch.zeros(1, 4, 1, 4, device=device)
    slots = torch.tensor([1, 1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="unique"):
        store_kvcache(key, value, cache, cache.clone(), slots)


def test_store_kvcache_accepts_an_empty_token_batch():
    key = torch.empty(0, 2, 7, device="npu", dtype=torch.bfloat16)
    value = torch.empty(0, 2, 5, device="npu", dtype=torch.bfloat16)
    key_cache = torch.zeros(2, 4, 2, 7, device="npu", dtype=torch.bfloat16)
    value_cache = torch.zeros(2, 4, 2, 5, device="npu", dtype=torch.bfloat16)
    slots = torch.empty(0, device="npu", dtype=torch.int64)

    result = store_kvcache(
        key,
        value,
        key_cache,
        value_cache,
        slots,
    )
    assert result is None
    assert torch.count_nonzero(key_cache).item() == 0
    assert torch.count_nonzero(value_cache).item() == 0
