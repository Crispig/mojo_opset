import pytest
import torch

from mojo_opset.functions.kv_cache import store_paged_kv_cache

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="VeXact compatibility tests require an Ascend NPU",
)


def test_store_kvcache_separate_dims() -> None:
    """Port the DeepSeek qk=192/value=128 cache-store regression."""
    torch.manual_seed(42)
    tokens, heads, key_dim, value_dim = 8, 16, 192, 128
    page_size, blocks = 256, 2
    key = torch.randn(tokens, heads, key_dim, device="npu", dtype=torch.bfloat16)
    value = torch.randn(tokens, heads, value_dim, device="npu", dtype=torch.bfloat16)
    key_cache = torch.zeros(
        blocks,
        heads,
        page_size,
        key_dim,
        device="npu",
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros(
        blocks,
        heads,
        page_size,
        value_dim,
        device="npu",
        dtype=torch.bfloat16,
    )
    chunks = torch.tensor([[0, 0, 0, tokens]], device="npu", dtype=torch.int32)
    store_paged_kv_cache(key, value, key_cache, value_cache, chunk_metadata=chunks, implementation="batch_invariant")

    for token in range(tokens):
        assert torch.equal(key_cache[0, :, token], key[token])
        assert torch.equal(value_cache[0, :, token], value[token])
