import pytest
import torch

from mojo_opset.functions.kv_cache import store_paged_kv_cache
from mojo_opset.kernels.npu_a2_batch_invariant.kv_cache import _store_cache_slots

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _assert_bits(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _reference(key, value, kc, vc, blocks, cumulative, contexts):
    result_k = kc.clone()
    result_v = vc.clone()
    boundaries = list(range(key.shape[0] + 1)) if cumulative is None else cumulative.cpu().tolist()
    for batch, (first, last) in enumerate(zip(boundaries, boundaries[1:])):
        context = int(contexts[batch].item())
        if context < 0:
            continue
        for token in range(first, last):
            logical = context + token - first
            page = int(blocks[batch, logical // 256].item())
            if page >= 0:
                result_k[page, :, logical % 256] = key[token]
                result_v[page, :, logical % 256] = value[token]
    return result_k, result_v


@pytest.mark.parametrize("key_dim,value_dim", [(128, 128), (128, 64), (5, 7)])
@pytest.mark.parametrize("tokens", [0, 17, 259])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tiled_kv_copy_preserves_partial_tiles_and_distinct_head_dimensions(key_dim, value_dim, tokens, dtype):
    key = torch.randn(tokens, 3, key_dim, device="npu", dtype=dtype)
    value = torch.randn(tokens, 3, value_dim, device="npu", dtype=key.dtype)
    kc = torch.zeros(3, 3, 256, key_dim, device="npu", dtype=key.dtype)
    vc = torch.zeros(3, 3, 256, value_dim, device="npu", dtype=key.dtype)
    cu = torch.tensor([0, tokens], device="npu", dtype=torch.int32)
    contexts = torch.tensor([253], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[2, 0, 1]], device="npu", dtype=torch.int32)
    expected = _reference(key, value, kc, vc, blocks, cu, contexts)
    store_paged_kv_cache(key, value, kc, vc, blocks, cu, contexts, implementation="batch_invariant")
    _assert_bits(kc, expected[0])
    _assert_bits(vc, expected[1])


def test_tiled_kv_copy_keeps_nan_payloads_subnormals_and_signed_zero():
    pattern = torch.tensor([-32768, 0, 1, 32640, -128, 32705, -63, 16256], device="npu", dtype=torch.int16)
    key = pattern.repeat(17 * 3 * 128 // pattern.numel()).view(torch.bfloat16).reshape(17, 3, 128)
    value = pattern.repeat(17 * 3 * 64 // pattern.numel()).view(torch.bfloat16).reshape(17, 3, 64)
    kc = torch.zeros(3, 3, 256, 128, device="npu", dtype=key.dtype)
    vc = torch.zeros(3, 3, 256, 64, device="npu", dtype=key.dtype)
    expected_k = torch.zeros_like(kc)
    expected_v = torch.zeros_like(vc)
    slots = torch.tensor([765, 766, 767, *range(14)], device="npu", dtype=torch.int64)
    _store_cache_slots(key, value, expected_k, expected_v, slots, layout="blocks_heads_page_dim", validate_slots=False)
    cu = torch.tensor([0, 17], device="npu", dtype=torch.int32)
    contexts = torch.tensor([253], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[2, 0, 1]], device="npu", dtype=torch.int32)
    store_paged_kv_cache(key, value, kc, vc, blocks, cu, contexts, implementation="batch_invariant")
    _assert_bits(kc, expected_k)
    _assert_bits(vc, expected_v)


@pytest.mark.parametrize("decode", [False, True])
def test_kv_store_graph_updates_page_table_lengths_and_source(decode):
    tokens = 2 if decode else 7
    key = torch.randn(tokens, 2, 128, device="npu", dtype=torch.bfloat16)
    value = torch.randn(tokens, 2, 64, device="npu", dtype=key.dtype)
    kc = torch.zeros(6, 2, 256, 128, device="npu", dtype=key.dtype)
    vc = torch.zeros(6, 2, 256, 64, device="npu", dtype=key.dtype)
    blocks = torch.tensor([[0, 1], [2, 3]], device="npu", dtype=torch.int32)
    cu = None if decode else torch.tensor([0, 3, 7], device="npu", dtype=torch.int32)
    contexts = torch.tensor([254, 255], device="npu", dtype=torch.int32)

    def run():
        return store_paged_kv_cache(key, value, kc, vc, blocks, cu, contexts, implementation="batch_invariant")

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    for context, pages in [([254, 255], [[0, 1], [2, 3]]), ([255, -1], [[4, 5], [1, 0]])]:
        contexts.copy_(torch.tensor(context, device="npu", dtype=torch.int32))
        blocks.copy_(torch.tensor(pages, device="npu", dtype=torch.int32))
        key.add_(0.125)
        value.sub_(0.25)
        kc.zero_()
        vc.zero_()
        expected = _reference(key, value, kc, vc, blocks, cu, contexts)
        graph.replay()
        torch.npu.synchronize()
        assert result[0] is kc and result[1] is vc
        _assert_bits(kc, expected[0])
        _assert_bits(vc, expected[1])


def test_kv_store_graph_replays_precomputed_chunks():
    key = torch.randn(7, 2, 128, device="npu", dtype=torch.bfloat16)
    value = torch.randn(7, 2, 64, device="npu", dtype=key.dtype)
    kc = torch.zeros(5, 2, 256, 128, device="npu", dtype=key.dtype)
    vc = torch.zeros(5, 2, 256, 64, device="npu", dtype=key.dtype)
    chunks = torch.tensor([[0, 0, 254, 2], [2, 1, 0, 5]], device="npu", dtype=torch.int32)

    def run():
        return store_paged_kv_cache(key, value, kc, vc, chunk_metadata=chunks, implementation="batch_invariant")

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for rows in [[[0, 0, 254, 2], [2, 1, 0, 5]], [[0, 4, 253, 3], [3, 2, 0, 4]]]:
        chunks.copy_(torch.tensor(rows, device="npu", dtype=torch.int32))
        kc.zero_()
        vc.zero_()
        expected_k = torch.zeros_like(kc)
        expected_v = torch.zeros_like(vc)
        for first, block, offset, count in rows:
            expected_k[block, :, offset : offset + count] = key[first : first + count].transpose(0, 1)
            expected_v[block, :, offset : offset + count] = value[first : first + count].transpose(0, 1)
        graph.replay()
        torch.npu.synchronize()
        _assert_bits(kc, expected_k)
        _assert_bits(vc, expected_v)
