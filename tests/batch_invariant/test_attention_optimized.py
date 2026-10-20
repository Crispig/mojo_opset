import importlib.util
import os

from pathlib import Path

import pytest
import torch

from mojo_opset.kernels.npu_a2_batch_invariant import attention

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.fixture(scope="module")
def previous_attention():
    root = os.environ.get("MOJO_BI_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set MOJO_BI_REFERENCE_ROOT to a frozen pre-optimization package")
    path = Path(root) / "mojo_opset/kernels/npu_a2_batch_invariant/attention.py"
    spec = importlib.util.spec_from_file_location("mojo_opset.kernels.npu_a2_batch_invariant._previous_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _cache(key, value, lengths):
    page_counts = [(length + 255) // 256 for length in lengths]
    page_count = sum(page_counts)
    cache_shape = (page_count, key.shape[1], 256, 128)
    kc = torch.randn(cache_shape, device="npu", dtype=key.dtype)
    vc = torch.randn_like(kc)
    blocks = torch.full((len(lengths), max(page_counts)), -1, device="npu", dtype=torch.int32)
    physical = torch.randperm(page_count).tolist()
    token_start = 0
    page_start = 0
    for batch, length in enumerate(lengths):
        for logical in range(page_counts[batch]):
            page = physical[page_start + logical]
            first = logical * 256
            count = min(256, length - first)
            kc[page, :, :count] = key[token_start + first : token_start + first + count].transpose(0, 1)
            vc[page, :, :count] = value[token_start + first : token_start + first + count].transpose(0, 1)
            blocks[batch, logical] = page
        token_start += length
        page_start += page_counts[batch]
    return kc, vc, blocks


@pytest.mark.parametrize("length", [63, 64, 65, 255, 256, 257, 1025])
def test_block_page_lookup_preserves_previous_attention_bits(previous_attention, length):
    torch.manual_seed(850 + length)
    q = torch.randn(length, 8, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(length, 2, 128, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, length], device="npu", dtype=torch.int32)
    kc, vc, blocks = _cache(k, v, [length])
    expected = previous_attention.attention_packed(q, k, v, cu, cu)
    _assert_bits(attention.attention_packed(q, k, v, cu, cu), expected)
    seq = cu[1:]
    _assert_bits(attention.attention_paged(q, kc, vc, cu, seq, blocks), expected)
    one = torch.tensor([0, 1], device="npu", dtype=torch.int32)
    _assert_bits(attention.attention_paged(q[-1:], kc, vc, one, seq, blocks), expected[-1:])


@pytest.mark.parametrize("scale", [0.125, 128.0, 2048.0])
def test_block_page_lookup_keeps_previous_finite_mask_arithmetic(previous_attention, scale):
    torch.manual_seed(703)
    q = torch.randn(65, 8, 128, device="npu", dtype=torch.bfloat16) * scale
    k = torch.randn(65, 2, 128, device="npu", dtype=q.dtype) * scale
    v = torch.randn_like(k)
    cu = torch.tensor([0, 65], device="npu", dtype=torch.int32)
    expected = previous_attention.attention_packed(q, k, v, cu, cu)
    _assert_bits(attention.attention_packed(q, k, v, cu, cu), expected)
    kc, vc, blocks = _cache(k, v, [65])
    _assert_bits(attention.attention_paged(q, kc, vc, cu, cu[1:], blocks), expected)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 129, 255, 257])
def test_dma_tail_keeps_dummy_qk_columns_zero(previous_attention, length):
    torch.manual_seed(703)
    q = torch.randn(length, 8, 128, device="npu", dtype=torch.bfloat16) * 2048
    k = torch.randn(length, 2, 128, device="npu", dtype=q.dtype) * 2048
    v = torch.randn_like(k)
    cu = torch.tensor([0, length], device="npu", dtype=torch.int32)
    expected = previous_attention.attention_packed(q, k, v, cu, cu)
    _assert_bits(attention.attention_packed(q, k, v, cu, cu), expected)
    kc, vc, blocks = _cache(k, v, [length])
    _assert_bits(attention.attention_paged(q, kc, vc, cu, cu[1:], blocks), expected)


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_dma_does_not_read_unused_page_values(previous_attention, poison):
    torch.manual_seed(703)
    q = torch.randn(65, 8, 128, device="npu", dtype=torch.bfloat16) * 2048
    k = torch.randn(65, 2, 128, device="npu", dtype=q.dtype) * 2048
    v = torch.randn_like(k)
    cu = torch.tensor([0, 65], device="npu", dtype=torch.int32)
    expected = previous_attention.attention_packed(q, k, v, cu, cu)
    kc, vc, blocks = _cache(k, v, [65])
    kc[:, :, 65:] = poison
    vc[:, :, 65:] = poison
    _assert_bits(attention.attention_paged(q, kc, vc, cu, cu[1:], blocks), expected)
    one = torch.tensor([0, 1], device="npu", dtype=torch.int32)
    _assert_bits(attention.attention_paged(q[-1:], kc, vc, one, cu[1:], blocks, grouped_decode=True), expected[-1:])


def test_attention_preserves_mixed_batch_and_chunked_query_bits():
    torch.manual_seed(29)
    lengths = [67, 259]
    q = torch.randn(sum(lengths), 8, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(sum(lengths), 2, 128, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, 67, 326], device="npu", dtype=torch.int32)
    seq = torch.tensor(lengths, device="npu", dtype=torch.int32)
    kc, vc, blocks = _cache(k, v, lengths)
    expected = attention.attention_packed(q, k, v, cu, cu)
    _assert_bits(attention.attention_paged(q, kc, vc, cu, seq, blocks), expected)
    suffix = torch.cat((q[64:67], q[-3:]))
    suffix_cu = torch.tensor([0, 3, 6], device="npu", dtype=torch.int32)
    actual = attention.attention_paged(suffix, kc, vc, suffix_cu, seq, blocks)
    _assert_bits(actual, torch.cat((expected[64:67], expected[-3:])))


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 32])
def test_grouped_decode_keeps_all_absolute_row_offsets(previous_attention, groups, batch):
    torch.manual_seed(983 + groups + batch)
    q = torch.randn(batch, 8 * groups, 128, device="npu", dtype=torch.bfloat16)
    kc = torch.randn(batch * 2, 8, 256, 128, device="npu", dtype=q.dtype)
    vc = torch.randn_like(kc)
    cu = torch.arange(batch + 1, device="npu", dtype=torch.int32)
    blocks = torch.randperm(batch * 2, device="npu", dtype=torch.int32).reshape(batch, 2)
    for offset in range(64):
        lengths = 257 + offset + torch.arange(batch, device="npu", dtype=torch.int32)
        expected = previous_attention.attention_paged(q, kc, vc, cu, lengths, blocks)
        _assert_bits(attention.attention_paged(q, kc, vc, cu, lengths, blocks, grouped_decode=True), expected)


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16])
def test_grouped_decode_keeps_extreme_finite_inputs(previous_attention, groups):
    torch.manual_seed(739)
    q = torch.randn(2, 8 * groups, 128, device="npu", dtype=torch.bfloat16) * 2048
    kc = torch.randn(4, 8, 256, 128, device="npu", dtype=q.dtype) * 2048
    vc = torch.randn_like(kc)
    cu = torch.tensor([0, 1, 2], device="npu", dtype=torch.int32)
    lengths = torch.tensor([257, 319], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[2, 0], [3, 1]], device="npu", dtype=torch.int32)
    expected = previous_attention.attention_paged(q, kc, vc, cu, lengths, blocks)
    _assert_bits(attention.attention_paged(q, kc, vc, cu, lengths, blocks, grouped_decode=True), expected)


@pytest.mark.parametrize("groups", [17, 32, 64])
def test_grouped_decode_preserves_large_head_groups(previous_attention, groups):
    torch.manual_seed(740 + groups)
    q = torch.randn(2, groups, 128, device="npu", dtype=torch.bfloat16) * 2048
    kc = torch.randn(4, 1, 256, 128, device="npu", dtype=q.dtype) * 2048
    vc = torch.randn_like(kc)
    cu = torch.tensor([0, 1, 2], device="npu", dtype=torch.int32)
    lengths = torch.tensor([65, 257], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[2, 0], [3, 1]], device="npu", dtype=torch.int32)
    expected = previous_attention.attention_paged(q, kc, vc, cu, lengths, blocks)
    _assert_bits(attention.attention_paged(q, kc, vc, cu, lengths, blocks, grouped_decode=True), expected)


def test_paged_attention_graph_replays_updated_lengths_and_pages(previous_attention):
    torch.manual_seed(55)
    q = torch.randn(2, 8, 128, device="npu", dtype=torch.bfloat16) * 2048
    kc = torch.randn(8, 2, 256, 128, device="npu", dtype=q.dtype) * 2048
    vc = torch.randn_like(kc)
    cu = torch.tensor([0, 1, 2], device="npu", dtype=torch.int32)
    seq = torch.tensor([65, 257], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[0, 1, 2], [3, 4, 5]], device="npu", dtype=torch.int32)
    for _ in range(3):
        attention.attention_paged(q, kc, vc, cu, seq, blocks, grouped_decode=True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = attention.attention_paged(q, kc, vc, cu, seq, blocks, grouped_decode=True)
    for lengths, pages in [([65, 257], [[0, 1, 2], [3, 4, 5]]), ([128, 513], [[6, 2, 1], [4, 7, 0]])]:
        seq.copy_(torch.tensor(lengths, device="npu", dtype=torch.int32))
        blocks.copy_(torch.tensor(pages, device="npu", dtype=torch.int32))
        expected = previous_attention.attention_paged(q, kc, vc, cu, seq, blocks, grouped_decode=True)
        graph.replay()
        torch.npu.synchronize()
        _assert_bits(captured, expected)


def test_paged_prefill_graph_preserves_empty_padding_rows():
    torch.manual_seed(56)
    q = torch.randn(2, 8, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 128, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    kc, vc, _ = _cache(k, v, [2])
    packed_cu = torch.tensor([0, 2], device="npu", dtype=torch.int32)
    expected = attention.attention_packed(q, k, v, packed_cu, packed_cu)
    cu = torch.tensor([0, 2, 2], device="npu", dtype=torch.int32)
    seq = torch.tensor([2, 0], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[0], [-1]], device="npu", dtype=torch.int32)
    for _ in range(3):
        attention.attention_paged(q, kc, vc, cu, seq, blocks)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = attention.attention_paged(q, kc, vc, cu, seq, blocks)
    for boundaries, lengths, pages in [([0, 2, 2], [2, 0], [[0], [-1]]), ([0, 0, 2], [0, 2], [[-1], [0]])]:
        cu.copy_(torch.tensor(boundaries, device="npu", dtype=torch.int32))
        seq.copy_(torch.tensor(lengths, device="npu", dtype=torch.int32))
        blocks.copy_(torch.tensor(pages, device="npu", dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        _assert_bits(captured, expected)


def test_grouped_decode_does_not_write_empty_padding_rows():
    q = torch.randn(2, 8, 128, device="npu", dtype=torch.bfloat16)
    kc = torch.randn(1, 2, 256, 128, device="npu", dtype=q.dtype)
    vc = torch.randn_like(kc)
    cu = torch.tensor([0, 1, 2], device="npu", dtype=torch.int32)
    seq = torch.tensor([65, 0], device="npu", dtype=torch.int32)
    blocks = torch.tensor([[0], [-1]], device="npu", dtype=torch.int32)
    expected = attention.attention_paged(q[:1], kc, vc, cu[:2], seq[:1], blocks[:1], grouped_decode=True)
    for _ in range(3):
        attention.attention_paged(q, kc, vc, cu, seq, blocks, grouped_decode=True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = attention.attention_paged(q, kc, vc, cu, seq, blocks, grouped_decode=True)
    captured[1].fill_(3)
    graph.replay()
    torch.npu.synchronize()
    _assert_bits(captured[:1], expected)
    assert torch.equal(captured[1], torch.full_like(captured[1], 3))
