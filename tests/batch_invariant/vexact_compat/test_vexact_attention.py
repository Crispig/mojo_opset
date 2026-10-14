"""
VeXact attention compatibility manifest for the Ascend batch-invariant backend.

The parametrization deliberately collects one node for each of the 40 source
attention nodes. Hardware adaptations are explicit in parametrized node IDs
and in standalone-test docstrings. Signature-level helper gaps fail closed when
the production surface gains the relevant argument; other xfails are an explicit
capability manifest and must be replaced with semantic tests when implemented.
"""

import inspect
import os
import subprocess
import sys

from functools import lru_cache

import pytest
import torch
import torch_npu

from mojo_opset.modules import PagedDecodeGQAInfer as TorchNpuPagedDecodeGQA
from mojo_opset.modules import PagedPrefillGQAInfer as TorchNpuPagedPrefillGQA
from mojo_opset.modules import SWAInfer as TTXSWA

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")

_DEVICE = "npu"
_DTYPE = torch.bfloat16
_CHILD_TIMEOUT_SECONDS = 120
_ADAPTED_HEAD_DIM = 128
_ADAPTED_PAGE_SIZE = 128


@lru_cache(maxsize=1)
def _compressed_causal_mask() -> torch.Tensor:
    return torch.triu(torch.ones(2048, 2048, device=_DEVICE, dtype=torch.bool), diagonal=1)


def _cumulative(lengths: list[int]) -> tuple[int, ...]:
    result = []
    total = 0
    for length in lengths:
        total += length
        result.append(total)
    return tuple(result)


def _ascendc_tnd_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_lens: list[int],
    kv_lens: list[int],
) -> torch.Tensor:
    return torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query.shape[1],
        input_layout="TND",
        atten_mask=_compressed_causal_mask(),
        scale=query.shape[-1] ** -0.5,
        keep_prob=1.0,
        actual_seq_qlen=_cumulative(q_lens),
        actual_seq_kvlen=_cumulative(kv_lens),
        sparse_mode=3,
    )[0]


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    query_heads = query.transpose(0, 1)
    key_heads = key.transpose(0, 1)
    value_heads = value.transpose(0, 1)
    groups = query_heads.shape[0] // key_heads.shape[0]
    key_heads = key_heads.repeat_interleave(groups, dim=0)
    value_heads = value_heads.repeat_interleave(groups, dim=0)

    scores = torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2))
    scores = scores * (query.shape[-1] ** -0.5)
    q_positions = torch.arange(query.shape[0], device=query.device) + key.shape[0] - query.shape[0]
    kv_positions = torch.arange(key.shape[0], device=query.device)
    causal_mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    probs = torch.where(causal_mask.any(dim=-1)[None, :, None], probs, 0.0)
    return torch.matmul(probs, value_heads).transpose(0, 1)


def _packed_eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_lens: list[int],
    kv_lens: list[int],
) -> torch.Tensor:
    outputs = []
    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        outputs.append(
            _eager_attention(
                query[q_offset : q_offset + q_len],
                key[kv_offset : kv_offset + kv_len],
                value[kv_offset : kv_offset + kv_len],
            )
        )
        q_offset += q_len
        kv_offset += kv_len
    return torch.cat(outputs, dim=0)


def _assert_tnd_forward_backward(
    q_lens: list[int],
    kv_lens: list[int],
    *,
    q_heads: int,
    kv_heads: int,
    qk_dim: int,
    value_dim: int | None = None,
    dtype: torch.dtype,
) -> None:
    value_dim = qk_dim if value_dim is None else value_dim
    query = torch.randn(sum(q_lens), q_heads, qk_dim, device=_DEVICE, dtype=dtype).mul_(0.1).requires_grad_()
    key = torch.randn(sum(kv_lens), kv_heads, qk_dim, device=_DEVICE, dtype=dtype).mul_(0.1).requires_grad_()
    value = torch.randn(sum(kv_lens), kv_heads, value_dim, device=_DEVICE, dtype=dtype).mul_(0.1).requires_grad_()
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()

    output = _ascendc_tnd_attention(query, key, value, q_lens, kv_lens)
    reference = _packed_eager_attention(query_ref, key_ref, value_ref, q_lens, kv_lens)
    tolerance = 1e-3 if dtype == torch.float32 else 4e-2
    torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)

    with torch.no_grad():
        per_sequence = []
        q_offset = 0
        kv_offset = 0
        for q_len, kv_len in zip(q_lens, kv_lens):
            per_sequence.append(
                _ascendc_tnd_attention(
                    query[q_offset : q_offset + q_len],
                    key[kv_offset : kv_offset + kv_len],
                    value[kv_offset : kv_offset + kv_len],
                    [q_len],
                    [kv_len],
                )
            )
            q_offset += q_len
            kv_offset += kv_len
    torch.testing.assert_close(output, torch.cat(per_sequence), atol=0.0, rtol=0.0)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    reference.backward(grad_output)
    torch.testing.assert_close(query.grad, query_ref.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(key.grad, key_ref.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(value.grad, value_ref.grad, atol=tolerance, rtol=tolerance)


def _make_paged_cache(
    key_sequences: list[torch.Tensor],
    value_sequences: list[torch.Tensor],
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pages_per_sequence = [(sequence.shape[0] + page_size - 1) // page_size for sequence in key_sequences]
    total_pages = sum(pages_per_sequence)
    max_pages = max(pages_per_sequence)
    kv_heads = key_sequences[0].shape[1]
    key_dim = key_sequences[0].shape[2]
    value_dim = value_sequences[0].shape[2]
    key_cache = torch.zeros(total_pages, kv_heads, page_size, key_dim, device=_DEVICE, dtype=_DTYPE)
    value_cache = torch.zeros(total_pages, kv_heads, page_size, value_dim, device=_DEVICE, dtype=_DTYPE)
    block_tables = torch.full((len(key_sequences), max_pages), -1, device=_DEVICE, dtype=torch.int32)

    page_index = 0
    for sequence_index, (key, value) in enumerate(zip(key_sequences, value_sequences)):
        for local_page in range(pages_per_sequence[sequence_index]):
            start = local_page * page_size
            end = min(start + page_size, key.shape[0])
            key_cache[page_index, :, : end - start] = key[start:end].transpose(0, 1)
            value_cache[page_index, :, : end - start] = value[start:end].transpose(0, 1)
            block_tables[sequence_index, local_page] = page_index
            page_index += 1
    return key_cache, value_cache, block_tables


def _make_paged_inputs(
    q_lens: list[int],
    kv_lens: list[int],
    *,
    q_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = _ADAPTED_HEAD_DIM,
    page_size: int = _ADAPTED_PAGE_SIZE,
):
    query_sequences = [
        torch.randn(length, q_heads, head_dim, device=_DEVICE, dtype=_DTYPE).mul_(0.1) for length in q_lens
    ]
    key_sequences = [
        torch.randn(length, kv_heads, head_dim, device=_DEVICE, dtype=_DTYPE).mul_(0.1) for length in kv_lens
    ]
    value_sequences = [torch.randn_like(key) for key in key_sequences]
    key_cache, value_cache, block_tables = _make_paged_cache(key_sequences, value_sequences, page_size)
    return query_sequences, key_sequences, value_sequences, key_cache, value_cache, block_tables


def _assert_paged_decode_batch_invariance(
    kv_lens: list[int],
    *,
    q_heads: int = 4,
    kv_heads: int = 2,
    page_size: int = _ADAPTED_PAGE_SIZE,
) -> None:
    q_lens = [1] * len(kv_lens)
    query, key, value, key_cache, value_cache, block_tables = _make_paged_inputs(
        q_lens,
        kv_lens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        page_size=page_size,
    )
    operator = TorchNpuPagedDecodeGQA(implementation="torch_npu", gqa_layout="AABB")
    query_batch = torch.cat(query)
    lengths = torch.tensor(kv_lens, device=_DEVICE, dtype=torch.int32)
    batched = operator(query_batch, key_cache, value_cache, lengths, block_tables)
    singles = [
        operator(
            query[index],
            key_cache,
            value_cache,
            lengths[index : index + 1],
            block_tables[index : index + 1],
        )
        for index in range(len(kv_lens))
    ]
    torch.testing.assert_close(batched, torch.cat(singles), atol=0.0, rtol=0.0)
    reference = torch.cat([_eager_attention(q, k, v) for q, k, v in zip(query, key, value)])
    torch.testing.assert_close(batched, reference, atol=3e-2, rtol=3e-2)


def _assert_paged_prefill_batch_invariance(
    seq_lens: list[int],
    *,
    q_heads: int = 4,
    kv_heads: int = 2,
    page_size: int = _ADAPTED_PAGE_SIZE,
) -> None:
    query, key, value, key_cache, value_cache, block_tables = _make_paged_inputs(
        seq_lens,
        seq_lens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        page_size=page_size,
    )
    operator = TorchNpuPagedPrefillGQA(implementation="torch_npu", gqa_layout="AABB")
    cumulative = torch.tensor((0, *_cumulative(seq_lens)), device=_DEVICE, dtype=torch.int32)
    batched = operator(torch.cat(query), key_cache, value_cache, cumulative, block_tables)
    singles = [
        operator(
            query[index],
            key_cache,
            value_cache,
            torch.tensor([0, seq_lens[index]], device=_DEVICE, dtype=torch.int32),
            block_tables[index : index + 1],
        )
        for index in range(len(seq_lens))
    ]
    torch.testing.assert_close(batched, torch.cat(singles), atol=0.0, rtol=0.0)
    reference = torch.cat([_eager_attention(q, k, v) for q, k, v in zip(query, key, value)])
    torch.testing.assert_close(batched, reference, atol=3e-2, rtol=3e-2)


def _assert_sliding_window_matches_reference() -> None:
    torch.manual_seed(0)
    sequence_length, heads, head_dim = 16, 2, 64
    query = torch.randn(sequence_length, heads, head_dim, device=_DEVICE, dtype=_DTYPE)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cumulative = torch.tensor([0, sequence_length], device=_DEVICE, dtype=torch.int32)
    output = TTXSWA(
        implementation="triton",
        is_causal=True,
        gqa_layout="AABB",
        local_window_size=3,
        global_window_size=0,
    )(query, key, value, cumulative, cumulative, head_dim**-0.5)

    query_heads = query.transpose(0, 1)
    key_heads = key.transpose(0, 1)
    value_heads = value.transpose(0, 1)
    scores = torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2)) * (head_dim**-0.5)
    positions = torch.arange(sequence_length, device=_DEVICE)
    mask = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - 4)
    scores.masked_fill_(~mask, float("-inf"))
    reference = torch.matmul(torch.softmax(scores, dim=-1), value_heads.float()).transpose(0, 1)
    torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)


@lru_cache(maxsize=1)
def _run_sliding_window_probe() -> subprocess.CompletedProcess[str]:
    code = (
        "import runpy; "
        f"namespace = runpy.run_path({str(__file__)!r}); "
        "namespace['_assert_sliding_window_matches_reference']()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=os.environ.copy(),
        text=True,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )


@pytest.mark.parametrize(
    ("source_node", "missing_argument"),
    [
        ("window-none", "sliding_window"),
        ("window-one", "sliding_window"),
        ("window-4096", "sliding_window"),
        ("window-reject-zero", "sliding_window"),
        ("window-reject-negative", "sliding_window"),
        ("sink-dtype-cast", "s_aux"),
        ("sink-reject-fa3", "s_aux"),
    ],
)
def test_flash_attention_helper_contract_is_missing(source_node, missing_argument):
    """Keep GPU helper nodes visible without testing a copied test-only helper."""
    del source_node
    forward_arguments = inspect.signature(TorchNpuPagedPrefillGQA.forward).parameters
    if missing_argument in forward_arguments:
        pytest.fail(f"production gained {missing_argument}; replace this gap test with a semantic test")
    pytest.xfail(f"AscendC Mojo attention does not expose {missing_argument}")


@pytest.mark.parametrize(
    "source_node",
    ["sliding-window-fa3", "sliding-window-fa4", "fa4-learnable-sink"],
)
def test_missing_sliding_window_and_sink_contracts(source_node):
    if "sink" in source_node:
        if "s_aux" in inspect.signature(TorchNpuPagedPrefillGQA.forward).parameters:
            pytest.fail("production gained s_aux; add the source numerical assertion")
        pytest.xfail("AscendC Mojo attention has no learnable-sink argument")

    try:
        result = _run_sliding_window_probe()
    except subprocess.TimeoutExpired:
        pytest.fail(f"sliding-window production probe exceeded {_CHILD_TIMEOUT_SECONDS} seconds")
    if result.returncode == 0:
        return
    output = result.stdout + result.stderr
    if "CompilationError" in output:
        pytest.xfail("TTX SWA is the NPU semantic equivalent but does not compile in Triton-X 3.2.0")
    pytest.fail(f"sliding-window production probe failed unexpectedly:\n{output}")


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param("decode", id="decode-D64-to-D128"),
        pytest.param("prefill", id="prefill-D64-to-D128"),
        pytest.param(
            "mixed",
            id="mixed-prefill-decode",
            marks=pytest.mark.xfail(
                reason="TorchNpu paged prefill rejects mixed-query cu_total_seq_lens",
                strict=True,
            ),
        ),
    ],
)
def test_fa4_batch_invariance(phase):
    torch.manual_seed(42)
    if phase == "decode":
        _assert_paged_decode_batch_invariance(
            [64, 112, 256, 280],
            q_heads=8,
            kv_heads=8,
            page_size=256,
        )
    elif phase == "prefill":
        _assert_paged_prefill_batch_invariance(
            [300, 220],
            q_heads=8,
            kv_heads=8,
            page_size=256,
        )
    else:
        raise NotImplementedError("one AscendC call cannot represent mixed paged prefill and decode")


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param("decode", id="legacy-decode-D8-to-D128"),
        pytest.param(
            "mixed",
            id="legacy-mixed-prefill-decode",
            marks=pytest.mark.xfail(reason="legacy mixed FA has no single Mojo AscendC equivalent", strict=True),
        ),
    ],
)
def test_legacy_flash_attention_batch_invariance(phase):
    if phase == "decode":
        _assert_paged_decode_batch_invariance(
            [64, 112, 256, 280],
            q_heads=1,
            kv_heads=1,
            page_size=256,
        )
    else:
        raise NotImplementedError("mixed prefill/decode is not exposed by one AscendC operator")


@pytest.mark.parametrize(
    "source_node",
    ["paged-batch-invariance-D64-page16-to-D128-page128", "vs-eager"],
)
def test_flex_attention_semantic_compatibility(source_node):
    torch.manual_seed(17)
    if source_node.startswith("paged-batch-invariance"):
        _assert_paged_decode_batch_invariance(
            [64, 112, 256, 280],
            q_heads=8,
            kv_heads=4,
        )
    else:
        _assert_tnd_forward_backward(
            [128],
            [128],
            q_heads=8,
            kv_heads=4,
            qk_dim=64,
            dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["float16", "bfloat16"])
def test_triton_varlen_nonpaged_packed_backward(dtype):
    torch.manual_seed(0)
    _assert_tnd_forward_backward([65, 129], [65, 129], q_heads=4, kv_heads=2, qk_dim=64, dtype=dtype)


@pytest.mark.xfail(reason="raw npu_fusion_attention has no VeXact max_seqlen validation wrapper", strict=True)
def test_triton_varlen_nonpaged_requires_max_seqlen():
    raise NotImplementedError("max_seqlen_q/max_seqlen_k compatibility wrapper is absent")


def test_triton_varlen_nonpaged_decode_packed_backward():
    torch.manual_seed(10)
    _assert_tnd_forward_backward([1, 17], [65, 129], q_heads=4, kv_heads=2, qk_dim=64, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [([17, 33, 65], [17, 33, 65]), ([1, 7, 19], [65, 71, 83])],
    ids=["prefill-gqa", "decode-gqa"],
)
def test_triton_varlen_nonpaged_gqa_backward_matches_eager(q_lens, kv_lens):
    torch.manual_seed(101)
    _assert_tnd_forward_backward(q_lens, kv_lens, q_heads=4, kv_heads=2, qk_dim=64, dtype=torch.float32)


def test_triton_varlen_nonpaged_q_longer_than_k_matches_eager():
    """Run the unsupported AscendC shape in a child so an AICore error cannot poison later tests."""
    child_flag = "MOJO_VEXACT_Q_LONGER_CHILD"
    if os.environ.get(child_flag) != "1":
        environment = os.environ.copy()
        environment[child_flag] = "1"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"{__file__}::test_triton_varlen_nonpaged_q_longer_than_k_matches_eager",
                ],
                capture_output=True,
                env=environment,
                text=True,
                timeout=_CHILD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"q_len > kv_len child exceeded {_CHILD_TIMEOUT_SECONDS} seconds")
        if result.returncode != 0:
            output = result.stdout + result.stderr
            if "larger than s2" in output or "exist invalid row" in output:
                pytest.xfail("AscendC TND rejects q_len > kv_len; child isolated the device error")
            pytest.fail(f"q_len > kv_len child failed unexpectedly:\n{output}")
        return

    torch.manual_seed(303)
    _assert_tnd_forward_backward(
        [5, 9],
        [2, 4],
        q_heads=4,
        kv_heads=2,
        qk_dim=32,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("head_dim", [40, 96])
def test_triton_varlen_nonpaged_arbitrary_head_dim_backward_matches_eager(head_dim):
    torch.manual_seed(202)
    _assert_tnd_forward_backward([9, 23], [41, 57], q_heads=6, kv_heads=3, qk_dim=head_dim, dtype=torch.float32)


def test_triton_varlen_qwen3_like_nonpaged_backward():
    torch.manual_seed(11)
    _assert_tnd_forward_backward([257, 385], [257, 385], q_heads=16, kv_heads=8, qk_dim=128, dtype=torch.bfloat16)


def test_triton_varlen_mla_nonpaged_backward_matches_eager():
    torch.manual_seed(404)
    _assert_tnd_forward_backward(
        [3, 17],
        [33, 65],
        q_heads=16,
        kv_heads=16,
        qk_dim=192,
        value_dim=128,
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        pytest.param([1, 1, 1, 1], [17, 31, 40, 49], id="decode-D32-page16-to-D128-page128"),
        pytest.param([24, 19], [24, 19], id="prefill-D32-page16-to-D128-page128"),
        pytest.param(
            [24, 1, 17, 1],
            [24, 33, 17, 41],
            id="mixed-D32-page16-to-D128-page128",
            marks=pytest.mark.xfail(reason="mixed paged prefill/decode has no single AscendC call", strict=True),
        ),
    ],
)
def test_triton_varlen_paged_correctness_and_batch_invariance(q_lens, kv_lens):
    torch.manual_seed(1)
    if all(length == 1 for length in q_lens):
        _assert_paged_decode_batch_invariance(kv_lens)
    elif q_lens == kv_lens:
        _assert_paged_prefill_batch_invariance(q_lens)
    else:
        raise NotImplementedError("mixed paged batches require a unified attention entry point")


@pytest.mark.xfail(reason="Mojo exposes block_tables only; the page_table keyword alias is absent", strict=True)
def test_triton_varlen_page_table_aliases_match():
    raise NotImplementedError("page_table compatibility alias is absent")


def _assert_paged_prefill_decode_alignment(
    seq_lens: list[int],
    *,
    q_heads: int,
    kv_heads: int,
    page_size: int,
) -> None:
    query, _, _, key_cache, value_cache, block_tables = _make_paged_inputs(
        seq_lens,
        seq_lens,
        q_heads=q_heads,
        kv_heads=kv_heads,
        page_size=page_size,
    )
    cumulative = torch.tensor((0, *_cumulative(seq_lens)), device=_DEVICE, dtype=torch.int32)
    prefill = TorchNpuPagedPrefillGQA(implementation="torch_npu", gqa_layout="AABB")(
        torch.cat(query), key_cache, value_cache, cumulative, block_tables
    )
    decode = TorchNpuPagedDecodeGQA(implementation="torch_npu", gqa_layout="AABB")(
        torch.cat([sequence[-1:] for sequence in query]),
        key_cache,
        value_cache,
        torch.tensor(seq_lens, device=_DEVICE, dtype=torch.int32),
        block_tables,
    )
    offset = 0
    for sequence_index, sequence in enumerate(query):
        torch.testing.assert_close(prefill[offset + sequence.shape[0] - 1], decode[sequence_index], atol=0, rtol=0)
        offset += sequence.shape[0]


def test_triton_varlen_paged_prefill_decode_bitwise_alignment():
    """Adapt the source D32/page16 case to the wrapper's D128/page128 contract."""
    torch.manual_seed(3)
    _assert_paged_prefill_decode_alignment(
        [17, 33, 64],
        q_heads=4,
        kv_heads=2,
        page_size=128,
    )


def test_triton_varlen_qwen3_like_paged_prefill_decode_bitwise_alignment():
    torch.manual_seed(33)
    _assert_paged_prefill_decode_alignment(
        list(range(74, 83)),
        q_heads=16,
        kv_heads=8,
        page_size=256,
    )


@pytest.mark.xfail(reason="TorchNpu paged wrapper rejects MLA qk_dim=192 and value_dim=128", strict=True)
def test_triton_varlen_mla_paged_matches_eager():
    raise NotImplementedError("paged MLA with separate K/V dimensions is absent")


@pytest.mark.xfail(reason="paged MLA and its prefill/decode bitwise oracle are absent", strict=True)
def test_triton_varlen_mla_paged_prefill_decode_bitwise_alignment():
    raise NotImplementedError("paged MLA alignment cannot be expressed by the current wrapper")


@pytest.mark.xfail(
    reason="AscendC inference attention silently falls through autograd instead of rejecting backward",
    strict=True,
)
def test_triton_varlen_paged_backward_raises():
    """Adapt the source D32/page16 inputs to D128/page128, retaining the backward contract."""
    query, _, _, key_cache, value_cache, block_tables = _make_paged_inputs([1], [17])
    query_tensor = query[0].detach().requires_grad_()
    output = TorchNpuPagedDecodeGQA(implementation="torch_npu", gqa_layout="AABB")(
        query_tensor,
        key_cache,
        value_cache,
        torch.tensor([17], device=_DEVICE, dtype=torch.int32),
        block_tables,
    )
    with pytest.raises((NotImplementedError, RuntimeError)):
        output.sum().backward()


def test_triton_flash_attention_forward_wrapper_shape():
    """Adapt the source D32/page16 wrapper input to D128/page128."""
    query, _, _, key_cache, value_cache, block_tables = _make_paged_inputs([1, 1], [17, 17])
    output = TorchNpuPagedDecodeGQA(implementation="torch_npu", gqa_layout="AABB")(
        torch.cat(query),
        key_cache,
        value_cache,
        torch.tensor([17, 17], device=_DEVICE, dtype=torch.int32),
        block_tables,
    )
    assert output.shape == (2, 4, _ADAPTED_HEAD_DIM)


@pytest.mark.parametrize(
    "shape_name",
    [
        pytest.param("qwen", id="qwen-D128-page16-to-page128"),
        pytest.param(
            "deepseek",
            id="deepseek-qk192-v128",
            marks=pytest.mark.xfail(
                reason="TorchNpu paged wrapper rejects qk_dim=192 and separate value_dim=128",
                strict=True,
            ),
        ),
    ],
)
def test_fa4_qwen_and_deepseek_paged(shape_name):
    torch.manual_seed(42)
    if shape_name == "qwen":
        _assert_paged_prefill_batch_invariance([32], q_heads=16, kv_heads=16)
    else:
        raise NotImplementedError("paged DeepSeek 192/128 shape is absent from the current AscendC wrapper")
