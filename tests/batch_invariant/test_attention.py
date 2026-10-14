import os

from functools import lru_cache

import pytest
import torch

from mojo_opset.functions.attention import flash_attention_infer
from mojo_opset.functions.paged_attention import paged_prefill_gqa_infer

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


os.environ.setdefault("MOJO_ASCEND_ARCH", "a2")
os.environ.setdefault("MOJO_DETERMINISTIC", "1")

_NPU_AVAILABLE = torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()
pytestmark = pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is required")

_DEVICE = torch.device("npu") if _NPU_AVAILABLE else torch.device("cpu")
_DTYPE = torch.bfloat16
_HEAD_DIM = 128
_PAGE_SIZE = 256
_Q_HEADS = 16
_KV_HEADS = 8


@lru_cache(maxsize=1)
def _attention_api():
    def packed(q, k, v, cu_q, cu_k, softmax_scale=None):
        max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
        max_k = int((cu_k[1:] - cu_k[:-1]).max().item())
        return flash_attention_infer(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_q,
            max_k,
            softmax_scale=softmax_scale,
            causal=True,
            implementation="batch_invariant",
        )

    def paged(q, kc, vc, cu_q, seq, blocks, softmax_scale=None):
        cumulative = torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32)))
        return paged_prefill_gqa_infer(
            q,
            kc,
            vc,
            cu_q,
            blocks,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cumulative,
            implementation="batch_invariant",
        )

    return packed, paged


def _make_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pages = (key.shape[0] + _PAGE_SIZE - 1) // _PAGE_SIZE
    cache_shape = (num_pages, _KV_HEADS, _PAGE_SIZE, _HEAD_DIM)
    key_cache = torch.zeros(cache_shape, device=_DEVICE, dtype=_DTYPE)
    value_cache = torch.zeros_like(key_cache)
    for page in range(num_pages):
        start = page * _PAGE_SIZE
        end = min(start + _PAGE_SIZE, key.shape[0])
        key_cache[page, :, : end - start] = key[start:end].transpose(0, 1)
        value_cache[page, :, : end - start] = value[start:end].transpose(0, 1)
    block_tables = torch.arange(num_pages, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    return key_cache, value_cache, block_tables


def _run_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    total_seq_len: int,
) -> torch.Tensor:
    _, attention_paged = _attention_api()
    return attention_paged(
        query,
        key_cache,
        value_cache,
        torch.tensor([0, query.shape[0]], device=_DEVICE, dtype=torch.int32),
        torch.tensor([total_seq_len], device=_DEVICE, dtype=torch.int32),
        block_tables,
        softmax_scale=_HEAD_DIM**-0.5,
    )


@pytest.mark.parametrize(
    ("seq_len", "prompt_len"),
    [(82, 64), (258, 255)],
)
def test_qwen3_packed_full_paged_and_prefill_decode_match_bitwise(
    seq_len: int,
    prompt_len: int,
):
    """Training, full prefill, and incremental rollout must share one result."""

    torch.manual_seed(33 + seq_len)
    query = torch.randn(seq_len, _Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE) * 0.1
    key = torch.randn(seq_len, _KV_HEADS, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE) * 0.1
    value = torch.randn_like(key)
    cumulative = torch.tensor([0, seq_len], device=_DEVICE, dtype=torch.int32)
    key_cache, value_cache, block_tables = _make_paged_cache(key, value)
    attention_packed, _ = _attention_api()

    packed = attention_packed(
        query,
        key,
        value,
        cumulative,
        cumulative,
        softmax_scale=_HEAD_DIM**-0.5,
    )
    full_paged = _run_paged(
        query,
        key_cache,
        value_cache,
        block_tables,
        total_seq_len=seq_len,
    )

    staged_outputs = [
        _run_paged(
            query[:prompt_len],
            key_cache,
            value_cache,
            block_tables,
            total_seq_len=prompt_len,
        )
    ]
    for position in range(prompt_len, seq_len):
        staged_outputs.append(
            _run_paged(
                query[position : position + 1],
                key_cache,
                value_cache,
                block_tables,
                total_seq_len=position + 1,
            )
        )
    staged = torch.cat(staged_outputs, dim=0)

    assert packed.shape == full_paged.shape == staged.shape == query.shape
    assert torch.equal(packed, full_paged)
    assert torch.equal(full_paged, staged)


def test_qwen3_attention_is_repeatable_bitwise():
    torch.manual_seed(91)
    seq_len = 82
    query = torch.randn(seq_len, _Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE) * 0.1
    key = torch.randn(seq_len, _KV_HEADS, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE) * 0.1
    value = torch.randn_like(key)
    cumulative = torch.tensor([0, seq_len], device=_DEVICE, dtype=torch.int32)
    attention_packed, _ = _attention_api()

    first = attention_packed(
        query,
        key,
        value,
        cumulative,
        cumulative,
        softmax_scale=_HEAD_DIM**-0.5,
    )
    second = attention_packed(
        query,
        key,
        value,
        cumulative,
        cumulative,
        softmax_scale=_HEAD_DIM**-0.5,
    )

    assert torch.equal(first, second)


def test_attention_rejects_noncontiguous_length_metadata():
    query = torch.zeros(4, _Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE)
    key = torch.zeros(4, _KV_HEADS, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE)
    value = torch.zeros_like(key)
    cumulative = torch.tensor([0, -1, 2, -1, 4], device=_DEVICE, dtype=torch.int32)[::2]
    attention_packed, _ = _attention_api()

    with pytest.raises(ValueError, match="cu_q_lens must be contiguous"):
        attention_packed(query, key, value, cumulative, cumulative)
