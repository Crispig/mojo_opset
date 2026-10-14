from functools import partial

import pytest
import torch

from mojo_opset.functions.gemm import batch_invariant_linear as _linear
from mojo_opset.functions.gemm import batch_invariant_matmul
from mojo_opset.functions.logprobs import logprobs_from_logits as _logprobs_from_logits
from mojo_opset.functions.softmax import softmax as _softmax

pytest.importorskip("triton")

linear = partial(_linear, implementation="batch_invariant")
matmul_persistent = partial(batch_invariant_matmul, implementation="batch_invariant")
softmax = partial(_softmax, implementation="batch_invariant")
logprobs_from_logits = partial(_logprobs_from_logits, implementation="batch_invariant")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="VeXact compatibility tests require an Ascend NPU",
)


def test_batch_invariance() -> None:
    batch, dim = 2048, 4096
    a = torch.linspace(-100, 100, batch * dim, device="npu").reshape(batch, dim)
    b = torch.linspace(-100, 100, dim * dim, device="npu").reshape(dim, dim)

    single_token = matmul_persistent(a[1:2], b)
    full_batch = matmul_persistent(a, b)[1:2]

    assert (single_token - full_batch).abs().max().item() == 0


def test_softmax_simple_batch_invariance() -> None:
    batch = 4
    heads = 8
    sequence_lengths = [32, 48, 24, 56]
    max_sequence_length = max(sequence_lengths)

    masks = torch.full(
        (batch, 1, max_sequence_length, max_sequence_length),
        float("-inf"),
        dtype=torch.bfloat16,
        device="npu",
    )
    logits = torch.zeros(
        batch,
        heads,
        max_sequence_length,
        max_sequence_length,
        dtype=torch.bfloat16,
        device="npu",
    )

    for index, sequence_length in enumerate(sequence_lengths):
        logits[index, :, :sequence_length, :sequence_length] = torch.linspace(
            -10,
            10,
            heads * sequence_length * sequence_length,
            dtype=torch.bfloat16,
            device="npu",
        ).reshape(heads, sequence_length, sequence_length)
        causal_mask = torch.tril(
            torch.ones(
                1,
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device="npu",
            )
        )
        masks[index, :, :sequence_length, :sequence_length] = torch.where(
            causal_mask,
            0.0,
            float("-inf"),
        )

    masked_logits = logits + masks
    first_sequence_length = sequence_lengths[0]
    single_batch = softmax(
        masked_logits[:1, :, :first_sequence_length, :first_sequence_length],
        dim=-1,
    )
    full_batch = softmax(masked_logits, dim=-1)[
        :1,
        :,
        :first_sequence_length,
        :first_sequence_length,
    ]

    assert (single_batch - full_batch).abs().max().item() < 1e-6


@pytest.mark.parametrize(
    ("num_tokens", "vocab_size"),
    [(256, 1024), (513, 1536), (256, 151936)],
)
def test_selected_token_logprob_batch_invariance(
    num_tokens: int,
    vocab_size: int,
) -> None:
    torch.manual_seed(123)
    logits = torch.randn(
        num_tokens,
        vocab_size,
        device="npu",
        dtype=torch.float32,
    )
    labels = torch.randint(
        0,
        vocab_size,
        (num_tokens,),
        device="npu",
        dtype=torch.long,
    )

    full_losses = -logprobs_from_logits(logits, labels, inplace_backward=False)
    permutation = torch.randperm(num_tokens, device=logits.device)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(
        num_tokens,
        device=logits.device,
    )
    permuted_losses = -logprobs_from_logits(
        logits[permutation].contiguous(),
        labels[permutation].contiguous(),
        inplace_backward=False,
    )

    torch.testing.assert_close(
        full_losses,
        permuted_losses[inverse_permutation],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("batch", "dim"),
    [(32, 256), (64, 512), (128, 1024)],
)
def test_mm_batch_invariance(
    dtype: torch.dtype,
    batch: int,
    dim: int,
) -> None:
    # Keep fixture generation off-device so the test isolates GEMM from the
    # torch_npu BF16 linspace generator, which faults for the 512**2 fixture.
    a = (
        torch.linspace(
            -1.0,
            1.0,
            batch * dim,
            dtype=dtype,
        )
        .reshape(batch, dim)
        .to("npu")
    )
    b = (
        torch.linspace(
            -1.0,
            1.0,
            dim * dim,
            dtype=dtype,
        )
        .reshape(dim, dim)
        .to("npu")
    )

    single_token = matmul_persistent(a[:1], b)
    full_batch = matmul_persistent(a, b)[:1]

    torch.testing.assert_close(single_token, full_batch, atol=0.0, rtol=0.0)


def _build_fused_linear_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    hidden = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        device="npu",
        dtype=dtype,
    )
    weight = torch.randn(
        vocab_size,
        hidden_size,
        device="npu",
        dtype=dtype,
    )
    labels = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device="npu",
        dtype=torch.long,
    )
    return hidden, weight, labels


def _chunked_selected_token_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    hidden_size = hidden.shape[-1]
    flat_hidden = hidden.reshape(-1, hidden_size)
    flat_labels = labels.reshape(-1)
    chunks: list[torch.Tensor] = []
    for start in range(0, flat_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, flat_hidden.shape[0])
        logits = linear(flat_hidden[start:end], weight).float()
        chunks.append(
            logprobs_from_logits(
                logits,
                flat_labels[start:end],
                inplace_backward=False,
            )
        )
    return torch.cat(chunks).reshape_as(labels)


@pytest.mark.parametrize(
    (
        "batch_size",
        "sequence_length",
        "hidden_size",
        "vocab_size",
        "chunk_size",
    ),
    [(16, 2048, 1024, 4096, 64), (1, 33, 96, 384, 7)],
)
def test_fused_linear_for_ppo_batch_invariance_forward(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    chunk_size: int,
) -> None:
    hidden, weight, labels = _build_fused_linear_inputs(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seed=11,
    )

    full_log_probs = _chunked_selected_token_logprobs(
        hidden,
        weight,
        labels,
        chunk_size,
    )
    total_tokens = batch_size * sequence_length
    permutation = torch.randperm(total_tokens, device=hidden.device)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(
        total_tokens,
        device=hidden.device,
    )
    hidden_permuted = hidden.reshape(total_tokens, hidden_size)[permutation].contiguous().reshape_as(hidden)
    labels_permuted = labels.reshape(total_tokens)[permutation].contiguous().reshape_as(labels)
    permuted_log_probs = _chunked_selected_token_logprobs(
        hidden_permuted,
        weight,
        labels_permuted,
        chunk_size,
    )

    torch.testing.assert_close(
        full_log_probs.reshape(-1),
        permuted_log_probs.reshape(-1)[inverse_permutation],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    (
        "batch_size",
        "sequence_length",
        "hidden_size",
        "vocab_size",
        "chunk_size",
    ),
    [(16, 2048, 1024, 4096, 64), (1, 33, 96, 384, 7)],
)
def test_fused_linear_for_ppo_log_probs_float32_in_bfloat16_model(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    chunk_size: int,
) -> None:
    hidden, weight, labels = _build_fused_linear_inputs(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seed=42,
        dtype=torch.bfloat16,
    )
    rolled_labels = torch.roll(labels, shifts=-1, dims=-1)

    fused_log_probs = _chunked_selected_token_logprobs(
        hidden,
        weight,
        rolled_labels,
        chunk_size,
    )
    logits = linear(hidden, weight).float()
    reference_log_probs = logprobs_from_logits(
        logits,
        rolled_labels,
        inplace_backward=False,
    )

    assert fused_log_probs.dtype == torch.float32
    torch.testing.assert_close(
        fused_log_probs,
        reference_log_probs,
        rtol=0.0,
        atol=0.0,
    )
