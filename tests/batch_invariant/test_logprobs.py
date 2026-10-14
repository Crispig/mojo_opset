from functools import partial

import pytest
import torch

from mojo_opset.functions.logprobs import logprobs_from_logits as _logprobs_from_logits

logprobs_from_logits = partial(_logprobs_from_logits, implementation="batch_invariant")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_logprobs_matches_fp32_reference(dtype):
    torch.manual_seed(101)
    logits = torch.randn(7, 1031, device="npu", dtype=dtype, requires_grad=True)
    labels = torch.tensor([0, 3, 17, 1024, 79, 512, 1030], device="npu")
    grad_output = torch.randn(7, device="npu", dtype=torch.float32)

    actual = logprobs_from_logits(logits, labels, inplace_backward=True)
    (actual_grad,) = torch.autograd.grad(actual, logits, grad_outputs=grad_output)

    reference_logits = logits.detach().clone().float().requires_grad_(True)
    reference = torch.log_softmax(reference_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (reference_grad,) = torch.autograd.grad(reference, reference_logits, grad_outputs=grad_output)

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    grad_atol = {
        torch.float16: 2e-5,
        torch.bfloat16: 4e-3,
        torch.float32: 3e-4,
    }[dtype]
    torch.testing.assert_close(
        actual_grad,
        reference_grad.to(dtype),
        atol=grad_atol,
        rtol=3e-4,
    )


def _run_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_leaf = logits.detach().clone().requires_grad_(True)
    output = logprobs_from_logits(logits_leaf, labels)
    (grad_logits,) = torch.autograd.grad(output, logits_leaf, grad_outputs=grad_output)
    return output.detach(), grad_logits.detach()


def test_logprobs_forward_and_backward_are_slice_and_permutation_invariant():
    torch.manual_seed(103)
    logits = torch.randn(59, 2053, device="npu", dtype=torch.bfloat16)
    labels = torch.randint(0, logits.shape[-1], (59,), device="npu")
    grad_output = torch.randn(59, device="npu", dtype=torch.float32)

    full_output, full_grad = _run_logprobs(logits, labels, grad_output)
    sliced_output, sliced_grad = _run_logprobs(logits[37:38], labels[37:38], grad_output[37:38])
    assert torch.equal(sliced_output, full_output[37:38])
    assert torch.equal(sliced_grad, full_grad[37:38])

    permutation = torch.randperm(logits.shape[0], device="npu")
    permuted_output, permuted_grad = _run_logprobs(logits[permutation], labels[permutation], grad_output[permutation])
    inverse = torch.argsort(permutation)
    assert torch.equal(permuted_output[inverse], full_output)
    assert torch.equal(permuted_grad[inverse], full_grad)


def test_logprobs_supports_leading_dimensions_and_int32_labels():
    torch.manual_seed(107)
    logits = torch.randn(2, 3, 257, device="npu", dtype=torch.float32)
    labels = torch.randint(0, 257, (2, 3), device="npu", dtype=torch.int32)

    actual = logprobs_from_logits(logits, labels)
    expected = torch.log_softmax(logits, dim=-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    assert actual.shape == labels.shape
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_logprobs_rejects_mismatched_labels():
    logits = torch.randn(3, 17, device="npu", dtype=torch.float32)
    labels = torch.zeros(2, device="npu", dtype=torch.long)
    with pytest.raises(ValueError, match="leading logits dimensions"):
        logprobs_from_logits(logits, labels)
