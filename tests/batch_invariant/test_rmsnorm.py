from functools import partial

import pytest
import torch
import torch.nn.functional as F

from mojo_opset.functions.rms_norm import rms_norm
from mojo_opset.modules import RMSNorm

batch_invariant_rms_norm = partial(rms_norm, implementation="batch_invariant")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _run_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_leaf = input.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)
    output = batch_invariant_rms_norm(input_leaf, weight_leaf, eps)
    grad_input, grad_weight = torch.autograd.grad(
        output,
        (input_leaf, weight_leaf),
        grad_outputs=grad_output,
    )
    return output.detach(), grad_input.detach(), grad_weight.detach()


def _run_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_leaf = input.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)
    output = F.rms_norm(
        input_leaf,
        (input.shape[-1],),
        weight=weight_leaf,
        eps=eps,
    )
    grad_input, grad_weight = torch.autograd.grad(
        output,
        (input_leaf, weight_leaf),
        grad_outputs=grad_output,
    )
    return output.detach(), grad_input.detach(), grad_weight.detach()


@pytest.mark.parametrize(
    "dtype,weight_dtype,hidden_size,atol,rtol",
    [
        (torch.float32, torch.float32, 1000, 2e-5, 2e-5),
        (torch.bfloat16, torch.bfloat16, 4096, 3e-2, 3e-2),
        (torch.bfloat16, torch.float32, 3072, 3e-2, 3e-2),
    ],
)
def test_rmsnorm_forward_backward_accuracy(
    dtype,
    weight_dtype,
    hidden_size,
    atol,
    rtol,
):
    torch.manual_seed(23)
    device = torch.device("npu")
    input = torch.randn(5, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=weight_dtype, device=device)
    grad_output = torch.randn_like(input)

    actual = _run_rmsnorm(input, weight, grad_output)
    expected = _run_reference(input, weight, grad_output)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_forward_and_dx_are_slice_and_permutation_invariant(dtype):
    torch.manual_seed(29)
    device = torch.device("npu")
    # 58 flattened rows exceed the 48 vector cores, exercising the persistent
    # grid loop as well as moving a row between different program IDs.
    input = torch.randn(29, 2, 1536, dtype=dtype, device=device)
    weight = torch.randn(1536, dtype=dtype, device=device)
    grad_output = torch.randn_like(input)

    full_output, full_grad_input, _ = _run_rmsnorm(input, weight, grad_output)
    sliced_output, sliced_grad_input, _ = _run_rmsnorm(
        input[25:26],
        weight,
        grad_output[25:26],
    )
    assert torch.equal(sliced_output, full_output[25:26])
    assert torch.equal(sliced_grad_input, full_grad_input[25:26])

    permutation = torch.randperm(input.shape[0], device=device)
    permuted_output, permuted_grad_input, _ = _run_rmsnorm(
        input[permutation],
        weight,
        grad_output[permutation],
    )
    inverse_permutation = torch.argsort(permutation)
    assert torch.equal(permuted_output[inverse_permutation], full_output)
    assert torch.equal(permuted_grad_input[inverse_permutation], full_grad_input)


def test_rmsnorm_weight_gradient_is_repeatable_for_a_fixed_full_batch():
    """dW is deterministic for one batch, but is not mathematically batch-invariant."""

    torch.manual_seed(31)
    device = torch.device("npu")
    input = torch.randn(113, 2048, dtype=torch.bfloat16, device=device)
    weight = torch.randn(2048, dtype=torch.bfloat16, device=device)
    grad_output = torch.randn_like(input)

    first = _run_rmsnorm(input, weight, grad_output)[2]
    second = _run_rmsnorm(input, weight, grad_output)[2]
    assert torch.equal(first, second)


def test_rmsnorm_module_interface_uses_batch_invariant_function():
    torch.manual_seed(37)
    device = torch.device("npu")
    input = torch.randn(3, 1024, dtype=torch.bfloat16, device=device)
    module = RMSNorm(
        normalized_shape=1024,
        eps=1e-6,
        device=device,
        dtype=torch.bfloat16,
        implementation="batch_invariant",
    )
    with torch.no_grad():
        module.weight.copy_(torch.randn_like(module.weight))

    output = module(input)
    expected = batch_invariant_rms_norm(
        input,
        module.weight,
        module.eps,
    )
    assert torch.equal(output, expected)
    assert module.implementation == "batch_invariant"


def test_rmsnorm_rejects_unsupported_dtype_and_empty_hidden_size():
    with pytest.raises(TypeError, match="float16, bfloat16, and float32"):
        batch_invariant_rms_norm(
            torch.ones(2, 8, device="npu", dtype=torch.float64),
            torch.ones(8, device="npu", dtype=torch.float64),
            1e-6,
        )

    with pytest.raises(ValueError, match="hidden size must be positive"):
        batch_invariant_rms_norm(
            torch.empty(2, 0, device="npu", dtype=torch.float32),
            torch.empty(0, device="npu", dtype=torch.float32),
            1e-6,
        )
