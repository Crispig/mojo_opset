from collections.abc import Callable
from functools import partial

import pytest
import torch

from mojo_opset import functions

reduce_sum = partial(functions.reduce_sum, implementation="batch_invariant")
mean = partial(functions.mean, implementation="batch_invariant")
amax = partial(functions.amax, implementation="batch_invariant")
softmax = partial(functions.softmax, implementation="batch_invariant")
log_softmax = partial(functions.log_softmax, implementation="batch_invariant")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-5, 2e-5
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 2e-2, 2e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("dim", "keepdim"),
    [(-1, False), (1, True), ((1, -1), False), (None, True)],
)
@pytest.mark.parametrize(
    ("implementation", "reference"),
    [(reduce_sum, torch.sum), (mean, torch.mean), (amax, torch.amax)],
)
def test_reductions_match_torch_reference(dtype, dim, keepdim, implementation, reference):
    torch.manual_seed(101)
    input = torch.randn(3, 5, 7, 11, dtype=dtype, device="npu")
    actual = implementation(input, dim=dim, keepdim=keepdim)
    expected = reference(input, dim=dim, keepdim=keepdim)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("implementation", [reduce_sum, mean])
def test_reduction_dtype_and_noncontiguous_input(implementation):
    torch.manual_seed(103)
    input = torch.randn(2, 7, 5, dtype=torch.bfloat16, device="npu").transpose(0, 2)
    actual = implementation(input, dim=(0, -1), keepdim=True, dtype=torch.float32)
    expected = (
        torch.sum(input, dim=(0, -1), keepdim=True, dtype=torch.float32)
        if implementation is reduce_sum
        else torch.mean(input, dim=(0, -1), keepdim=True, dtype=torch.float32)
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("implementation", "reference"),
    [(reduce_sum, torch.sum), (mean, torch.mean), (amax, torch.amax)],
)
def test_reduction_backward_matches_torch(implementation, reference):
    torch.manual_seed(107)
    data = torch.randn(4, 6, 9, dtype=torch.float32, device="npu")
    input = data.detach().clone().requires_grad_(True)
    reference_input = data.detach().clone().requires_grad_(True)
    grad_output = torch.randn(4, 1, 1, dtype=torch.float32, device="npu")

    actual = implementation(input, dim=(1, 2), keepdim=True)
    expected = reference(reference_input, dim=(1, 2), keepdim=True)
    actual_grad = torch.autograd.grad(actual, input, grad_outputs=grad_output)[0]
    expected_grad = torch.autograd.grad(expected, reference_input, grad_outputs=grad_output)[0]

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)


def test_amax_backward_splits_ties_like_torch():
    input = torch.tensor(
        [[1.0, 4.0, 4.0, -2.0], [3.0, 3.0, 3.0, 1.0]],
        dtype=torch.float32,
        device="npu",
        requires_grad=True,
    )
    output = amax(input, dim=-1)
    gradient = torch.autograd.grad(output, input, grad_outputs=torch.tensor([2.0, 3.0], device="npu"))[0]
    expected = torch.tensor(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
        dtype=torch.float32,
        device="npu",
    )
    assert torch.equal(gradient, expected)


def test_amax_propagates_nan_in_forward_and_backward():
    input = torch.tensor(
        [[1.0, float("nan"), 2.0]],
        dtype=torch.float32,
        device="npu",
        requires_grad=True,
    )
    output = amax(input, dim=-1)
    (gradient,) = torch.autograd.grad(output, input)
    assert torch.isnan(output).all()
    assert torch.isnan(gradient).all()


@pytest.mark.parametrize("implementation", [reduce_sum, mean, amax])
def test_reduction_is_bitwise_invariant_to_batch_slicing_and_permutation(implementation):
    torch.manual_seed(109)
    data = torch.randn(7, 5, 1031, dtype=torch.bfloat16, device="npu")
    grad_output = torch.randn(7, dtype=torch.bfloat16, device="npu")
    input = data.detach().clone().requires_grad_(True)

    whole = implementation(input, dim=(1, 2))
    whole_grad = torch.autograd.grad(whole, input, grad_outputs=grad_output)[0]

    sliced_outputs = []
    sliced_gradients = []
    for row, row_grad in zip(data, grad_output):
        sliced_input = row.detach().clone().requires_grad_(True)
        sliced_output = implementation(sliced_input, dim=(0, 1))
        sliced_gradient = torch.autograd.grad(
            sliced_output,
            sliced_input,
            grad_outputs=row_grad,
        )[0]
        sliced_outputs.append(sliced_output)
        sliced_gradients.append(sliced_gradient)
    assert torch.equal(whole, torch.stack(sliced_outputs))
    assert torch.equal(whole_grad, torch.stack(sliced_gradients))

    permutation = torch.tensor([5, 0, 6, 2, 1, 4, 3], device="npu")
    permuted_input = data[permutation].detach().clone().requires_grad_(True)
    permuted_output = implementation(permuted_input, dim=(1, 2))
    permuted_gradient = torch.autograd.grad(
        permuted_output,
        permuted_input,
        grad_outputs=grad_output[permutation],
    )[0]
    assert torch.equal(permuted_output, whole[permutation])
    assert torch.equal(permuted_gradient, whole_grad[permutation])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [-1, 1, -2])
@pytest.mark.parametrize(
    ("implementation", "reference"),
    [(softmax, torch.softmax), (log_softmax, torch.log_softmax)],
)
def test_softmax_variants_match_torch_reference(dtype, dim, implementation, reference):
    torch.manual_seed(113)
    input = torch.randn(4, 9, 37, dtype=dtype, device="npu")
    actual = implementation(input, dim=dim)
    expected = reference(input, dim=dim)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("implementation", "reference"),
    [(softmax, torch.softmax), (log_softmax, torch.log_softmax)],
)
def test_softmax_dtype_and_noncontiguous_input(implementation, reference):
    torch.manual_seed(127)
    input = torch.randn(3, 11, 5, dtype=torch.bfloat16, device="npu").transpose(0, 2)
    actual = implementation(input, dim=1, dtype=torch.float32)
    expected = reference(input, dim=1, dtype=torch.float32)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("implementation", "reference"),
    [(softmax, torch.softmax), (log_softmax, torch.log_softmax)],
)
def test_softmax_supports_an_empty_reduction_dimension(implementation, reference):
    input = torch.empty(3, 0, 5, dtype=torch.float32, device="npu", requires_grad=True)
    actual = implementation(input, dim=1)
    expected = reference(input, dim=1)
    assert actual.shape == input.shape
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert input.grad is not None
    assert input.grad.shape == input.shape


def test_log_softmax_defaults_to_the_last_dimension():
    input = torch.randn(2, 17, dtype=torch.float32, device="npu")
    assert torch.equal(log_softmax(input), log_softmax(input, dim=-1))


@pytest.mark.parametrize(
    ("implementation", "reference"),
    [(softmax, torch.softmax), (log_softmax, torch.log_softmax)],
)
def test_softmax_backward_matches_torch(implementation, reference):
    torch.manual_seed(131)
    data = torch.randn(3, 13, 17, dtype=torch.float32, device="npu")
    input = data.detach().clone().requires_grad_(True)
    reference_input = data.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(data)

    actual = implementation(input, dim=1)
    expected = reference(reference_input, dim=1)
    actual_grad = torch.autograd.grad(actual, input, grad_outputs=grad_output)[0]
    expected_grad = torch.autograd.grad(expected, reference_input, grad_outputs=grad_output)[0]

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("implementation", [softmax, log_softmax])
def test_softmax_is_bitwise_invariant_to_batch_slicing_and_permutation(implementation):
    torch.manual_seed(137)
    data = torch.randn(7, 1031, 3, dtype=torch.bfloat16, device="npu")
    grad_output = torch.randn_like(data)
    input = data.detach().clone().requires_grad_(True)

    whole = implementation(input, dim=1)
    whole_grad = torch.autograd.grad(whole, input, grad_outputs=grad_output)[0]

    sliced_outputs = []
    sliced_gradients = []
    for row, row_grad in zip(data, grad_output):
        sliced_input = row.detach().clone().requires_grad_(True)
        sliced_output = implementation(sliced_input, dim=0)
        sliced_gradient = torch.autograd.grad(
            sliced_output,
            sliced_input,
            grad_outputs=row_grad,
        )[0]
        sliced_outputs.append(sliced_output)
        sliced_gradients.append(sliced_gradient)
    assert torch.equal(whole, torch.stack(sliced_outputs))
    assert torch.equal(whole_grad, torch.stack(sliced_gradients))

    permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2], device="npu")
    permuted_input = data[permutation].detach().clone().requires_grad_(True)
    permuted_output = implementation(permuted_input, dim=1)
    permuted_gradient = torch.autograd.grad(
        permuted_output,
        permuted_input,
        grad_outputs=grad_output[permutation],
    )[0]
    assert torch.equal(permuted_output, whole[permutation])
    assert torch.equal(permuted_gradient, whole_grad[permutation])


@pytest.mark.parametrize(
    ("operator_type", "reference"),
    [
        (functions.reduce_sum, torch.sum),
        (functions.mean, torch.mean),
        (functions.amax, torch.amax),
        (functions.softmax, torch.softmax),
        (functions.log_softmax, torch.log_softmax),
    ],
)
def test_stateless_backend_operator_interface(
    operator_type: Callable,
    reference: Callable[..., torch.Tensor],
):
    torch.manual_seed(139)
    input = torch.randn(2, 19, dtype=torch.float32, device="npu")
    actual = operator_type(input, dim=-1, implementation="batch_invariant")
    expected = reference(input, dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
