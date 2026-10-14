from functools import partial

import pytest
import torch

from mojo_opset.functions.gemm import batch_invariant_linear
from mojo_opset.functions.gemm import batch_invariant_matmul

pytest.importorskip("triton")

linear = partial(batch_invariant_linear, implementation="batch_invariant")
matmul_persistent = partial(batch_invariant_matmul, implementation="batch_invariant")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="batch_invariant kernels require an Ascend NPU",
)


def _cpu_reference(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    result = a.detach().cpu().float() @ b.detach().cpu().float()
    if bias is not None:
        result += bias.detach().cpu().float()
    return result


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 2e-4, 2e-4),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 4e-2, 4e-2),
    ],
)
@pytest.mark.parametrize("shape", [(7, 19, 13), (5, 37, 77)])
def test_matmul_tail_accuracy(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    shape: tuple[int, int, int],
) -> None:
    torch.manual_seed(11)
    m, k, n = shape
    a = torch.randn((m, k), device="npu", dtype=dtype)
    b = torch.randn((k, n), device="npu", dtype=dtype)
    bias = torch.randn((n,), device="npu", dtype=dtype)

    actual = matmul_persistent(a, b, bias)
    expected = _cpu_reference(a, b, bias)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=atol, rtol=rtol)


def test_mojo_gemm_operator_uses_batch_invariant_linear() -> None:
    from mojo_opset.modules.gemm import BatchInvariantGemm

    torch.manual_seed(16)
    operator = BatchInvariantGemm(
        19, 13, bias=True, device="npu", dtype=torch.float16, implementation="batch_invariant"
    )
    input = torch.randn((3, 19), device="npu", dtype=torch.float16)
    expected = linear(input, operator.weight, operator.bias)

    torch.testing.assert_close(operator(input), expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matmul_slice_and_permutation_are_bitwise_exact(dtype: torch.dtype) -> None:
    torch.manual_seed(12)
    a = torch.randn((17, 37), device="npu", dtype=dtype)
    b = torch.randn((37, 29), device="npu", dtype=dtype)
    bias = torch.randn((29,), device="npu", dtype=dtype)
    permutation = torch.tensor([8, 1, 15, 0, 12, 6, 2, 16, 4, 10, 3, 14, 5, 9, 13, 7, 11], device="npu")

    full = matmul_persistent(a, b, bias)
    sliced = matmul_persistent(a[8:9], b, bias)
    permuted = matmul_persistent(a[permutation], b, bias)

    torch.testing.assert_close(sliced, full[8:9], atol=0.0, rtol=0.0)
    torch.testing.assert_close(permuted, full[permutation], atol=0.0, rtol=0.0)


def test_linear_leading_dims_forward_and_dx_are_batch_invariant() -> None:
    torch.manual_seed(13)
    dtype = torch.bfloat16
    input = torch.randn((2, 3, 37), device="npu", dtype=dtype)
    weight = torch.randn((29, 37), device="npu", dtype=dtype)
    bias = torch.randn((29,), device="npu", dtype=dtype)
    grad_output = torch.randn((2, 3, 29), device="npu", dtype=dtype)

    full_input = input.detach().clone().requires_grad_(True)
    full_output = linear(full_input, weight, bias)
    full_output.backward(grad_output)

    one_input = input[1:2, 2:3].detach().clone().requires_grad_(True)
    one_output = linear(one_input, weight, bias)
    one_output.backward(grad_output[1:2, 2:3])

    torch.testing.assert_close(one_output, full_output[1:2, 2:3], atol=0.0, rtol=0.0)
    torch.testing.assert_close(one_input.grad, full_input.grad[1:2, 2:3], atol=0.0, rtol=0.0)


def test_linear_wgrad_is_deterministic_for_a_fixed_batch() -> None:
    torch.manual_seed(14)
    input = torch.randn((2, 3, 37), device="npu", dtype=torch.bfloat16)
    weight = torch.randn((29, 37), device="npu", dtype=torch.bfloat16)
    bias = torch.randn((29,), device="npu", dtype=torch.bfloat16)
    grad_output = torch.randn((2, 3, 29), device="npu", dtype=torch.bfloat16)

    def run_backward():
        x = input.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True)
        linear(x, w, b).backward(grad_output)
        return x.grad, w.grad, b.grad

    first = run_backward()
    second = run_backward()
    for actual, expected in zip(first, second):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
