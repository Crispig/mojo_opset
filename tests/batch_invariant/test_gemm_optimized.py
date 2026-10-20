import pytest
import torch

from mojo_opset.kernels.npu_a2_batch_invariant.matmul import _matmul_persistent_kernel
from mojo_opset.kernels.npu_a2_batch_invariant.matmul import _matrix_layout
from mojo_opset.kernels.npu_a2_batch_invariant.matmul import _select_tile
from mojo_opset.kernels.npu_a2_batch_invariant.matmul import batch_invariant_matmul_bwd
from mojo_opset.kernels.npu_a2_batch_invariant.matmul import batch_invariant_matmul_fwd
from mojo_opset.kernels.npu_a2_batch_invariant.utils import get_num_cores

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _old_bf16_tile(a, b, bias=None):
    """Use the original tiles with the unchanged accumulator/K-loop kernel."""
    m, k = a.shape
    n = b.shape[1]
    block_m, block_n, block_k = (64, 128, 64)
    if n <= 64:
        block_m, block_n = 64, 64
        block_k = 32 if k <= 32 else 64
    if bias is not None:
        b = b.contiguous()
    a, transposed_a = _matrix_layout(a)
    b, transposed_b = _matrix_layout(b)
    output = torch.empty((m, n), dtype=a.dtype, device=a.device)
    _matmul_persistent_kernel[(get_num_cores("cube"),)](
        a,
        b,
        output,
        output if bias is None else bias,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_BIAS=bias is not None,
        A_TRANSPOSED=transposed_a,
        B_TRANSPOSED=transposed_b,
        multibuffer=True,
        sync_solver=False,
    )
    return output


def _assert_same_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


def test_bf16_default_tile_preserves_k_and_other_dtype_tables():
    assert _select_tile(torch.bfloat16, 4096, 12288) == (128, 256, 64)
    assert _select_tile(torch.bfloat16, 19, 13) == (64, 64, 32)
    assert _select_tile(torch.bfloat16, 37, 29) == (64, 64, 64)
    assert _select_tile(torch.float16, 4096, 12288) == (64, 128, 64)
    assert _select_tile(torch.float32, 4096, 12288) == (64, 64, 32)


@pytest.mark.parametrize("rows", [1, 8, 31, 64, 65, 257, 1024])
@pytest.mark.parametrize("shape", [(37, 77), (4096, 1024), (4096, 4096), (4096, 12288), (12288, 4096)])
def test_bf16_larger_output_tile_preserves_original_bits(rows, shape):
    torch.manual_seed(103)
    k, n = shape
    a = torch.randn((rows, k), dtype=torch.bfloat16, device="npu")
    b = torch.randn((k, n), dtype=a.dtype, device=a.device)
    actual = batch_invariant_matmul_fwd(a, b)
    _assert_same_bits(actual, _old_bf16_tile(a, b))
    if rows > 1:
        one = batch_invariant_matmul_fwd(a[-1:], b)
        _assert_same_bits(one, actual[-1:])
        permutation = torch.randperm(rows, device=a.device)
        _assert_same_bits(batch_invariant_matmul_fwd(a[permutation], b), actual[permutation])


@pytest.mark.parametrize("transposed_a", [False, True])
@pytest.mark.parametrize("transposed_b", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_bf16_larger_tile_preserves_layouts_and_bias(transposed_a, transposed_b, has_bias):
    torch.manual_seed(107)
    m, k, n = 65, 129, 259
    a = torch.randn((k, m) if transposed_a else (m, k), dtype=torch.bfloat16, device="npu")
    b = torch.randn((n, k) if transposed_b else (k, n), dtype=a.dtype, device=a.device)
    a = a.t() if transposed_a else a
    b = b.t() if transposed_b else b
    bias = torch.randn(n, dtype=a.dtype, device=a.device) if has_bias else None
    _assert_same_bits(batch_invariant_matmul_fwd(a, b, bias), _old_bf16_tile(a, b, bias))


@pytest.mark.parametrize("rows", [8, 65, 257])
def test_bf16_larger_tile_preserves_backward_and_fixed_batch_repeats(rows):
    torch.manual_seed(109)
    a = torch.randn((rows, 129), dtype=torch.bfloat16, device="npu")
    b = torch.randn((129, 259), dtype=a.dtype, device=a.device)
    grad = torch.randn((rows, 259), dtype=a.dtype, device=a.device)
    first = batch_invariant_matmul_bwd(grad, a, b, True, (True, True, True))
    repeated = batch_invariant_matmul_bwd(grad, a, b, True, (True, True, True))
    expected = (
        _old_bf16_tile(grad, b.t()),
        _old_bf16_tile(a.t(), grad),
        _old_bf16_tile(torch.ones((1, rows), dtype=a.dtype, device=a.device), grad).reshape(-1),
    )
    for actual, reference, repeat in zip(first, expected, repeated):
        _assert_same_bits(actual, reference)
        _assert_same_bits(actual, repeat)
    single_dx = batch_invariant_matmul_bwd(grad[-1:], a[-1:], b, False, (True, False, False))[0]
    _assert_same_bits(single_dx, first[0][-1:])


@pytest.mark.parametrize("rows", [1, 8, 32, 257])
def test_bf16_lm_head_tail_preserves_original_bits(rows):
    torch.manual_seed(127)
    a = torch.randn((rows, 4096), dtype=torch.bfloat16, device="npu")
    weight = torch.randn((151936, 4096), dtype=a.dtype, device=a.device)
    _assert_same_bits(batch_invariant_matmul_fwd(a, weight.t()), _old_bf16_tile(a, weight.t()))


@pytest.mark.parametrize("scale", [1e-30, 1e-8, 1e8, 1e30])
def test_bf16_output_tile_preserves_wide_finite_exponents(scale):
    torch.manual_seed(131)
    a = (torch.randn((65, 129), device="npu") * scale).to(torch.bfloat16)
    b = (torch.randn((129, 259), device=a.device) / scale).to(a.dtype)
    a[0].zero_()
    a[1].fill_(-0.0)
    _assert_same_bits(batch_invariant_matmul_fwd(a, b), _old_bf16_tile(a, b))
