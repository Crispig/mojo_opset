import pytest
import torch
import triton
import triton.language as tl

from mojo_opset.kernels.npu_a2_batch_invariant.rmsnorm import rmsnorm_bwd
from mojo_opset.kernels.npu_a2_batch_invariant.rmsnorm import rmsnorm_fwd
from mojo_opset.kernels.npu_a2_batch_invariant.utils import get_num_cores
from mojo_opset.kernels.npu_a2_batch_invariant.utils import libentry

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@libentry()
@triton.jit
def _single_row_reference(
    input_ptr,
    weight_ptr,
    output_ptr,
    rstd_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Freeze the original single-row arithmetic, including its fp32 rstd."""
    for row in range(tl.program_id(0), n_rows, tl.num_programs(0)):
        offsets = row.to(tl.int64) * n_cols
        sum_squares = tl.zeros((1,), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            values = tl.load(input_ptr + offsets + cols, mask=mask, other=0.0).to(tl.float32)
            sum_squares += tl.sum(tl.where(mask, values * values, 0.0))
        reciprocal_rms = 1.0 / tl.sqrt(sum_squares / n_cols + eps)
        tl.store(rstd_ptr + row.to(tl.int64) + tl.arange(0, 1), reciprocal_rms)
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            values = tl.load(input_ptr + offsets + cols, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            output = values * reciprocal_rms * weight
            tl.store(output_ptr + offsets + cols, output, mask=mask)


def _reference(input, weight, eps=1e-6):
    contiguous = input.contiguous()
    n_cols = input.shape[-1]
    n_rows = input.numel() // n_cols
    output = torch.empty_like(contiguous)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=input.device)
    _single_row_reference[(get_num_cores("vector"),)](
        contiguous, weight, output, rstd, n_rows, n_cols, eps, BLOCK_SIZE=1024
    )
    return output, rstd


def _assert_same_bits(actual, expected):
    integer_dtype = torch.int32 if actual.element_size() == 4 else torch.int16
    assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [128, 1000, 1024, 1536, 4096])
def test_vectorized_rmsnorm_preserves_single_row_bits(dtype, hidden_size):
    torch.manual_seed(91)
    input = torch.randn((19, hidden_size), dtype=dtype, device="npu")
    weight = torch.randn(hidden_size, dtype=dtype, device="npu")
    actual, rstd = rmsnorm_fwd(input, weight, 1e-6)
    expected, expected_rstd = _reference(input, weight)
    _assert_same_bits(actual, expected)
    _assert_same_bits(rstd, expected_rstd)


@pytest.mark.parametrize("hidden_size", [128, 1536, 4096])
def test_vectorized_rmsnorm_preserves_backward_and_row_permutations(hidden_size):
    torch.manual_seed(93)
    input = torch.randn((113, hidden_size), dtype=torch.bfloat16, device="npu")
    weight = torch.randn(hidden_size, dtype=input.dtype, device=input.device)
    grad = torch.randn_like(input)
    output, rstd = rmsnorm_fwd(input, weight, 1e-6)
    reference_output, reference_rstd = _reference(input, weight)
    dx, dw = rmsnorm_bwd(grad, input, weight, rstd)
    reference_dx, reference_dw = rmsnorm_bwd(grad, input, weight, reference_rstd)
    _assert_same_bits(output, reference_output)
    _assert_same_bits(dx, reference_dx)
    _assert_same_bits(dw, reference_dw)

    indices = torch.randperm(input.shape[0], device=input.device)
    permuted, permuted_rstd = rmsnorm_fwd(input[indices], weight, 1e-6)
    _assert_same_bits(permuted, output[indices])
    _assert_same_bits(permuted_rstd, rstd[indices])
    sliced, sliced_rstd = rmsnorm_fwd(input[98:99], weight, 1e-6)
    _assert_same_bits(sliced, output[98:99])
    _assert_same_bits(sliced_rstd, rstd[98:99])


def test_vectorized_rmsnorm_graph_replay_uses_current_inputs():
    torch.manual_seed(97)
    input = torch.randn((1031, 128), dtype=torch.bfloat16, device="npu")
    weight = torch.randn(128, dtype=input.dtype, device=input.device)
    for _ in range(3):
        rmsnorm_fwd(input, weight, 1e-6)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, rstd = rmsnorm_fwd(input, weight, 1e-6)
    replacement = torch.randn_like(input)
    input.copy_(replacement)
    graph.replay()
    expected, expected_rstd = _reference(input, weight)
    _assert_same_bits(output, expected)
    _assert_same_bits(rstd, expected_rstd)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [128, 1536, 4096])
def test_rmsnorm_row_task_boundaries_preserve_single_token_and_backward_bits(dtype, hidden_size):
    torch.manual_seed(109)
    # Vary only the grouping of independent rows, including task-size boundaries.
    core_count = get_num_cores("vector")
    boundaries = {1, 39, 40, 79, 80, 159, 160, 319, 320}
    for scale in (1, 2, 4, 8):
        boundaries.update((max(1, scale * core_count - 1), scale * core_count))
    backing = torch.randn((max(boundaries), hidden_size * 2), dtype=dtype, device="npu")
    probe = torch.randn(hidden_size, dtype=dtype, device="npu")
    weight = torch.randn(hidden_size * 2, dtype=dtype, device="npu")[::2]
    reference_weight = weight.contiguous()
    single_output, single_rstd = _reference(probe[None], reference_weight)
    for rows in sorted(boundaries):
        input = backing[:rows, ::2]
        input[-1].copy_(probe)
        output, rstd = rmsnorm_fwd(input, weight, 1e-6)
        expected, expected_rstd = _reference(input, reference_weight)
        _assert_same_bits(output, expected)
        _assert_same_bits(rstd, expected_rstd)
        _assert_same_bits(output[-1:], single_output)
        _assert_same_bits(rstd[-1:], single_rstd)

        permutation = torch.randperm(rows, device=input.device)
        permuted, permuted_rstd = rmsnorm_fwd(input[permutation], weight, 1e-6)
        _assert_same_bits(permuted, output[permutation])
        _assert_same_bits(permuted_rstd, rstd[permutation])

        if rows in (1, 80, 160, 320):
            grad = torch.ones_like(input)
            dx, dw = rmsnorm_bwd(grad, input, weight, rstd)
            reference_dx, reference_dw = rmsnorm_bwd(grad, input, weight, expected_rstd)
            _assert_same_bits(dx, reference_dx)
            _assert_same_bits(dw, reference_dw)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_rmsnorm_preserves_signed_zero_and_live_graph_weights(dtype):
    input = torch.zeros((67, 128), dtype=dtype, device="npu")
    input[:, 1::2] = -0.0
    weight = torch.randn(128, dtype=dtype, device="npu")
    actual, rstd = rmsnorm_fwd(input, weight, 1e-6)
    expected, reference_rstd = _reference(input, weight)
    _assert_same_bits(actual, expected)
    _assert_same_bits(rstd, reference_rstd)

    for _ in range(3):
        rmsnorm_fwd(input, weight, 1e-6)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, captured_rstd = rmsnorm_fwd(input, weight, 1e-6)
    input.copy_(torch.randn_like(input))
    weight.copy_(torch.randn_like(weight))
    graph.replay()
    expected, expected_rstd = _reference(input, weight)
    _assert_same_bits(output, expected)
    _assert_same_bits(captured_rstd, expected_rstd)
