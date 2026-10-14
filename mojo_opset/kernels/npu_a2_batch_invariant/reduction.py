from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import get_num_cores
from .utils import libentry
from .utils import require_deterministic_mode

_BLOCK_SIZE = 1024
_REDUCE_SUM = 0
_REDUCE_MEAN = 1
_REDUCE_AMAX = 2
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit
def _reduce_rows_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    REDUCTION_KIND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce contiguous rows with a fixed, batch-independent reduction tree."""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_index in range(program_id, n_rows, num_programs):
        row_index_i64 = row_index.to(tl.int64)
        row_ptr = input_ptr + row_index_i64 * n_cols

        if REDUCTION_KIND == 2:
            accumulator = float("-inf")
            nan_count = 0
            for column_start in range(0, n_cols, BLOCK_SIZE):
                column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
                column_mask = column_offsets < n_cols
                values = tl.load(
                    row_ptr + column_offsets,
                    mask=column_mask,
                    other=float("-inf"),
                ).to(tl.float32)
                is_nan = column_mask & (values != values)
                nan_count += tl.sum(is_nan.to(tl.int32), axis=0)
                finite_values = tl.where(is_nan, float("-inf"), values)
                accumulator = tl.maximum(
                    accumulator,
                    tl.max(finite_values, axis=0),
                )
            accumulator = tl.where(nan_count > 0, float("nan"), accumulator)
        else:
            accumulator = 0.0
            for column_start in range(0, n_cols, BLOCK_SIZE):
                column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
                column_mask = column_offsets < n_cols
                values = tl.load(
                    row_ptr + column_offsets,
                    mask=column_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(values, axis=0)
            if REDUCTION_KIND == 1:
                accumulator /= n_cols

        tl.store(output_ptr + row_index_i64, accumulator)


@libentry()
@triton.jit
def _sum_mean_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    n_rows,
    n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_index in range(program_id, n_rows, num_programs):
        row_index_i64 = row_index.to(tl.int64)
        grad_value = tl.load(grad_output_ptr + row_index_i64).to(tl.float32) * scale
        row_ptr = grad_input_ptr + row_index_i64 * n_cols
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            tl.store(row_ptr + column_offsets, grad_value, mask=column_mask)


@libentry()
@triton.jit
def _amax_backward_kernel(
    input_ptr,
    output_ptr,
    grad_output_ptr,
    grad_input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Split ``amax`` gradients equally between ties, matching torch.amax."""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_index in range(program_id, n_rows, num_programs):
        row_index_i64 = row_index.to(tl.int64)
        row_input_ptr = input_ptr + row_index_i64 * n_cols
        row_grad_ptr = grad_input_ptr + row_index_i64 * n_cols
        maximum = tl.load(output_ptr + row_index_i64).to(tl.float32)
        grad_output = tl.load(grad_output_ptr + row_index_i64).to(tl.float32)

        tie_count = 0.0
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            values = tl.load(
                row_input_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            is_maximum = column_mask & (values == maximum)
            tie_count += tl.sum(is_maximum.to(tl.float32), axis=0)

        split_gradient = grad_output / tie_count
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            values = tl.load(
                row_input_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            gradient = tl.where(values == maximum, split_gradient, 0.0)
            # Triton-X 3.2.0 miscompiles a scalar predicate selecting a vector
            # result here. Make the NaN predicate lane-shaped to keep the
            # generated AIVEC load/store widths consistent.
            maximum_is_nan = column_mask & (maximum != maximum)
            gradient = tl.where(maximum_is_nan, maximum, gradient)
            tl.store(row_grad_ptr + column_offsets, gradient, mask=column_mask)


def _validate_tensor(input: torch.Tensor) -> None:
    if input.device.type != "npu":
        raise ValueError(f"batch_invariant reductions require an NPU tensor, got {input.device}")
    if input.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"batch_invariant reductions support float16, bfloat16, and float32; got {input.dtype}")


def _resolve_output_dtype(
    input: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.dtype:
    output_dtype = input.dtype if dtype is None else dtype
    if output_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"batch_invariant reductions support float16, bfloat16, and float32 outputs; got {output_dtype}"
        )
    return output_dtype


def _launch_reduce_rows(
    input_2d: torch.Tensor,
    reduction_kind: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    require_deterministic_mode()
    _validate_tensor(input_2d)
    output_dtype = _resolve_output_dtype(input_2d, output_dtype)
    n_rows, n_cols = input_2d.shape
    if reduction_kind == _REDUCE_AMAX and n_cols == 0 and n_rows != 0:
        raise IndexError("amax(): Expected reduction dim to be specified for input.numel() == 0")

    output = torch.empty((n_rows,), dtype=output_dtype, device=input_2d.device)
    if n_rows == 0:
        return output

    _reduce_rows_kernel[(get_num_cores("vector"),)](
        input_2d,
        output,
        n_rows,
        n_cols,
        REDUCTION_KIND=reduction_kind,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return output


def _sum_mean_backward(grad_output, input_shape, input_dtype, mean):
    n_rows, n_cols = input_shape
    grad_input = torch.empty(input_shape, dtype=input_dtype, device=grad_output.device)
    if grad_input.numel() != 0:
        scale = 1.0 / n_cols if mean else 1.0
        _sum_mean_backward_kernel[(get_num_cores("vector"),)](
            grad_output,
            grad_input,
            n_rows,
            n_cols,
            scale,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return grad_input


def reduce_sum_fwd(input_2d, output_dtype):
    return _launch_reduce_rows(input_2d, _REDUCE_SUM, output_dtype)


def reduce_sum_bwd(grad_output, input_shape, input_dtype):
    return _sum_mean_backward(grad_output, input_shape, input_dtype, mean=False)


def mean_fwd(input_2d, output_dtype):
    return _launch_reduce_rows(input_2d, _REDUCE_MEAN, output_dtype)


def mean_bwd(grad_output, input_shape, input_dtype):
    return _sum_mean_backward(grad_output, input_shape, input_dtype, mean=True)


def amax_fwd(input_2d, output_dtype):
    return _launch_reduce_rows(input_2d, _REDUCE_AMAX, output_dtype)


def amax_bwd(grad_output, input_2d, output):
    grad_input = torch.empty_like(input_2d)
    if input_2d.numel() != 0:
        n_rows, n_cols = input_2d.shape
        _amax_backward_kernel[(get_num_cores("vector"),)](
            input_2d,
            output,
            grad_output,
            grad_input,
            n_rows,
            n_cols,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return grad_input


__all__ = ["reduce_sum_fwd", "reduce_sum_bwd", "mean_fwd", "mean_bwd", "amax_fwd", "amax_bwd"]
