from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import get_num_cores
from .utils import libentry
from .utils import require_deterministic_mode

_BLOCK_SIZE = 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit
def _softmax_rows_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    LOG_SOFTMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Three-pass row softmax with a fixed fp32 reduction order."""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_index in range(program_id, n_rows, num_programs):
        row_index_i64 = row_index.to(tl.int64)
        input_row_ptr = input_ptr + row_index_i64 * n_cols
        output_row_ptr = output_ptr + row_index_i64 * n_cols

        maximum = float("-inf")
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            values = tl.load(
                input_row_ptr + column_offsets,
                mask=column_mask,
                other=float("-inf"),
            ).to(tl.float32)
            maximum = tl.maximum(maximum, tl.max(values, axis=0))

        denominator = 0.0
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            values = tl.load(
                input_row_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            exponentials = tl.exp(values - maximum)
            denominator += tl.sum(tl.where(column_mask, exponentials, 0.0), axis=0)

        if LOG_SOFTMAX:
            log_denominator = tl.log(denominator)
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            values = tl.load(
                input_row_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            if LOG_SOFTMAX:
                result = values - maximum - log_denominator
            else:
                result = tl.exp(values - maximum) / denominator
            tl.store(output_row_ptr + column_offsets, result, mask=column_mask)


@libentry()
@triton.jit
def _softmax_backward_rows_kernel(
    output_ptr,
    grad_output_ptr,
    grad_input_ptr,
    n_rows,
    n_cols,
    LOG_SOFTMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax derivatives with the row reduction kept inside this Triton kernel."""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_index in range(program_id, n_rows, num_programs):
        row_index_i64 = row_index.to(tl.int64)
        output_row_ptr = output_ptr + row_index_i64 * n_cols
        grad_output_row_ptr = grad_output_ptr + row_index_i64 * n_cols
        grad_input_row_ptr = grad_input_ptr + row_index_i64 * n_cols

        reduction = 0.0
        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            output = tl.load(
                output_row_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_output_row_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            if LOG_SOFTMAX:
                reduction += tl.sum(grad_output, axis=0)
            else:
                reduction += tl.sum(output * grad_output, axis=0)

        for column_start in range(0, n_cols, BLOCK_SIZE):
            column_offsets = column_start + tl.arange(0, BLOCK_SIZE)
            column_mask = column_offsets < n_cols
            output = tl.load(
                output_row_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_output_row_ptr + column_offsets,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            if LOG_SOFTMAX:
                grad_input = grad_output - tl.exp(output) * reduction
            else:
                grad_input = output * (grad_output - reduction)
            tl.store(grad_input_row_ptr + column_offsets, grad_input, mask=column_mask)


def _validate_tensor(input: torch.Tensor) -> None:
    if input.device.type != "npu":
        raise ValueError(f"batch_invariant softmax requires an NPU tensor, got {input.device}")
    if input.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"batch_invariant softmax supports float16, bfloat16, and float32; got {input.dtype}")


def _resolve_output_dtype(
    input: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.dtype:
    output_dtype = input.dtype if dtype is None else dtype
    if output_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"batch_invariant softmax supports float16, bfloat16, and float32 outputs; got {output_dtype}")
    return output_dtype


def _softmax_forward(input_2d, output_dtype, log_softmax):
    require_deterministic_mode()
    _validate_tensor(input_2d)
    output_dtype = _resolve_output_dtype(input_2d, output_dtype)
    n_rows, n_cols = input_2d.shape
    output = torch.empty((n_rows, n_cols), dtype=output_dtype, device=input_2d.device)
    if output.numel() != 0:
        _softmax_rows_kernel[(get_num_cores("vector"),)](
            input_2d,
            output,
            n_rows,
            n_cols,
            LOG_SOFTMAX=log_softmax,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return output


def _softmax_backward(grad_output, output, input_dtype, log_softmax):
    grad_input = torch.empty(output.shape, dtype=input_dtype, device=output.device)
    if grad_input.numel() != 0:
        n_rows, n_cols = output.shape
        _softmax_backward_rows_kernel[(get_num_cores("vector"),)](
            output,
            grad_output,
            grad_input,
            n_rows,
            n_cols,
            LOG_SOFTMAX=log_softmax,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return grad_input


def softmax_fwd(input_2d, output_dtype):
    return _softmax_forward(input_2d, output_dtype, log_softmax=False)


def softmax_bwd(grad_output, output, input_dtype):
    return _softmax_backward(grad_output, output, input_dtype, log_softmax=False)


def log_softmax_fwd(input_2d, output_dtype):
    return _softmax_forward(input_2d, output_dtype, log_softmax=True)


def log_softmax_bwd(grad_output, output, input_dtype):
    return _softmax_backward(grad_output, output, input_dtype, log_softmax=True)


__all__ = ["softmax_fwd", "softmax_bwd", "log_softmax_fwd", "log_softmax_bwd"]
