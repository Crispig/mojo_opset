from typing import Tuple

import torch
import triton
import triton.language as tl

from .utils import get_num_cores
from .utils import libentry
from .utils import require_deterministic_mode

# The reduction tile is intentionally independent of the number of rows.  It
# matches VeXact's GPU RMSNorm tile, so adding or removing unrelated rows never
# changes a row's floating-point reduction tree.
_REDUCTION_BLOCK_SIZE = 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit
def _rmsnorm_fwd_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    rstd_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_idx in range(program_id, n_rows, num_programs):
        row_idx_i64 = row_idx.to(tl.int64)
        input_row = input_ptr + row_idx_i64 * input_row_stride
        output_row = output_ptr + row_idx_i64 * output_row_stride

        sum_squares = tl.zeros((1,), dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offsets < n_cols
            values = tl.load(input_row + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            sum_squares += tl.sum(tl.where(col_mask, values * values, 0.0))

        reciprocal_rms = 1.0 / tl.sqrt(sum_squares / n_cols + eps)
        tl.store(rstd_ptr + row_idx_i64 + tl.arange(0, 1), reciprocal_rms)

        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offsets < n_cols
            values = tl.load(input_row + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            output = values * reciprocal_rms * weight
            tl.store(output_row + col_offsets, output, mask=col_mask)


@libentry()
@triton.jit
def _rmsnorm_dx_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    rstd_ptr,
    grad_input_ptr,
    grad_output_row_stride,
    input_row_stride,
    grad_input_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_idx in range(program_id, n_rows, num_programs):
        row_idx_i64 = row_idx.to(tl.int64)
        grad_output_row = grad_output_ptr + row_idx_i64 * grad_output_row_stride
        input_row = input_ptr + row_idx_i64 * input_row_stride
        grad_input_row = grad_input_ptr + row_idx_i64 * grad_input_row_stride
        reciprocal_rms = tl.load(rstd_ptr + row_idx_i64).to(tl.float32)

        projection_sum = tl.zeros((1,), dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offsets < n_cols
            grad_output = tl.load(grad_output_row + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            values = tl.load(input_row + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            normalized = values * reciprocal_rms
            weighted_grad = grad_output * weight
            projection_sum += tl.sum(tl.where(col_mask, weighted_grad * normalized, 0.0))

        projection_mean = projection_sum / n_cols
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offsets < n_cols
            grad_output = tl.load(grad_output_row + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            values = tl.load(input_row + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            normalized = values * reciprocal_rms
            weighted_grad = grad_output * weight
            grad_input = reciprocal_rms * (weighted_grad - normalized * projection_mean)
            tl.store(grad_input_row + col_offsets, grad_input, mask=col_mask)


@libentry()
@triton.jit
def _rmsnorm_dw_partial_kernel(
    grad_output_ptr,
    input_ptr,
    rstd_ptr,
    partial_grad_weight_ptr,
    grad_output_row_stride,
    input_row_stride,
    partial_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Accumulate one deterministic, atomics-free row bucket per program."""

    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        partial = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # Each program owns rows program_id, program_id + num_programs, ... .
        # Both this order and num_programs are fixed for a given NPU target.
        for row_idx in range(program_id, n_rows, num_programs):
            row_idx_i64 = row_idx.to(tl.int64)
            grad_output = tl.load(
                grad_output_ptr + row_idx_i64 * grad_output_row_stride + col_offsets,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.load(
                input_ptr + row_idx_i64 * input_row_stride + col_offsets,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            reciprocal_rms = tl.load(rstd_ptr + row_idx_i64).to(tl.float32)
            normalized = values * reciprocal_rms
            partial += grad_output * normalized

        tl.store(
            partial_grad_weight_ptr + program_id * partial_row_stride + col_offsets,
            partial,
            mask=col_mask,
        )


@libentry()
@triton.jit
def _rmsnorm_dw_reduce_kernel(
    partial_grad_weight_ptr,
    grad_weight_ptr,
    partial_row_stride,
    num_partials,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce partial wgrad rows in an explicit ascending order."""

    col_task_id = tl.program_id(0)
    num_col_tasks = tl.num_programs(0)
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for col_block_id in range(col_task_id, num_col_blocks, num_col_tasks):
        col_offsets = col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols
        grad_weight = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for partial_idx in range(0, num_partials):
            partial = tl.load(
                partial_grad_weight_ptr + partial_idx * partial_row_stride + col_offsets,
                mask=col_mask,
                other=0.0,
            )
            grad_weight += partial
        tl.store(grad_weight_ptr + col_offsets, grad_weight, mask=col_mask)


def _validate_inputs(input: torch.Tensor, weight: torch.Tensor) -> None:
    if input.ndim < 1:
        raise ValueError("RMSNorm input must have at least one dimension")
    if input.device.type != "npu" or weight.device != input.device:
        raise ValueError("RMSNorm input and weight must be on the same NPU")
    if weight.ndim != 1:
        raise ValueError(f"RMSNorm weight must be 1D, got shape {tuple(weight.shape)}")
    if input.shape[-1] == 0:
        raise ValueError("RMSNorm hidden size must be positive")
    if input.shape[-1] != weight.numel():
        raise ValueError(f"RMSNorm input/weight hidden dimensions differ: {input.shape[-1]} != {weight.numel()}")
    if input.dtype not in _SUPPORTED_DTYPES or weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"RMSNorm supports float16, bfloat16, and float32 input/weight; got {input.dtype} and {weight.dtype}"
        )


def rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run VeXact-style RMSNorm and return output plus fp32 reciprocal RMS."""

    require_deterministic_mode()
    _validate_inputs(input, weight)

    original_shape = input.shape
    n_cols = original_shape[-1]
    input_2d = input.reshape(-1, n_cols).contiguous()
    weight_contiguous = weight.contiguous()
    n_rows = input_2d.shape[0]
    output_2d = torch.empty_like(input_2d)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=input.device)

    if n_rows == 0:
        return output_2d.reshape(original_shape), rstd

    num_programs = get_num_cores("vector")
    _rmsnorm_fwd_kernel[(num_programs,)](
        input_2d,
        weight_contiguous,
        output_2d,
        rstd,
        input_2d.stride(0),
        output_2d.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=_REDUCTION_BLOCK_SIZE,
    )
    return output_2d.reshape(original_shape), rstd


def rmsnorm_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute row-independent dX and deterministic full-batch dW.

    dW is deliberately specified as run-to-run deterministic, not
    mathematically batch-invariant: changing the batch changes the set and
    order of contributions to a shared parameter gradient.
    """

    require_deterministic_mode()
    _validate_inputs(input, weight)
    if grad_output.shape != input.shape:
        raise ValueError(f"grad_output shape {tuple(grad_output.shape)} != input shape {tuple(input.shape)}")

    original_shape = input.shape
    n_cols = original_shape[-1]
    input_2d = input.reshape(-1, n_cols).contiguous()
    grad_output_2d = grad_output.reshape(-1, n_cols).contiguous()
    weight_contiguous = weight.contiguous()
    rstd_contiguous = rstd.reshape(-1).contiguous()
    n_rows = input_2d.shape[0]
    if rstd_contiguous.numel() != n_rows:
        raise ValueError(f"rstd has {rstd_contiguous.numel()} rows, expected {n_rows}")

    grad_input_2d = torch.empty_like(input_2d)
    if n_rows == 0:
        return grad_input_2d.reshape(original_shape), torch.zeros_like(weight)

    num_programs = get_num_cores("vector")
    _rmsnorm_dx_kernel[(num_programs,)](
        grad_output_2d,
        input_2d,
        weight_contiguous,
        rstd_contiguous,
        grad_input_2d,
        grad_output_2d.stride(0),
        input_2d.stride(0),
        grad_input_2d.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=_REDUCTION_BLOCK_SIZE,
    )

    partial_grad_weight = torch.empty((num_programs, n_cols), dtype=torch.float32, device=weight.device)
    _rmsnorm_dw_partial_kernel[(num_programs,)](
        grad_output_2d,
        input_2d,
        rstd_contiguous,
        partial_grad_weight,
        grad_output_2d.stride(0),
        input_2d.stride(0),
        partial_grad_weight.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=_REDUCTION_BLOCK_SIZE,
    )

    grad_weight = torch.empty_like(weight_contiguous)
    num_col_blocks = triton.cdiv(n_cols, _REDUCTION_BLOCK_SIZE)
    _rmsnorm_dw_reduce_kernel[(min(num_programs, num_col_blocks),)](
        partial_grad_weight,
        grad_weight,
        partial_grad_weight.stride(0),
        num_programs,
        n_cols,
        BLOCK_SIZE=_REDUCTION_BLOCK_SIZE,
    )
    return grad_input_2d.reshape(original_shape), grad_weight.reshape(weight.shape)


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inference-only convenience wrapper for batch-invariant RMSNorm."""

    output, _ = rmsnorm_fwd(input, weight, eps)
    return output


rms_norm_fwd = rmsnorm_fwd
rms_norm_bwd = rmsnorm_bwd
rms_norm_infer_fwd = rmsnorm


def group_rms_norm_infer_fwd(input_groups, weight, eps):
    # Q/K heads use the same row reductions as the separate training norms.
    return [
        rmsnorm(
            x, weight[index] if weight is not None else torch.ones(x.shape[-1], dtype=x.dtype, device=x.device), eps
        )
        for index, x in enumerate(input_groups)
    ]


__all__ = [
    "rmsnorm",
    "rmsnorm_bwd",
    "rmsnorm_fwd",
    "rms_norm_fwd",
    "rms_norm_bwd",
    "rms_norm_infer_fwd",
    "group_rms_norm_infer_fwd",
]
