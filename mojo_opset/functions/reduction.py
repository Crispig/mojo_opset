from math import prod
from typing import Optional
from typing import Sequence
from typing import Union

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ReductionDim = Optional[Union[int, Sequence[int]]]


class ReductionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, op_id, output_dtype, implementation):
        forward, backward = load_impl(op_id, implementation, require_backward=True)
        output = forward(input_2d, output_dtype)
        ctx.backward_kernel = backward
        ctx.op_id = op_id
        ctx.input_shape = input_2d.shape
        ctx.input_dtype = input_2d.dtype
        if op_id == "amax":
            ctx.save_for_backward(input_2d, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        if ctx.op_id == "amax":
            grad_input = ctx.backward_kernel(grad_output, *ctx.saved_tensors)
        else:
            grad_input = ctx.backward_kernel(grad_output, ctx.input_shape, ctx.input_dtype)
        return grad_input, None, None, None


def _normalize_dims(ndim: int, dim: ReductionDim) -> tuple[int, ...]:
    effective_ndim = max(ndim, 1)
    if dim is None:
        return tuple(range(effective_ndim))
    if isinstance(dim, int):
        requested_dims = (dim,)
    else:
        requested_dims = tuple(dim)
        # PyTorch treats an empty dim list as an all-dim reduction.
        if not requested_dims:
            return tuple(range(effective_ndim))

    normalized_dims = []
    for axis in requested_dims:
        if not isinstance(axis, int):
            raise TypeError(f"reduction dimensions must be integers, got {type(axis).__name__}")
        if axis < -effective_ndim or axis >= effective_ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-effective_ndim}, {effective_ndim - 1}], but got {axis})"
            )
        normalized_axis = axis % effective_ndim
        if normalized_axis in normalized_dims:
            raise RuntimeError(f"dim {normalized_axis} appears multiple times in the list of dims")
        normalized_dims.append(normalized_axis)

    # Reduction order is a semantic property, not an artifact of tuple order.
    return tuple(sorted(normalized_dims))


def _canonicalize_reduction(
    input: torch.Tensor,
    dims: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if input.ndim == 0:
        return input.reshape(1, 1), (), ()

    reduced = set(dims)
    kept_dims = tuple(axis for axis in range(input.ndim) if axis not in reduced)
    permutation = kept_dims + dims
    canonical = input.permute(permutation) if permutation != tuple(range(input.ndim)) else input

    kept_shape = tuple(input.shape[axis] for axis in kept_dims)
    reduced_shape = tuple(input.shape[axis] for axis in dims)
    n_rows = prod(kept_shape) if kept_shape else 1
    n_cols = prod(reduced_shape) if reduced_shape else 1
    return canonical.contiguous().reshape(n_rows, n_cols), kept_shape, tuple(input.shape)


def _restore_reduction_shape(
    output: torch.Tensor,
    input_shape: tuple[int, ...],
    dims: tuple[int, ...],
    kept_shape: tuple[int, ...],
    keepdim: bool,
) -> torch.Tensor:
    if not input_shape:
        return output.reshape(())
    if not keepdim:
        return output.reshape(kept_shape)

    reduced = set(dims)
    output_shape = tuple(1 if axis in reduced else size for axis, size in enumerate(input_shape))
    return output.reshape(output_shape)


def _reduce(input, dim, keepdim, dtype, op_id, implementation):
    output_dtype = input.dtype if dtype is None else dtype
    if input.dtype not in _SUPPORTED_DTYPES or output_dtype not in _SUPPORTED_DTYPES:
        raise TypeError("reductions support float16, bfloat16, and float32 inputs/outputs")
    dims = _normalize_dims(input.ndim, dim)
    compute_input = input if input.dtype == output_dtype else input.to(output_dtype)
    input_2d, kept_shape, input_shape = _canonicalize_reduction(compute_input, dims)
    output = ReductionFunction.apply(input_2d, op_id, output_dtype, implementation)
    return _restore_reduction_shape(output, input_shape, dims, kept_shape, keepdim)


def reduce_sum(input, dim=None, keepdim=False, dtype=None, *, implementation=None):
    """Sum the selected dimensions using the chosen provider's reduction order."""
    return _reduce(input, dim, keepdim, dtype, "reduce_sum", implementation)


def mean(input, dim=None, keepdim=False, dtype=None, *, implementation=None):
    """Mean over selected dimensions, with first-order gradients."""
    return _reduce(input, dim, keepdim, dtype, "mean", implementation)


def amax(input, dim=None, keepdim=False, *, implementation=None):
    """Maximum over selected dimensions, with first-order gradients."""
    return _reduce(input, dim, keepdim, None, "amax", implementation)
