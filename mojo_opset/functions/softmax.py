from math import prod

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, op_id, output_dtype, implementation):
        forward, backward = load_impl(op_id, implementation, require_backward=True)
        output = forward(input_2d, output_dtype)
        ctx.backward_kernel = backward
        ctx.input_dtype = input_2d.dtype
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = ctx.backward_kernel(grad_output.contiguous(), output, ctx.input_dtype)
        return grad_input, None, None, None


def _normalize_dim(ndim: int, dim: int) -> int:
    if not isinstance(dim, int):
        raise TypeError(f"softmax dim must be an integer, got {type(dim).__name__}")
    effective_ndim = max(ndim, 1)
    if dim < -effective_ndim or dim >= effective_ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-effective_ndim}, {effective_ndim - 1}], but got {dim})"
        )
    return dim % effective_ndim


def _softmax(input, dim, dtype, op_id, implementation):
    output_dtype = input.dtype if dtype is None else dtype
    if input.dtype not in _SUPPORTED_DTYPES or output_dtype not in _SUPPORTED_DTYPES:
        raise TypeError("softmax supports float16, bfloat16, and float32 inputs/outputs")
    normalized_dim = _normalize_dim(input.ndim, dim)
    compute_input = input if input.dtype == output_dtype else input.to(output_dtype)
    if input.ndim == 0:
        return SoftmaxFunction.apply(compute_input.reshape(1, 1), op_id, output_dtype, implementation).reshape(())
    canonical = compute_input.movedim(normalized_dim, -1).contiguous()
    canonical_shape = canonical.shape
    input_2d = canonical.reshape(prod(canonical_shape[:-1]), canonical_shape[-1])
    output = SoftmaxFunction.apply(input_2d, op_id, output_dtype, implementation).reshape(canonical_shape)
    if normalized_dim != input.ndim - 1:
        output = output.movedim(-1, normalized_dim).contiguous()
    return output


def softmax(input, dim=-1, dtype=None, *, implementation=None):
    """Softmax with provider-controlled reduction and first-order gradients."""
    return _softmax(input, dim, dtype, "softmax", implementation)


def log_softmax(input, dim=-1, dtype=None, *, implementation=None):
    """Log-softmax with provider-controlled reduction and first-order gradients."""
    return _softmax(input, dim, dtype, "log_softmax", implementation)
