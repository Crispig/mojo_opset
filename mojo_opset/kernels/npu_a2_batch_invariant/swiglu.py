import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .utils import get_num_cores

_DTYPES = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def _swiglu_fwd_kernel(
    x1,
    x2,
    scales,
    output,
    count,
    columns,
    PRODUCT_DTYPE: tl.constexpr,
    HAS_SCALES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    for task in range(tl.program_id(0), tl.cdiv(count, BLOCK_SIZE), tl.num_programs(0)):
        offsets = task * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < count
        first = tl.load(x1 + offsets, mask, other=0.0).to(tl.float32)
        second = tl.load(x2 + offsets, mask, other=0.0).to(tl.float32)
        # Keep both HF dtype roundings despite fusing the launches.
        silu = (first * tl.sigmoid(first)).to(x1.dtype.element_ty).to(tl.float32)
        result = (silu * second).to(PRODUCT_DTYPE)
        if HAS_SCALES:
            scale = tl.load(scales + offsets // columns, mask, other=0.0).to(tl.float32)
            result = result.to(tl.float32) * scale
        tl.store(output + offsets, result, mask)


def swiglu_fwd(x1, x2, scales=None):
    inputs = (x1, x2) if scales is None else (x1, x2, scales)
    if x1.device.type != "npu" or any(value.device != x1.device for value in inputs):
        raise ValueError("batch_invariant SwiGLU inputs must be on the same NPU")
    if any(value.dtype not in _DTYPES for value in inputs):
        raise TypeError("batch_invariant SwiGLU supports float16, bfloat16, and float32")
    product_shape = torch.broadcast_shapes(x1.shape, x2.shape)
    product_dtype = torch.result_type(x1, x2)
    output_shape, output_dtype = product_shape, product_dtype
    if scales is not None:
        scale_view = scales.reshape(*x1.shape[:-1], 1)
        output_shape = torch.broadcast_shapes(product_shape, scale_view.shape)
        output_dtype = torch.result_type(torch.empty(product_shape, dtype=product_dtype, device="meta"), scale_view)
    output = torch.empty(output_shape, dtype=output_dtype, device=x1.device)
    if output.numel():
        if scales is not None:
            scales = scale_view.expand(output_shape)[..., 0].contiguous().reshape(-1)
        _swiglu_fwd_kernel[(min(triton.cdiv(output.numel(), 4096), get_num_cores("vector")),)](
            x1.expand(output_shape).contiguous(),
            x2.expand(output_shape).contiguous(),
            scales,
            output,
            output.numel(),
            output_shape[-1] if output_shape else 1,
            PRODUCT_DTYPE=_DTYPES[product_dtype],
            HAS_SCALES=scales is not None,
            BLOCK_SIZE=4096,
        )
    return output


def swiglu_bwd(grad_output, x1, x2, scales=None):
    silu = F.silu(x1)
    grad_scales = None
    if scales is not None:
        base = silu * x2
        scale_view = scales.reshape(*x1.shape[:-1], 1)
        grad_scales = (grad_output * base).sum_to_size(scale_view.shape).reshape(scales.shape).to(scales.dtype)
        grad_output = (grad_output * scale_view).to(base.dtype)
    # Preserve the two multiplication gradients' dtype rounding before the
    # native SiLU backward, matching HF's unfused BF16 autograd path.
    grad_x1 = torch.ops.aten.silu_backward((grad_output * x2).to(x1.dtype), x1)
    grad_x2 = (grad_output * silu).to(x2.dtype)
    return grad_x1.contiguous(), grad_x2.contiguous(), grad_scales
