import torch
import torch.nn.functional as F


def swiglu_fwd(x1, x2, scales=None):
    # HF rounds SiLU to the activation dtype before multiplication. A fused
    # FP32 SiLU/product changes BF16 output bits even with deterministic GEMM.
    output = F.silu(x1) * x2
    if scales is not None:
        output = output * scales.reshape(*x1.shape[:-1], 1)
    return output.contiguous()


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
