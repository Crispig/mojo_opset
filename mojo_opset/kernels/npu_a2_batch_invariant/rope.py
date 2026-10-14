import torch
import triton
import triton.language as tl

from mojo_opset.kernels.torch_reference.rope_cos_sin import rope_cos_sin_fwd as _device_rope_cos_sin

from .utils import libentry
from .utils import require_deterministic_mode


@libentry()
@triton.jit
def _rope_forward_kernel(
    x,
    cos,
    sin,
    output,
    stride_x_token,
    stride_x_head,
    stride_x_dim,
    stride_cos_token,
    stride_cos_dim,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_head_idx = tl.program_id(0)
    token_idx = token_head_idx // num_heads
    head_idx = token_head_idx % num_heads
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_offsets < head_dim

    rotary_start: tl.constexpr = head_dim - rope_dim
    rotary_offsets = dim_offsets - rotary_start
    rotary_mask = dim_mask & (rotary_offsets >= 0)
    half_rope_dim: tl.constexpr = rope_dim // 2
    first_half = rotary_offsets < half_rope_dim
    partner_offsets = tl.where(
        first_half,
        dim_offsets + half_rope_dim,
        dim_offsets - half_rope_dim,
    )

    x_base = x + token_idx * stride_x_token + head_idx * stride_x_head
    values = tl.load(x_base + dim_offsets * stride_x_dim, mask=dim_mask, other=0.0).to(tl.float32)
    partners = tl.load(
        x_base + partner_offsets * stride_x_dim,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    cos_values = tl.load(
        cos + token_idx * stride_cos_token + rotary_offsets * stride_cos_dim,
        mask=rotary_mask,
        other=1.0,
    ).to(tl.float32)
    sin_values = tl.load(
        sin + token_idx * stride_cos_token + rotary_offsets * stride_cos_dim,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = values * cos_values + tl.where(first_half, -partners, partners) * sin_values
    result = tl.where(rotary_mask, rotated, values)

    output_base = output + token_idx * stride_x_token + head_idx * stride_x_head
    tl.store(output_base + dim_offsets * stride_x_dim, result, mask=dim_mask)


@libentry()
@triton.jit
def _rope_backward_kernel(
    grad_output,
    cos,
    sin,
    grad_input,
    stride_x_token,
    stride_x_head,
    stride_x_dim,
    stride_cos_token,
    stride_cos_dim,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_head_idx = tl.program_id(0)
    token_idx = token_head_idx // num_heads
    head_idx = token_head_idx % num_heads
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_offsets < head_dim

    rotary_start: tl.constexpr = head_dim - rope_dim
    rotary_offsets = dim_offsets - rotary_start
    rotary_mask = dim_mask & (rotary_offsets >= 0)
    half_rope_dim: tl.constexpr = rope_dim // 2
    first_half = rotary_offsets < half_rope_dim
    partner_offsets = tl.where(
        first_half,
        dim_offsets + half_rope_dim,
        dim_offsets - half_rope_dim,
    )
    partner_rotary_offsets = partner_offsets - rotary_start

    grad_base = grad_output + token_idx * stride_x_token + head_idx * stride_x_head
    grad_values = tl.load(
        grad_base + dim_offsets * stride_x_dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    grad_partners = tl.load(
        grad_base + partner_offsets * stride_x_dim,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    cos_values = tl.load(
        cos + token_idx * stride_cos_token + rotary_offsets * stride_cos_dim,
        mask=rotary_mask,
        other=1.0,
    ).to(tl.float32)
    partner_sin = tl.load(
        sin + token_idx * stride_cos_token + partner_rotary_offsets * stride_cos_dim,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    rotated_grad = grad_values * cos_values + tl.where(
        first_half,
        grad_partners * partner_sin,
        -grad_partners * partner_sin,
    )
    result = tl.where(rotary_mask, rotated_grad, grad_values)

    grad_input_base = grad_input + token_idx * stride_x_token + head_idx * stride_x_head
    tl.store(grad_input_base + dim_offsets * stride_x_dim, result, mask=dim_mask)


def _to_token_major(x: torch.Tensor, head_first: bool) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim == 3:
        if head_first:
            normalized = x.permute(1, 0, 2).contiguous()
        else:
            normalized = x.contiguous()
        return normalized, tuple(x.shape)
    if x.ndim == 4:
        if head_first:
            normalized = x.permute(0, 2, 1, 3).contiguous()
        else:
            normalized = x.contiguous()
        batch_size, seq_len, num_heads, head_dim = normalized.shape
        return normalized.reshape(batch_size * seq_len, num_heads, head_dim), tuple(x.shape)
    raise ValueError(f"RoPE expects a 3D or 4D tensor, got shape {tuple(x.shape)}")


def _from_token_major(x: torch.Tensor, shape: tuple[int, ...], head_first: bool) -> torch.Tensor:
    if len(shape) == 3:
        return x.permute(1, 0, 2).contiguous() if head_first else x
    if head_first:
        batch_size, num_heads, seq_len, head_dim = shape
        return x.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return x.reshape(shape)


def _normalize_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_count: int,
    original_shape: tuple[int, ...],
    head_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shapes differ: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.ndim == 2:
        if cos.shape[0] == token_count:
            return cos.contiguous(), sin.contiguous()
        if len(original_shape) == 4:
            batch_size = original_shape[0]
            seq_len = original_shape[2] if head_first else original_shape[1]
            if cos.shape[0] == seq_len:
                expanded_shape = (batch_size, seq_len, cos.shape[-1])
                return (
                    cos.unsqueeze(0).expand(expanded_shape).contiguous().reshape(token_count, -1),
                    sin.unsqueeze(0).expand(expanded_shape).contiguous().reshape(token_count, -1),
                )
    elif cos.ndim == 3 and len(original_shape) == 4:
        if cos.shape[0] * cos.shape[1] == token_count:
            return cos.contiguous().reshape(token_count, -1), sin.contiguous().reshape(token_count, -1)
    raise ValueError(f"cos/sin shape {tuple(cos.shape)} cannot describe {token_count} token positions")


def _launch_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backward: bool,
) -> torch.Tensor:
    token_count, num_heads, head_dim = x.shape
    rope_dim = cos.shape[-1]
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError(f"rope_dim must be positive, even, and <= head_dim; got {rope_dim} and {head_dim}")
    if x.device != cos.device or x.device != sin.device:
        raise ValueError("x, cos, and sin must be on the same device")

    output = torch.empty_like(x)
    if output.numel() == 0:
        return output
    kernel = _rope_backward_kernel if backward else _rope_forward_kernel
    kernel[(token_count * num_heads,)](
        x,
        cos,
        sin,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        BLOCK_SIZE=triton.next_power_of_2(head_dim),
    )
    return output


def _apply_rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    require_deterministic_mode()
    if cos.requires_grad or sin.requires_grad:
        raise NotImplementedError("batch_invariant RoPE treats cos and sin as constant buffers.")
    q_normalized, q_shape = _to_token_major(q, head_first)
    k_normalized, k_shape = _to_token_major(k, head_first)
    if q_normalized.shape[0] != k_normalized.shape[0]:
        raise ValueError("q and k must describe the same token positions")
    cos, sin = _normalize_cos_sin(cos, sin, q_normalized.shape[0], q_shape, head_first)
    return (
        _from_token_major(_launch_rope(q_normalized, cos, sin, backward=False), q_shape, head_first),
        _from_token_major(_launch_rope(k_normalized, cos, sin, backward=False), k_shape, head_first),
    )


def rope_cos_sin_fwd(x, inv_freq, cos, sin, cu_q_lens, total_seq_lens, position_ids, attention_scaling):
    require_deterministic_mode()
    # Ignore precomputed CPU tables: HF evaluates the trigonometric functions
    # on the execution device and rounds cos/sin to the hidden-state dtype.
    with torch.autocast(device_type=x.device.type, enabled=False):
        cos, sin = _device_rope_cos_sin(
            x, inv_freq.float().to(x.device), None, None, cu_q_lens, total_seq_lens, position_ids, attention_scaling
        )
    return cos.to(x.dtype), sin.to(x.dtype)


def rope_infer_fwd(q, k, cos, sin, head_first, keep_cos_sin_dtype):
    if keep_cos_sin_dtype:
        raise NotImplementedError("Batch-invariant RoPE requires cos/sin rounded to the activation dtype")
    return _apply_rope_forward(q, k, cos.to(q.dtype), sin.to(q.dtype), head_first)


def rope_fwd(q, k, cos, sin, unsqueeze_dim):
    return _apply_rope_forward(q, k, cos, sin, head_first=unsqueeze_dim % q.ndim == q.ndim - 3)


def rope_bwd(grad_q, grad_k, cos, sin, unsqueeze_dim):
    head_first = unsqueeze_dim % grad_q.ndim == grad_q.ndim - 3
    q, q_shape = _to_token_major(grad_q, head_first)
    k, k_shape = _to_token_major(grad_k, head_first)
    cos, sin = _normalize_cos_sin(cos, sin, q.shape[0], q_shape, head_first)
    return (
        _from_token_major(_launch_rope(q, cos, sin, backward=True), q_shape, head_first),
        _from_token_major(_launch_rope(k, cos, sin, backward=True), k_shape, head_first),
    )


__all__ = ["rope_fwd", "rope_bwd", "rope_infer_fwd", "rope_cos_sin_fwd"]
