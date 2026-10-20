import torch
import triton
import triton.language as tl

from mojo_opset.kernels.torch_reference.rope_cos_sin import rope_cos_sin_fwd as _device_rope_cos_sin

from .utils import get_num_cores
from .utils import libentry
from .utils import require_deterministic_mode


@triton.jit
def _rotate_heads(
    x,
    cos,
    sin,
    output,
    first_token,
    first_head,
    token_count,
    stride_cos_token,
    stride_cos_dim,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    HALF_BLOCK_SIZE: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,
):
    token_offsets = first_token + tl.arange(0, BLOCK_TOKENS)
    head_offsets = first_head + tl.arange(0, BLOCK_HEADS)
    half_offsets = tl.arange(0, HALF_BLOCK_SIZE)
    rotary_start: tl.constexpr = head_dim - rope_dim
    half_rope_dim: tl.constexpr = rope_dim // 2
    mask = (
        (token_offsets[:, None, None] < token_count)
        & (head_offsets[None, :, None] < num_heads)
        & (half_offsets[None, None, :] < half_rope_dim)
    )
    offsets = (
        token_offsets[:, None, None] * num_heads * head_dim
        + head_offsets[None, :, None] * head_dim
        + rotary_start
        + half_offsets[None, None, :]
    )
    first = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(x + offsets + half_rope_dim, mask=mask, other=0.0).to(tl.float32)
    table_offsets = token_offsets[:, None] * stride_cos_token + half_offsets[None, :] * stride_cos_dim
    table_mask = (token_offsets[:, None] < token_count) & (half_offsets[None, :] < half_rope_dim)
    cos_first = tl.load(cos + table_offsets, mask=table_mask, other=1.0).to(tl.float32)[:, None, :]
    cos_second = tl.load(cos + table_offsets + half_rope_dim * stride_cos_dim, mask=table_mask, other=1.0).to(
        tl.float32
    )[:, None, :]
    sin_first = tl.load(sin + table_offsets, mask=table_mask, other=0.0).to(tl.float32)[:, None, :]
    sin_second = tl.load(sin + table_offsets + half_rope_dim * stride_cos_dim, mask=table_mask, other=0.0).to(
        tl.float32
    )[:, None, :]
    if BACKWARD:
        rotated_first = first * cos_first + second * sin_second
        rotated_second = second * cos_second + -first * sin_first
    else:
        rotated_first = first * cos_first + -second * sin_first
        rotated_second = second * cos_second + first * sin_second
    tl.store(output + offsets, rotated_first, mask=mask)
    tl.store(output + offsets + half_rope_dim, rotated_second, mask=mask)
    if rotary_start:
        dims = tl.arange(0, HEAD_BLOCK_SIZE)
        prefix_offsets = (
            token_offsets[:, None, None] * num_heads * head_dim
            + head_offsets[None, :, None] * head_dim
            + dims[None, None, :]
        )
        prefix_mask = (
            (token_offsets[:, None, None] < token_count)
            & (head_offsets[None, :, None] < num_heads)
            & (dims[None, None, :] < rotary_start)
        )
        prefix = tl.load(x + prefix_offsets, mask=prefix_mask, other=0.0)
        tl.store(output + prefix_offsets, prefix, mask=prefix_mask)


@libentry()
@triton.jit
def _rope_kernel(
    q,
    k,
    cos,
    sin,
    q_output,
    k_output,
    token_count,
    stride_cos_token,
    stride_cos_dim,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    q_head_dim: tl.constexpr,
    k_head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    Q_BLOCK_HEADS: tl.constexpr,
    K_BLOCK_HEADS: tl.constexpr,
    HALF_BLOCK_SIZE: tl.constexpr,
    Q_HEAD_BLOCK_SIZE: tl.constexpr,
    K_HEAD_BLOCK_SIZE: tl.constexpr,
):
    # Broadcast a token's tables over its independent heads, as in the A2 RoPE.
    for task in range(tl.program_id(0), tl.cdiv(token_count, BLOCK_TOKENS), tl.num_programs(0)):
        for head_block in range(tl.cdiv(q_heads, Q_BLOCK_HEADS)):
            _rotate_heads(
                q,
                cos,
                sin,
                q_output,
                task * BLOCK_TOKENS,
                head_block * Q_BLOCK_HEADS,
                token_count,
                stride_cos_token,
                stride_cos_dim,
                q_heads,
                q_head_dim,
                rope_dim,
                BACKWARD,
                BLOCK_TOKENS,
                Q_BLOCK_HEADS,
                HALF_BLOCK_SIZE,
                Q_HEAD_BLOCK_SIZE,
            )
        for head_block in range(tl.cdiv(k_heads, K_BLOCK_HEADS)):
            _rotate_heads(
                k,
                cos,
                sin,
                k_output,
                task * BLOCK_TOKENS,
                head_block * K_BLOCK_HEADS,
                token_count,
                stride_cos_token,
                stride_cos_dim,
                k_heads,
                k_head_dim,
                rope_dim,
                BACKWARD,
                BLOCK_TOKENS,
                K_BLOCK_HEADS,
                HALF_BLOCK_SIZE,
                K_HEAD_BLOCK_SIZE,
            )


@libentry()
@triton.jit
def _packed_positions_kernel(
    position_ids,
    cu_q_lens,
    total_seq_lens,
    tokens,
    cu_q_stride,
    total_seq_stride,
    sequences: tl.constexpr,
    HAS_CONTEXT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(tokens, BLOCK_SIZE), tl.num_programs(0)):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        positions = tl.full((BLOCK_SIZE,), -1, tl.int32)
        for sequence in range(sequences):
            start = tl.load(cu_q_lens + sequence * cu_q_stride)
            end = tl.load(cu_q_lens + (sequence + 1) * cu_q_stride)
            context = tl.load(total_seq_lens + sequence * total_seq_stride) - (end - start) if HAS_CONTEXT else 0
            positions = tl.where((offsets >= start) & (offsets < end), offsets - start + context, positions)
        tl.store(position_ids + offsets, positions, mask=offsets < tokens)


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
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backward: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count, q_heads, q_head_dim = q.shape
    _, k_heads, k_head_dim = k.shape
    rope_dim = cos.shape[-1]
    if rope_dim <= 0 or rope_dim > min(q_head_dim, k_head_dim) or rope_dim % 2:
        raise ValueError("rope_dim must be positive, even, and <= both Q/K head dimensions")
    if q.device != k.device or q.device != cos.device or q.device != sin.device:
        raise ValueError("q, k, cos, and sin must be on the same device")

    q_output, k_output = torch.empty_like(q), torch.empty_like(k)
    block_tokens = 2
    # Bound live FP32 tiles independently of the model's total head count.
    head_capacity = max(
        1, 8192 // block_tokens // max(triton.next_power_of_2(q_head_dim), triton.next_power_of_2(k_head_dim))
    )
    tasks = triton.cdiv(token_count, block_tokens)
    if tasks == 0:
        return q_output, k_output
    _rope_kernel[(min(tasks, get_num_cores("vector")),)](
        q,
        k,
        cos,
        sin,
        q_output,
        k_output,
        token_count,
        cos.stride(0),
        cos.stride(1),
        q_heads=q_heads,
        k_heads=k_heads,
        q_head_dim=q_head_dim,
        k_head_dim=k_head_dim,
        rope_dim=rope_dim,
        BACKWARD=backward,
        BLOCK_TOKENS=block_tokens,
        Q_BLOCK_HEADS=min(head_capacity, triton.next_power_of_2(q_heads)),
        K_BLOCK_HEADS=min(head_capacity, triton.next_power_of_2(k_heads)),
        HALF_BLOCK_SIZE=triton.next_power_of_2(rope_dim // 2),
        Q_HEAD_BLOCK_SIZE=triton.next_power_of_2(q_head_dim),
        K_HEAD_BLOCK_SIZE=triton.next_power_of_2(k_head_dim),
    )
    return q_output, k_output


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
    q_output, k_output = _launch_rope(q_normalized, k_normalized, cos, sin, backward=False)
    return (
        _from_token_major(q_output, q_shape, head_first),
        _from_token_major(k_output, k_shape, head_first),
    )


def rope_cos_sin_fwd(x, inv_freq, cos, sin, cu_q_lens, total_seq_lens, position_ids, attention_scaling):
    require_deterministic_mode()
    if cu_q_lens is not None:
        if x.ndim != 2:
            raise ValueError("packed prefill x must have shape [tokens, hidden]")
        if cu_q_lens.device != x.device or (total_seq_lens is not None and total_seq_lens.device != x.device):
            raise ValueError("Batch-invariant packed RoPE requires sequence metadata on x.device")
        tokens = x.shape[0]
        position_ids = torch.empty(tokens, dtype=torch.int32, device=x.device)
        if tokens:
            _packed_positions_kernel[(min(triton.cdiv(tokens, 256), get_num_cores("vector")),)](
                position_ids,
                cu_q_lens,
                total_seq_lens,
                tokens,
                cu_q_lens.stride(0),
                total_seq_lens.stride(0) if total_seq_lens is not None else 0,
                cu_q_lens.numel() - 1,
                HAS_CONTEXT=total_seq_lens is not None,
                BLOCK_SIZE=256,
            )
        cu_q_lens, total_seq_lens = None, None
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
    q_output, k_output = _launch_rope(q, k, cos, sin, backward=True)
    return (
        _from_token_major(q_output, q_shape, head_first),
        _from_token_major(k_output, k_shape, head_first),
    )


__all__ = ["rope_fwd", "rope_bwd", "rope_infer_fwd", "rope_cos_sin_fwd"]
