"""
Batch-invariant floating-point GEMM kernels for Ascend 910B.

The reduction dimension is visited in one fixed, serial ``BLOCK_K`` order and
every output tile owns its accumulator.  Consequently, changing or permuting
the M/batch dimensions cannot change an output element's reduction tree.

Weight and bias gradients reduce over the flattened training batch.  They are
deterministic for a fixed batch, but (as reductions over that batch) are not
claimed to be invariant when the training batch itself changes.
"""

import math

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from .utils import get_num_cores
from .utils import require_deterministic_mode


class _TileConfig(NamedTuple):
    block_m: int
    block_n: int
    block_k: int


# This is a source-controlled replacement for runtime autotuning.  The table is
# selected only by SoC (this a2 module), dtype, K and N. In particular, M must
# never participate in tile selection.
_A2_TILE_TABLE = {
    torch.float16: {
        "small_kn": _TileConfig(64, 64, 32),
        "small_n": _TileConfig(64, 64, 64),
        "default": _TileConfig(64, 128, 64),
    },
    torch.bfloat16: {
        "small_kn": _TileConfig(64, 64, 32),
        "small_n": _TileConfig(64, 64, 64),
        "default": _TileConfig(64, 128, 64),
    },
    torch.float32: {
        "small_kn": _TileConfig(32, 32, 16),
        "small_n": _TileConfig(32, 64, 32),
        "default": _TileConfig(64, 64, 32),
    },
}


def _select_tile(dtype: torch.dtype, k: int, n: int) -> _TileConfig:
    """Return the pinned a2 tile without consulting M, batch, or timings."""
    try:
        table = _A2_TILE_TABLE[dtype]
    except KeyError as exc:
        raise TypeError(f"batch_invariant GEMM supports only float32, float16, and bfloat16; got {dtype}.") from exc

    if k <= 32 and n <= 64:
        return table["small_kn"]
    if n <= 64:
        return table["small_n"]
    return table["default"]


@triton.jit
def _matmul_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    A_TRANSPOSED: tl.constexpr,
    B_TRANSPOSED: tl.constexpr,
):
    core_id = tl.program_id(axis=0)
    core_count = tl.num_programs(axis=0)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile_count = tiles_m * tiles_n

    for tile_id in range(core_id, tile_count, core_count):
        tile_m = tile_id // tiles_n
        tile_n = tile_id % tiles_n
        offs_m = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        mask_m = offs_m < M
        mask_n = offs_n < N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K):
            offs_k = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
            mask_k = offs_k < K
            if A_TRANSPOSED:
                a = tl.load(
                    a_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am,
                    mask=mask_k[:, None] & mask_m[None, :],
                    other=0.0,
                )
                a = tl.trans(a)
            else:
                a = tl.load(
                    a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
            if B_TRANSPOSED:
                b = tl.load(
                    b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
            accumulator = tl.dot(a, b, acc=accumulator)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            accumulator += bias[None, :]

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            accumulator.to(c_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _check_device_and_dtype(*tensors: torch.Tensor) -> None:
    first = tensors[0]
    if first.device.type != "npu":
        raise ValueError(f"batch_invariant GEMM requires NPU tensors; got {first.device}.")
    if first.dtype not in _A2_TILE_TABLE:
        _select_tile(first.dtype, 0, 0)
    for tensor in tensors[1:]:
        if tensor.device != first.device:
            raise ValueError(f"All GEMM operands must be on {first.device}; got {tensor.device}.")
        if tensor.dtype != first.dtype:
            raise TypeError(f"All GEMM operands must have dtype {first.dtype}; got {tensor.dtype}.")


def _matrix_layout(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Choose a cube-friendly logical layout without changing tensor values."""
    if tensor.stride(-1) == 1:
        return tensor, False
    if tensor.stride(-2) == 1:
        return tensor, True
    return tensor.contiguous(), False


def _matmul_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    require_deterministic_mode()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul_persistent expects two 2D tensors; got {a.ndim}D and {b.ndim}D.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible GEMM shapes: {tuple(a.shape)} and {tuple(b.shape)}.")

    operands = (a, b) if bias is None else (a, b, bias)
    _check_device_and_dtype(*operands)
    m, k = a.shape
    n = b.shape[1]
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != n):
        raise ValueError(f"bias must have shape ({n},); got {tuple(bias.shape)}.")

    output = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if m == 0 or n == 0:
        return output

    # Ascend's transposed-B dot epilogue miscompiles broadcast bias addition.
    if bias is not None:
        b = b.contiguous()
    a, a_transposed = _matrix_layout(a)
    b, b_transposed = _matrix_layout(b)
    tile = _select_tile(a.dtype, k, n)
    bias_ptr = output if bias is None else bias
    _matmul_persistent_kernel[(get_num_cores("cube"),)](
        a,
        b,
        output,
        bias_ptr,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        HAS_BIAS=bias is not None,
        A_TRANSPOSED=a_transposed,
        B_TRANSPOSED=b_transposed,
        multibuffer=True,
        sync_solver=False,
    )
    return output


def batch_invariant_matmul_bwd(grad_output, a, b, has_bias, needs_input_grad):
    grad_a = grad_b = grad_bias = None
    if needs_input_grad[0]:
        grad_a = _matmul_forward(grad_output, b.transpose(0, 1), None)
    if needs_input_grad[1]:
        # Weight gradients reduce over the batch: deterministic for a fixed batch,
        # not invariant when the reduction's input batch itself changes.
        grad_b = _matmul_forward(a.transpose(0, 1), grad_output, None)
    if has_bias and needs_input_grad[2]:
        reduction_width = grad_output.shape[0]
        ones = torch.ones((1, reduction_width), dtype=grad_output.dtype, device=grad_output.device)
        grad_bias = _matmul_forward(ones, grad_output, None).reshape(-1)
    return grad_a, grad_b, grad_bias


def batch_invariant_matmul_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``a @ b + bias`` with an M/batch-independent reduction tree."""
    return _matmul_forward(a, b, bias)


def batch_invariant_linear_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-invariant ``torch.nn.functional.linear`` for arbitrary leading dims."""
    if input.ndim < 1:
        raise ValueError("linear input must have at least one dimension.")
    if weight.ndim != 2:
        raise ValueError(f"linear weight must be 2D; got {weight.ndim}D.")
    in_features = input.shape[-1]
    if weight.shape[1] != in_features:
        raise ValueError(f"linear input feature size {in_features} does not match weight shape {tuple(weight.shape)}.")

    rows = math.prod(input.shape[:-1])
    input_2d = input.reshape(rows, in_features)
    output_2d = batch_invariant_matmul_fwd(input_2d, weight.transpose(0, 1), bias)
    return output_2d.reshape(*input.shape[:-1], weight.shape[0])


def batch_invariant_linear_bwd(grad_output, input, weight, has_bias, needs_input_grad):
    input_2d = input.reshape(math.prod(input.shape[:-1]), input.shape[-1])
    grad_2d = grad_output.reshape(input_2d.shape[0], weight.shape[0])
    grad_input, grad_weight, grad_bias = batch_invariant_matmul_bwd(
        grad_2d,
        input_2d,
        weight.transpose(0, 1),
        has_bias,
        needs_input_grad,
    )
    return (
        grad_input.reshape(input.shape) if grad_input is not None else None,
        grad_weight.transpose(0, 1) if grad_weight is not None else None,
        grad_bias,
    )


__all__ = [
    "batch_invariant_linear_fwd",
    "batch_invariant_linear_bwd",
    "batch_invariant_matmul_fwd",
    "batch_invariant_matmul_bwd",
]
