import pytest
import torch

from mojo_opset.functions.rope import rope

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def apply_rope(q, k, cos, sin, head_first=True):
    return rope(
        q,
        k,
        cos,
        sin,
        unsqueeze_dim=q.ndim - (3 if head_first else 2),
        implementation="batch_invariant",
    )


def _reference(x, cos, sin):
    rope_dim = cos.shape[-1]
    passthrough, rotary = torch.split(x, [x.shape[-1] - rope_dim, rope_dim], dim=-1)
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((passthrough, (rotary.float() * cos + rotated.float() * sin).to(x.dtype)), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_first", [False, True])
def test_rope_3d_accuracy_batch_invariance_and_backward(dtype, head_first):
    torch.manual_seed(23)
    device = torch.device("npu")
    tokens, q_heads, k_heads, head_dim, rope_dim = 7, 4, 2, 16, 12
    q_shape = (q_heads, tokens, head_dim) if head_first else (tokens, q_heads, head_dim)
    k_shape = (k_heads, tokens, head_dim) if head_first else (tokens, k_heads, head_dim)
    q = torch.randn(q_shape, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(k_shape, dtype=dtype, device=device, requires_grad=True)
    angles = torch.randn(tokens, rope_dim // 2, dtype=torch.float32, device=device)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)

    q_out, k_out = apply_rope(q, k, cos, sin, head_first)
    cos_view = cos[:, None, :] if not head_first else cos[None, :, :]
    sin_view = sin[:, None, :] if not head_first else sin[None, :, :]
    torch.testing.assert_close(q_out, _reference(q, cos_view, sin_view), rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_out, _reference(k, cos_view, sin_view), rtol=0.0, atol=0.0)

    token_dim = 1 if head_first else 0
    permutation = torch.tensor([5, 1, 6, 0, 3, 2, 4], device=device)
    q_permuted, k_permuted = apply_rope(
        q.index_select(token_dim, permutation),
        k.index_select(token_dim, permutation),
        cos[permutation],
        sin[permutation],
        head_first,
    )
    assert torch.equal(q_permuted, q_out.index_select(token_dim, permutation))
    assert torch.equal(k_permuted, k_out.index_select(token_dim, permutation))

    grad_q = torch.randn_like(q_out)
    grad_k = torch.randn_like(k_out)
    torch.autograd.backward((q_out, k_out), (grad_q, grad_k))
    q_grad = q.grad.detach().clone()
    k_grad = k.grad.detach().clone()
    q.grad = None
    k.grad = None
    torch.autograd.backward(
        (q_permuted, k_permuted),
        (grad_q.index_select(token_dim, permutation), grad_k.index_select(token_dim, permutation)),
    )
    # index_select's backward restores the original token order.
    assert torch.equal(q.grad, q_grad)
    assert torch.equal(k.grad, k_grad)


def test_rope_4d_broadcast_positions():
    device = torch.device("npu")
    batch, heads, seq_len, head_dim = 3, 2, 5, 8
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    angles = torch.randn(seq_len, head_dim // 2, device=device)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    q_out, k_out = apply_rope(q, k, cos, sin, head_first=True)
    cos_view = cos[None, None, :, :]
    sin_view = sin[None, None, :, :]
    assert torch.equal(q_out, _reference(q, cos_view, sin_view))
    assert torch.equal(k_out, _reference(k, cos_view, sin_view))


def test_rope_rejects_cos_sin_gradients_instead_of_silently_dropping_them():
    q = torch.randn(2, 3, 8, device="npu", dtype=torch.float32)
    k = torch.randn_like(q)
    cos = torch.randn(2, 8, device="npu", dtype=torch.float32, requires_grad=True)
    sin = torch.randn(2, 8, device="npu", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="constant buffers"):
        apply_rope(q, k, cos, sin, head_first=False)


def test_rope_accepts_an_empty_token_batch():
    q = torch.empty(0, 3, 8, device="npu", dtype=torch.bfloat16, requires_grad=True)
    k = torch.empty(0, 2, 8, device="npu", dtype=torch.bfloat16, requires_grad=True)
    cos = torch.empty(0, 8, device="npu", dtype=torch.float32)
    sin = torch.empty_like(cos)

    q_output, k_output = apply_rope(q, k, cos, sin, head_first=False)
    assert q_output.shape == q.shape
    assert k_output.shape == k.shape
    (q_output.sum() + k_output.sum()).backward()
    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
