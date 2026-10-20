from unittest.mock import patch

import pytest
import torch

from mojo_opset.functions.position_embedding import rope_infer
from mojo_opset.functions.rope import rope
from mojo_opset.modules import RoPECosSin

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _reference(x, cos, sin):
    rotary_start = x.shape[-1] - cos.shape[-1]
    first, second = x[..., rotary_start:].float().chunk(2, dim=-1)
    partners = torch.cat((-second, first), dim=-1)
    rotated = x[..., rotary_start:].float() * cos[:, None, :] + partners * sin[:, None, :]
    return torch.cat((x[..., :rotary_start], rotated.to(x.dtype)), dim=-1)


def _reference_backward(gradient, cos, sin):
    rotary_start = gradient.shape[-1] - cos.shape[-1]
    first, second = gradient[..., rotary_start:].float().chunk(2, dim=-1)
    first_sin, second_sin = sin[:, None, :].chunk(2, dim=-1)
    partners = torch.cat((second * second_sin, -first * first_sin), dim=-1)
    rotated = gradient[..., rotary_start:].float() * cos[:, None, :] + partners
    return torch.cat((gradient[..., :rotary_start], rotated.to(gradient.dtype)), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rope_dim", [96, 128])
def test_vectorized_rope_keeps_elementwise_math_and_gradients(dtype, rope_dim):
    torch.manual_seed(163)
    q = torch.randn(67, 32, 128, device="npu", dtype=dtype, requires_grad=True)
    k = torch.randn(67, 8, 128, device="npu", dtype=dtype, requires_grad=True)
    # Deliberately independent halves: training accepts arbitrary fixed tables.
    cos = torch.randn(67, rope_dim, device="npu", dtype=torch.float32)
    sin = torch.randn_like(cos)
    actual = rope(q, k, cos, sin, unsqueeze_dim=1, implementation="batch_invariant")
    expected = [_reference(x, cos, sin) for x in (q, k)]
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
    gradients = [torch.randn_like(x) for x in actual]
    actual_gradients = torch.autograd.grad(actual, (q, k), gradients)
    # The custom backward rounds once after both FP32 contributions, unlike
    # autograd through two independently cast BF16 input branches.
    expected_gradients = [_reference_backward(x, cos, sin) for x in gradients]
    for got, want in zip(actual_gradients, expected_gradients):
        assert torch.equal(got, want)


def test_large_rope_does_not_depend_on_token_head_launch_grid():
    torch.manual_seed(167)
    tokens = 4097
    q = torch.randn(tokens, 32, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(tokens, 8, 128, device="npu", dtype=torch.bfloat16)
    cos = torch.randn(tokens, 128, device="npu", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    actual = rope_infer(q, k, cos, sin, head_first=False, implementation="batch_invariant")
    for got, x in zip(actual, (q, k)):
        assert torch.equal(got, _reference(x, cos.float(), sin.float()))
    index = torch.tensor([4096, 17, 0, 2048, 63], device="npu")
    selected = rope_infer(
        q[index], k[index], cos[index], sin[index], head_first=False, implementation="batch_invariant"
    )
    for got, full in zip(selected, actual):
        assert torch.equal(got, full[index])


@pytest.mark.parametrize("heads,head_dim,rope_dim", [(64, 128, 128), (32, 88, 88), (32, 128, 48)])
def test_rope_tiles_wide_head_groups_and_partial_dimensions(heads, head_dim, rope_dim):
    q = torch.randn(17, heads, head_dim, device="npu", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(17, 8, head_dim, device="npu", dtype=torch.bfloat16, requires_grad=True)
    cos = torch.randn(17, rope_dim, device="npu")
    sin = torch.randn_like(cos)
    actual = rope(q, k, cos, sin, unsqueeze_dim=1, implementation="batch_invariant")
    for got, x in zip(actual, (q, k)):
        assert torch.equal(got, _reference(x, cos, sin))
    upstream = [torch.randn_like(x) for x in actual]
    gradients = torch.autograd.grad(actual, (q, k), upstream)
    for got, gradient in zip(gradients, upstream):
        assert torch.equal(got, _reference_backward(gradient, cos, sin))


@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_packed_rope_positions_stay_on_device_and_match_explicit_positions(with_context, strided):
    op = RoPECosSin(1000000.0, 128, implementation="batch_invariant", device="npu")
    x = torch.zeros(67, 128, device="npu", dtype=torch.bfloat16)
    cu = torch.tensor([0, 40, 67], device="npu", dtype=torch.int32)
    total = torch.tensor([61, 79], device="npu", dtype=torch.int32) if with_context else None
    if strided:
        cu = torch.tensor([0, -1, 40, -1, 67], device="npu", dtype=torch.int32)[::2]
        if total is not None:
            total = torch.tensor([61, -1, 79], device="npu", dtype=torch.int32)[::2]
    positions = torch.cat(
        (torch.arange(40) + (21 if with_context else 0), torch.arange(27) + (52 if with_context else 0))
    ).to(device="npu", dtype=torch.int32)
    expected = op(x, position_ids=positions)
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device metadata must not reach the host")):
        actual = op(x, cu_q_lens=cu, total_seq_lens=total)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("strided", [False, True])
def test_packed_rope_graph_replay_uses_current_device_metadata(strided):
    op = RoPECosSin(1000000.0, 128, implementation="batch_invariant", device="npu")
    x = torch.zeros(67, 128, device="npu", dtype=torch.bfloat16)
    cu = torch.tensor([0, 40, 67], device="npu", dtype=torch.int32)
    total = torch.tensor([61, 79], device="npu", dtype=torch.int32)
    if strided:
        cu = torch.tensor([0, -1, 40, -1, 67], device="npu", dtype=torch.int32)[::2]
        total = torch.tensor([61, -1, 79], device="npu", dtype=torch.int32)[::2]
    q = torch.randn(67, 32, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(67, 8, 128, device="npu", dtype=torch.bfloat16)

    def run():
        cos, sin = op(x, cu_q_lens=cu, total_seq_lens=total)
        return rope_infer(q, k, cos, sin, head_first=False, implementation="batch_invariant")

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for lengths, totals in [([0, 40, 67], [61, 79]), ([0, 19, 67], [128, 97])]:
        cu.copy_(torch.tensor(lengths, device="npu", dtype=torch.int32))
        total.copy_(torch.tensor(totals, device="npu", dtype=torch.int32))
        q.add_(0.125)
        expected = run()
        graph.replay()
        torch.npu.synchronize()
        for got, want in zip(captured, expected):
            assert torch.equal(got, want)


@pytest.mark.parametrize("cpu_metadata", ["cu_q_lens", "total_seq_lens"])
def test_packed_rope_requires_metadata_on_the_input_device(cpu_metadata):
    op = RoPECosSin(1000000.0, 128, implementation="batch_invariant", device="npu")
    x = torch.zeros(67, 128, device="npu", dtype=torch.bfloat16)
    metadata = {
        "cu_q_lens": torch.tensor([0, 40, 67], device="npu", dtype=torch.int32),
        "total_seq_lens": torch.tensor([61, 79], device="npu", dtype=torch.int32),
    }
    metadata[cpu_metadata] = metadata[cpu_metadata].cpu()
    with pytest.raises(ValueError, match="sequence metadata on x.device"):
        op(x, **metadata)
