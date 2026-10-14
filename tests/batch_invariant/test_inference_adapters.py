from functools import partial

import pytest
import torch

from mojo_opset.functions.kv_cache import store_paged_kv_cache
from mojo_opset.functions.rms_norm import rms_norm
from mojo_opset.kernels.npu_a2_batch_invariant.attention import attention_packed
from mojo_opset.modules import GroupRMSNormInfer
from mojo_opset.modules import PagedDecodeGQAInfer
from mojo_opset.modules import PagedPrefillGQAInfer
from mojo_opset.modules import RoPECosSin
from mojo_opset.modules import SwiGLU

batch_invariant_rms_norm = partial(rms_norm, implementation="batch_invariant")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def test_group_norm_matches_training_and_has_exact_backend():
    norm = GroupRMSNormInfer(2, 128, 1e-6, implementation="batch_invariant", device="npu", dtype=torch.bfloat16)
    assert norm.implementation == "batch_invariant"
    with torch.no_grad():
        norm.weight.fill_(1)
    inputs = [torch.randn(67, heads, 128, device="npu", dtype=torch.bfloat16) for heads in (4, 2)]
    with torch.no_grad():
        actual = norm(inputs)
    for index, tensor in enumerate(inputs):
        assert torch.equal(actual[index], batch_invariant_rms_norm(tensor, norm.weight[index], 1e-6))


def test_swiglu_preserves_intermediate_rounding():
    op = SwiGLU(implementation="batch_invariant")
    assert op.implementation == "batch_invariant"
    gate = torch.randn(67, 1024, device="npu", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    assert torch.equal(op(gate, up), torch.nn.functional.silu(gate) * up)
    assert torch.equal(op(gate, up)[-1], op(gate[-1:], up[-1:])[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_scales", [False, True])
def test_swiglu_backward_matches_unfused_training(dtype, with_scales):
    op = SwiGLU(implementation="batch_invariant")
    inputs = [torch.randn(7, 129, device="npu", dtype=dtype).requires_grad_(True) for _ in range(2)]
    if with_scales:
        inputs.append(torch.randn(7, device="npu", dtype=dtype).requires_grad_(True))
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
    output = op(*inputs)
    expected = torch.nn.functional.silu(reference_inputs[0]) * reference_inputs[1]
    if with_scales:
        expected = expected * reference_inputs[2][:, None]
    grad = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, inputs, grad)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad)
    assert torch.equal(output, expected)
    for actual, reference in zip(actual_grads, expected_grads):
        assert torch.equal(actual, reference)


def test_rotary_matches_hf_device_math():
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    cfg = Qwen3Config(head_dim=128, rope_theta=1000000.0)
    reference = Qwen3RotaryEmbedding(cfg).to("npu")
    op = RoPECosSin(1000000.0, 128, init_max_length=512, implementation="batch_invariant", device="cpu").to("npu")
    assert op.implementation == "batch_invariant"
    hidden = torch.zeros(4, 128, device="npu", dtype=torch.bfloat16)
    positions = torch.tensor([0, 63, 256, 511], device="npu", dtype=torch.int32)
    actual = op(hidden, position_ids=positions)
    expected = reference(hidden, positions[None])
    for got, want in zip(actual, expected):
        assert got.dtype == hidden.dtype
        assert torch.equal(got, want[0])


def test_paged_adapters_share_packed_attention_arithmetic():
    prefill = PagedPrefillGQAInfer(implementation="batch_invariant")
    decode = PagedDecodeGQAInfer(implementation="batch_invariant")
    assert prefill.implementation == decode.implementation == "batch_invariant"
    q = torch.randn(67, 4, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(67, 2, 128, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kc = torch.zeros(1, 2, 256, 128, device="npu", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    cu = torch.tensor([0, 67], dtype=torch.int32, device="npu")
    blocks = torch.tensor([[0]], dtype=torch.int32, device="npu")
    store_paged_kv_cache(
        k,
        v,
        kc,
        vc,
        blocks,
        cu,
        torch.zeros(1, dtype=torch.int32, device="npu"),
        implementation="batch_invariant",
    )
    with torch.no_grad():
        expected = attention_packed(q, k, v, cu, cu)
        assert torch.equal(prefill(q, kc, vc, cu, blocks), expected)
        assert torch.equal(decode(q[-1:], kc, vc, cu[1:], blocks), expected[-1:])
        with pytest.raises(NotImplementedError, match="causal AABB"):
            prefill(q, kc, vc, cu, blocks, mask=torch.zeros(1, device="npu"))
