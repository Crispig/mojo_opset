import ast

from importlib import import_module
from pathlib import Path

import pytest
import torch

from mojo_opset import functions
from mojo_opset.functions._dispatch import load_impl
from mojo_opset.kernels.npu_a2_batch_invariant import OPS
from mojo_opset.kernels.npu_a2_batch_invariant.rmsnorm import rmsnorm_fwd
from mojo_opset.modules.gemm import BatchInvariantGemm


@pytest.mark.parametrize("op_id, module_name", OPS.items())
def test_provider_binds_each_operator_in_its_own_file(op_id, module_name):
    forward, _ = load_impl(op_id, "batch_invariant", target="npu.a2")
    module = import_module(f"mojo_opset.kernels.npu_a2_batch_invariant.{module_name}")
    assert forward is getattr(module, f"{op_id}_fwd")


def test_refactored_dispatch_resolves_exact_provider():
    forward, backward = load_impl("rms_norm", "batch_invariant", target="npu.a2", require_backward=True)
    assert forward is rmsnorm_fwd
    assert backward.__module__.startswith("mojo_opset.kernels.npu_a2_batch_invariant.")


def test_unused_moe_bmm_and_seeded_sampling_are_not_advertised():
    for op_id in ("bmm", "group_gemm", "group_gemm_same_nk", "group_gemm_same_mn", "moe_scatter", "gumbel_sample"):
        with pytest.raises(LookupError, match="no exact binding"):
            load_impl(op_id, "batch_invariant", target="npu.a2")


def test_public_name_has_no_hardware_specific_alias():
    with pytest.raises(LookupError, match="No provider package"):
        load_impl("rms_norm", "npu_batch_invariant", target="npu.a2")


@pytest.mark.parametrize("target", ["cpu.generic", "npu.a5", "ilu.generic", "mlu.generic"])
def test_unvalidated_targets_fail_without_fallback(target):
    with pytest.raises(LookupError, match="No provider package"):
        load_impl("rms_norm", "batch_invariant", target=target)


@pytest.mark.parametrize("target", ["cpu.generic", "npu.a5", "ilu.generic", "mlu.generic"])
def test_per_operator_function_cannot_bypass_target_dispatch(monkeypatch, target):
    module = import_module("mojo_opset.functions.softmax")

    def select_target(op_id, implementation, **kwargs):
        return load_impl(op_id, implementation, target=target, **kwargs)

    monkeypatch.setattr(module, "load_impl", select_target)
    with pytest.raises(LookupError, match="No provider package"):
        module.softmax(torch.zeros(1), implementation="batch_invariant")


def test_batch_invariant_kernels_do_not_own_autograd():
    from mojo_opset.kernels import npu_a2_batch_invariant as provider

    for path in Path(provider.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert all(ast.unparse(base) != "torch.autograd.Function" for base in node.bases), path
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("save_for_backward", "saved_tensors"), path
                assert ast.unparse(node) != "torch.autograd.grad", path
                assert node.attr not in ("enable_grad", "requires_grad_"), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mojo_opset.functions"), path


@pytest.mark.parametrize(
    "op_id",
    [
        "batch_invariant_matmul",
        "batch_invariant_linear",
        "reduce_sum",
        "mean",
        "amax",
        "softmax",
        "log_softmax",
        "rms_norm",
        "rope",
        "swiglu",
    ],
)
def test_training_provider_exports_plain_forward_and_backward(op_id):
    forward, backward = load_impl(op_id, "batch_invariant", target="npu.a2", require_backward=True)
    module = import_module(f"mojo_opset.kernels.npu_a2_batch_invariant.{OPS[op_id]}")
    assert forward is getattr(module, f"{op_id}_fwd")
    assert backward is getattr(module, f"{op_id}_bwd")


@pytest.mark.parametrize(
    "op_id, module_name",
    [
        ("batch_invariant_matmul", "gemm"),
        ("reduce_sum", "reduction"),
        ("softmax", "softmax"),
    ],
)
def test_backward_keeps_the_provider_selected_by_forward(monkeypatch, op_id, module_name):
    module = import_module(f"mojo_opset.functions.{module_name}")
    calls = []

    def forward(input, second, bias=None):
        calls.append("forward")
        if module_name == "gemm":
            return input @ second
        if module_name == "reduction":
            return input.sum(dim=-1)
        return torch.softmax(input, dim=-1)

    def backward(grad_output, *saved):
        calls.append("backward")
        if module_name == "gemm":
            input, weight, has_bias, needs = saved
            assert not has_bias and needs == (True, True, False)
            return grad_output @ weight.T, input.T @ grad_output, None
        if module_name == "reduction":
            shape, dtype = saved
            return grad_output[:, None].expand(shape).to(dtype)
        output, dtype = saved
        return (output * (grad_output - (output * grad_output).sum(-1, keepdim=True))).to(dtype)

    def select(requested_op, implementation, *, require_backward=False):
        assert requested_op == op_id and implementation == "batch_invariant" and require_backward
        calls.append("select")
        return forward, backward

    monkeypatch.setattr(module, "load_impl", select)
    input = torch.randn(3, 7, requires_grad=True)
    if module_name == "gemm":
        weight = torch.randn(7, 5, requires_grad=True)
        output = functions.batch_invariant_matmul(input, weight, implementation="batch_invariant")
    else:
        output = getattr(functions, op_id)(input, dim=-1, implementation="batch_invariant")
    monkeypatch.setattr(module, "load_impl", lambda *args, **kwargs: pytest.fail("backward must not redispatch"))
    output.sum().backward()
    assert calls == ["select", "forward", "backward"]
    assert input.grad is not None


def test_batch_invariant_gemm_keeps_parameters_state_dict_and_gradients():
    exact = BatchInvariantGemm(7, 5, bias=True, device="npu", dtype=torch.bfloat16, implementation="batch_invariant")
    reference = torch.nn.Linear(7, 5, bias=True, device="npu", dtype=torch.bfloat16)
    reference.load_state_dict(exact.state_dict(), strict=True)
    source = torch.randn(3, 2, 7, device="npu", dtype=torch.bfloat16)
    actual_input = source.clone().requires_grad_(True)
    reference_input = source.clone().requires_grad_(True)
    actual = exact(actual_input)
    expected = reference(reference_input)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=2e-2, rtol=2e-2)
    for actual_param, expected_param in zip(exact.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "name",
    [
        "batch_invariant_linear",
        "batch_invariant_matmul",
        "reduce_sum",
        "mean",
        "amax",
        "softmax",
        "log_softmax",
    ],
)
def test_new_operators_have_no_reference_fallback(name):
    with pytest.raises(LookupError, match="no exact binding"):
        load_impl(name, "torch_reference")


def test_logprobs_has_no_reference_fallback():
    logits = torch.randn(4, 19)
    labels = torch.tensor([0, 7, 18, 3])
    with pytest.raises(LookupError, match="no exact binding"):
        functions.logprobs_from_logits(logits, labels, implementation="torch_reference")


def test_gemm_names_are_explicit_without_generic_aliases():
    from mojo_opset import modules

    assert modules.BatchInvariantGemm is BatchInvariantGemm
    assert not hasattr(modules, "Linear")
    assert not hasattr(functions, "linear")
    assert not hasattr(functions, "matmul")
    assert OPS["batch_invariant_linear"] == OPS["batch_invariant_matmul"] == "matmul"


def test_public_linear_and_matmul_share_bitwise_gemm_result():
    input = torch.randn(3, 2, 19, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(13, 19, device="npu", dtype=torch.bfloat16)
    bias = torch.randn(13, device="npu", dtype=torch.bfloat16)
    linear = functions.batch_invariant_linear(input, weight, bias, implementation="batch_invariant")
    matmul = functions.batch_invariant_matmul(
        input.reshape(-1, 19),
        weight.T,
        bias,
        implementation="batch_invariant",
    )
    assert torch.equal(linear.reshape(-1, 13), matmul)
    single = functions.batch_invariant_linear(input[1:2], weight, bias, implementation="batch_invariant")
    assert torch.equal(linear[1:2], single)


def test_original_reference_paged_store_is_unchanged():
    k = torch.randn(3, 2, 4)
    v = torch.randn_like(k)
    kc = torch.zeros(1, 2, 8, 4)
    vc = torch.zeros_like(kc)
    chunks = torch.tensor([[0, 0, 3, 1], [2, 0, 0, 1]], dtype=torch.int32)
    functions.store_paged_kv_cache(
        k,
        v,
        kc,
        vc,
        chunk_metadata=chunks,
        implementation="torch_reference",
    )
    assert torch.equal(kc[0, :, 3], k[0])
    assert torch.equal(vc[0, :, 0], v[2])


def test_slot_store_does_not_add_a_second_public_operator():
    from mojo_opset.kernels.npu_a2_batch_invariant import OPS
    from mojo_opset.kernels.torch_reference import OPS as reference_ops

    assert not hasattr(functions, "store_kvcache")
    assert "store_kvcache" not in OPS
    assert "store_kvcache" not in reference_ops
    assert OPS["store_paged_kv_cache"] == "kv_cache"


def test_paged_store_keeps_legacy_provider_head_dim_guard(monkeypatch):
    from mojo_opset.functions import kv_cache

    def unsupported_forward(*args):
        pytest.fail("a provider without separate head dimensions must not execute")

    monkeypatch.setattr(kv_cache, "load_impl", lambda *args: (unsupported_forward, None))
    key, value = torch.randn(2, 1, 5), torch.randn(2, 1, 7)
    kc, vc = torch.zeros(1, 1, 4, 5), torch.zeros(1, 1, 4, 7)
    with pytest.raises(ValueError, match="matching K/V head dimensions"):
        kv_cache.store_paged_kv_cache(key, value, kc, vc)
