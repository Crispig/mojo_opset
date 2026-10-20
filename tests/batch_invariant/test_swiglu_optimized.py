import pytest
import torch
import torch.nn.functional as F

from mojo_opset.functions.swiglu import swiglu
from mojo_opset.kernels.npu_a2_batch_invariant.swiglu import swiglu_fwd

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _reference(first, second, scales=None):
    result = F.silu(first) * second
    if scales is not None:
        result = result * scales.reshape(*first.shape[:-1], 1)
    return result.contiguous()


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern_in_gate", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_fused_swiglu_preserves_all_16bit_patterns(dtype, pattern_in_gate, scaled):
    torch.manual_seed(90)
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).reshape(256, 256).to("npu")
    regular = torch.randn_like(patterns)
    first, second = (patterns, regular) if pattern_in_gate else (regular, patterns)
    scales = torch.randn(256, device="npu", dtype=torch.float32) if scaled else None
    _assert_bits(swiglu_fwd(first, second, scales), _reference(first, second, scales))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 17), (2, 0), (1, 128), (17, 513), (2, 3, 128)])
def test_fused_swiglu_preserves_native_forward_and_backward(dtype, scale_dtype, shape):
    torch.manual_seed(19)
    first = torch.randn(shape, device="npu", dtype=dtype, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    inputs = (first, second)
    if scale_dtype is not None:
        scales = torch.randn(shape[:-1], device="npu", dtype=scale_dtype, requires_grad=True)
        inputs += (scales,)
    expected = _reference(*inputs)
    actual = swiglu(*inputs, implementation="batch_invariant")
    _assert_bits(actual, expected)
    gradient = torch.randn_like(expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, inputs, gradient),
        torch.autograd.grad(expected, inputs, gradient),
    ):
        _assert_bits(actual_grad.contiguous(), expected_grad.contiguous())


@pytest.mark.parametrize("first_shape,second_shape", [((), ()), ((), (7, 17)), ((3, 1), (3, 17)), ((7, 17), (17,))])
@pytest.mark.parametrize(
    "first_dtype,second_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16),
    ],
)
def test_fused_swiglu_preserves_broadcast_and_promotion(first_shape, second_shape, first_dtype, second_dtype):
    first = torch.randn(first_shape, device="npu", dtype=first_dtype)
    second = torch.randn(second_shape, device="npu", dtype=second_dtype)
    _assert_bits(swiglu_fwd(first, second), _reference(first, second))


@pytest.mark.parametrize(
    "first_dtype,scale_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fused_swiglu_preserves_scalar_scale_promotion(first_dtype, scale_dtype):
    first = torch.randn((), device="npu", dtype=first_dtype)
    second = torch.randn_like(first)
    scales = torch.randn((), device="npu", dtype=scale_dtype)
    _assert_bits(swiglu_fwd(first, second, scales), _reference(first, second, scales))


def test_fused_swiglu_preserves_float32_exponents_and_batch_slices():
    torch.manual_seed(31)
    first = torch.randint(-(2**31), 2**31, (257, 513), dtype=torch.int32).view(torch.float32).to("npu")
    second = torch.randn_like(first)
    _assert_bits(swiglu_fwd(first, second), _reference(first, second))
    expected = swiglu_fwd(first, second)
    for start, end in ((0, 1), (17, 33), (33, 160), (160, 257)):
        _assert_bits(swiglu_fwd(first[start:end], second[start:end]), expected[start:end])


def test_fused_swiglu_preserves_strided_inputs_and_graph_replay():
    first = torch.randn(513, 17, device="npu", dtype=torch.bfloat16).T
    second = torch.randn_like(first)
    scales = torch.randn(34, device="npu", dtype=torch.float32)[::2]
    for _ in range(3):
        swiglu_fwd(first, second, scales)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = swiglu_fwd(first, second, scales)
    for _ in range(2):
        first.copy_(torch.randn_like(first))
        second.copy_(torch.randn_like(second))
        scales.copy_(torch.randn_like(scales))
        graph.replay()
        torch.npu.synchronize()
        _assert_bits(captured, _reference(first, second, scales))
