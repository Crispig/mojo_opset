import os

from functools import lru_cache

import torch
import triton

try:
    from triton.runtime.libentry import libentry
except ImportError:

    def libentry():
        def _decorator(fn):
            return fn

        return _decorator


@lru_cache(maxsize=1)
def get_npu_properties() -> dict:
    override = os.environ.get("MOJO_ASCEND_ARCH")
    target = triton.runtime.driver.active.get_current_target()
    arch = str(target.arch)
    if override not in (None, "a2"):
        raise RuntimeError(f"batch_invariant is validated only for MOJO_ASCEND_ARCH=a2; got {override!r}.")
    if "Ascend910" not in arch:
        raise RuntimeError(f"batch_invariant is validated only for Ascend 910 devices; Triton reported {arch!r}.")
    return triton.runtime.driver.active.utils.get_device_properties("npu")


def get_num_cores(core_type: str = "vector") -> int:
    if core_type == "vector":
        return get_npu_properties()["num_vectorcore"]
    if core_type == "cube":
        return get_npu_properties()["num_aicore"]
    raise ValueError(f"Unsupported NPU core type: {core_type!r}")


def require_deterministic_mode() -> None:
    deterministic = os.environ.get("MOJO_DETERMINISTIC", "0").strip().lower()
    if deterministic not in {"1", "true", "yes", "on"}:
        raise RuntimeError("batch_invariant kernels require MOJO_DETERMINISTIC=1 before importing mojo_opset.")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms must be enabled.")
