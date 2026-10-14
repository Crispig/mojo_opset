import torch

from .reduction import amax
from .reduction import reduce_sum


def logprobs_from_logits(logits, labels, inplace_backward=False, *, implementation=None):
    """Selected-token FP32 logp; keep the same reduction formula on both engines.

    The compatibility flag never enables mutation of logits during backward.
    """
    del inplace_backward
    if logits.ndim < 2:
        raise ValueError(f"logits must have shape [..., vocab], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            "labels must match the leading logits dimensions: "
            f"labels={tuple(labels.shape)}, logits={tuple(logits.shape)}"
        )
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must use int32 or int64, got {labels.dtype}")
    logits_float = logits.float()
    shifted = logits_float - amax(logits_float, dim=-1, keepdim=True, implementation=implementation)
    log_probs = shifted - reduce_sum(shifted.exp(), dim=-1, keepdim=True, implementation=implementation).log()
    return log_probs.gather(dim=-1, index=labels.long().unsqueeze(-1)).squeeze(-1)
