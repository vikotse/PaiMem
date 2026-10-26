"""Serialize local MPS inference across embedding and reranking models."""
from __future__ import annotations

from contextlib import contextmanager
import threading


_inference_lock = threading.RLock()


@contextmanager
def inference_guard(device):
    # Inspect the model's actual device, including automatically selected MPS.
    # CPU/CUDA callers neither import torch here nor touch the MPS runtime.
    if str(device).split(":", 1)[0] != "mps":
        yield
        return

    import torch

    with _inference_lock:
        try:
            yield
        finally:
            # Metal work is asynchronous; keep ownership until it completes,
            # including when inference raises after submitting GPU work.
            torch.mps.synchronize()
