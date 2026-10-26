from __future__ import annotations

import math
import os
import threading
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

from .mps_runtime import inference_guard


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        if batch_size <= 0:
            raise ValueError("reranker batch_size must be positive")
        if max_length <= 0:
            raise ValueError("reranker max_length must be positive")
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        visible_devices = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        use_multi_gpu = resolved_device.startswith("cuda") and len(visible_devices) > 1
        model_kwargs = {"torch_dtype": torch.float16} if resolved_device.startswith("cuda") else {}
        encoder_class = CrossEncoder
        if use_multi_gpu:
            if torch.cuda.device_count() < 2:
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES requests multiple GPUs, but fewer than two CUDA devices are available"
                )
            model_kwargs["device_map"] = "balanced"

            class ShardedCrossEncoder(CrossEncoder):
                def to(self, *args, **kwargs):
                    # predict() calls to(self.device); preserve Accelerate's layer placement.
                    if getattr(self, "_preserve_device_map", False):
                        target = args[0] if len(args) == 1 and not kwargs else kwargs.get("device")
                        if (
                            (len(args) == 1 and not kwargs or not args and set(kwargs) == {"device"})
                            and torch.device(target) == torch.device(self.device)
                        ):
                            return self
                        raise ValueError("Cannot move or cast a reranker managed by a multi-GPU device map")
                    return super().to(*args, **kwargs)

            encoder_class = ShardedCrossEncoder

        self.model = encoder_class(
            model_name,
            device=None if use_multi_gpu else resolved_device,
            max_length=max_length,
            model_kwargs=model_kwargs,
            trust_remote_code=False,
        )
        if use_multi_gpu:
            self.model._preserve_device_map = True
        self.batch_size = batch_size
        self._lock = threading.RLock()

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, document) for document in documents]
        with self._lock, inference_guard(self.model.device):
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(scores, dtype=np.float32).reshape(-1)


class VllmReranker:
    """Score through a vLLM pooling server, preserving its score scale and order."""

    def __init__(
        self, model_name: str, base_url: str, batch_size: int = 32,
        timeout: float = 120, api_key: str = "", max_length: int = 512,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("MEMORY_RERANKER_MODEL must be a nonempty vLLM served model name")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("reranker batch_size must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("MEMORY_RERANKER_MAX_LENGTH must be a positive integer")
        try:
            parsed_timeout = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("MEMORY_RERANKER_TIMEOUT must be positive and finite") from None
        if isinstance(timeout, bool) or not math.isfinite(parsed_timeout) or parsed_timeout <= 0:
            raise ValueError("MEMORY_RERANKER_TIMEOUT must be positive and finite")
        if not isinstance(base_url, str):
            raise ValueError("MEMORY_RERANKER_BASE_URL must be an HTTP(S) server root URL")
        base_url = base_url.strip()
        try:
            url = urlsplit(base_url)
            url.port  # Validate malformed ports before the first search.
            valid = (url.scheme in {"http", "https"} and url.hostname
                     and not url.username and not url.password and not url.query
                     and not url.fragment and url.path in {"", "/"})
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                "MEMORY_RERANKER_BASE_URL must be an HTTP(S) server root URL "
                "without credentials, query or path (e.g. http://127.0.0.1:8089)"
            )
        if not isinstance(api_key, str):
            raise ValueError("MEMORY_RERANKER_API_KEY must be a string")
        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_length = max_length
        self.timeout = parsed_timeout
        self._api_key = api_key.strip()

    @staticmethod
    def _scores(payload, count: int) -> np.ndarray:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != count:
            raise ValueError(f"vLLM /score must return exactly {count} scores")
        values = np.empty(count, dtype=np.float32)
        seen = set()
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("vLLM /score returned an invalid score item")
            index, score = item.get("index"), item.get("score")
            if type(index) is not int or not 0 <= index < count or index in seen:
                raise ValueError("vLLM /score returned invalid or duplicate score indices")
            if type(score) not in (int, float):
                raise ValueError("vLLM /score returned a non-numeric score")
            try:
                number = float(score)
            except (ValueError, OverflowError):
                raise ValueError("vLLM /score returned a non-finite score") from None
            if not math.isfinite(number) or abs(number) > float(np.finfo(np.float32).max):
                raise ValueError("vLLM /score returned a non-finite or out-of-range score")
            seen.add(index)
            values[index] = number
        return values

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)
        import httpx

        scores = []
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        # Reuse connections across batches; close them even when a batch fails.
        # Each call owns its client, so concurrent searches need no global lock.
        with httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=False) as client:
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start:start + self.batch_size]
                batch_number = start // self.batch_size + 1
                try:
                    response = client.post(self.base_url + "/score", json={
                        "model": self.model_name,
                        "queries": [query] * len(batch),
                        "documents": batch,
                        "truncate_prompt_tokens": self.max_length,
                    })
                except httpx.RequestError as exc:
                    raise RuntimeError(
                        f"vLLM reranker request failed ({type(exc).__name__}; batch={batch_number})"
                    ) from None
                if response.status_code != 200:
                    # Do not expose server response bodies, credentials or memory text.
                    raise RuntimeError(
                        f"vLLM reranker request failed (HTTP {response.status_code}; batch={batch_number})"
                    )
                try:
                    payload = response.json()
                except ValueError:
                    raise ValueError(f"vLLM /score returned invalid JSON (batch={batch_number})") from None
                scores.append(self._scores(payload, len(batch)))
        return np.concatenate(scores)


def reranker_options(overrides: dict[str, object] | None = None) -> dict:
    """Resolve startup settings before env defaults, without parsing unused options."""
    overrides = overrides or {}

    def setting(name: str, default):
        key = f"reranker_{name}"
        return overrides[key] if key in overrides else os.environ.get(f"MEMORY_RERANKER_{name.upper()}", default)

    backend = setting("backend", "cross-encoder")
    if not isinstance(backend, str) or backend.strip().lower() not in {"cross-encoder", "vllm"}:
        raise ValueError("MEMORY_RERANKER_BACKEND must be cross-encoder or vllm")
    backend = backend.strip().lower()
    if backend == "cross-encoder":
        return {"backend": backend}
    return dict(backend=backend, base_url=setting("base_url", ""),
                api_key=setting("api_key", ""), timeout=setting("timeout", 120))


def build_reranker(
    model_name: str,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 512,
    *,
    backend: str = "cross-encoder",
    base_url: str = "",
    api_key: str = "",
    timeout: float = 120,
) -> Reranker:
    if backend == "vllm":
        return VllmReranker(model_name, base_url, batch_size, timeout, api_key, max_length=max_length)
    if backend != "cross-encoder":
        raise ValueError("reranker backend must be cross-encoder or vllm")
    return CrossEncoderReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
