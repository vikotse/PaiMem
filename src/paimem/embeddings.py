from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from email.utils import parsedate_to_datetime
import os
import time
from typing import Protocol

import numpy as np

from .mps_runtime import inference_guard


def _integer_setting(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class _RateLimiter:
    """Reserve each attempt in a shared, per-instance rolling minute."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._events: deque[tuple[float, int]] = deque()
        self._tokens = 0
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        if self.tpm and tokens > self.tpm:
            raise ValueError("A single embedding input exceeds EMBEDDING_TPM (UTF-8 byte estimate)")
        if not (self.rpm or self.tpm):
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= 60:
                    self._tokens -= self._events.popleft()[1]
                if ((not self.rpm or len(self._events) < self.rpm)
                        and (not self.tpm or self._tokens + tokens <= self.tpm)):
                    self._events.append((now, tokens))
                    self._tokens += tokens
                    return
                delay = max(0.001, self._events[0][0] + 60 - now)
            # Never hold the reservation lock while sleeping or making HTTP calls.
            time.sleep(delay)


def _retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local Hugging Face embedding backend for server-side benchmark runs."""

    def __init__(self, model_name: str, device: str | None = None, query_prefix: str | None = None) -> None:
        self.batch_size = _integer_setting("EMBEDDING_BATCH_SIZE", 64, 1)
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        if query_prefix is None and "bge" in model_name.lower() and "-en-" in model_name.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        self.query_prefix = query_prefix or ""
        self.identity = f"{model_name}|query_prefix={self.query_prefix}"
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        with inference_guard(self.model.device):
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 32,
            )
            return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + text for text in texts])


class RemoteEmbedder:
    """OpenAI-compatible embeddings, including Alibaba Cloud text-embedding-v4."""

    def __init__(self, model_name: str, base_url: str, api_key: str,
                 query_prefix: str | None = None) -> None:
        self.batch_size = _integer_setting("EMBEDDING_BATCH_SIZE", 10, 1)
        self._limiter = _RateLimiter(
            _integer_setting("EMBEDDING_RPM", 0, 0),
            _integer_setting("EMBEDDING_TPM", 0, 0),
        )
        from urllib.parse import urlsplit

        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("EMBEDDING_BASE_URL must be an HTTP(S) base URL without credentials or query parameters")
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/embeddings"):
            raise ValueError("EMBEDDING_BASE_URL must end at the API base (e.g. /v1), not /embeddings")
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self._api_key = api_key
        self.dimension: int | None = None
        self.identity = "remote:" + json.dumps(
            [self.base_url, model_name, self.query_prefix], separators=(",", ":"))

    def _request_batch(self, client, batch: list[str], batch_number: int):
        import httpx

        logger = logging.getLogger(__name__)
        max_attempts = 3
        tokens = sum(len(text.encode("utf-8")) for text in batch)
        for attempt in range(1, max_attempts + 1):
            self._limiter.acquire(tokens)
            retry_after = 0.0
            try:
                response = client.post(
                    self.base_url + "/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model_name, "input": batch, "encoding_format": "float"},
                )
            except httpx.RequestError as error:
                # Log only the class: exception text can contain URLs or credentials.
                reason = type(error).__name__
                retryable = isinstance(error, (httpx.TimeoutException, httpx.NetworkError,
                                               httpx.RemoteProtocolError))
            else:
                if response.status_code == 200:
                    return response
                reason = f"HTTP {response.status_code}"
                retryable = response.status_code in {429, 502, 503, 504}
                retry_after = _retry_after(response.headers.get("Retry-After"))
                response.close()
            if not retryable or attempt == max_attempts:
                logger.error("Remote embedding failed batch=%d attempt=%d/%d reason=%s",
                             batch_number, attempt, max_attempts, reason)
                raise RuntimeError(
                    f"Remote embedding request failed ({reason}; batch={batch_number}; attempts={attempt})"
                ) from None
            delay = max(float(2 ** (attempt - 1)), retry_after)
            logger.warning("Remote embedding retry batch=%d attempt=%d/%d reason=%s delay_seconds=%.1f",
                           batch_number, attempt, max_attempts, reason, delay)
            time.sleep(delay)

    def _batches(self, texts: list[str]):
        # Validate all inputs first, avoiding partially submitted oversized calls.
        sizes = [len(text.encode("utf-8")) for text in texts]
        tpm = self._limiter.tpm
        if tpm and any(size > tpm for size in sizes):
            raise ValueError("A single embedding input exceeds EMBEDDING_TPM (UTF-8 byte estimate)")
        batch, tokens = [], 0
        for text, size in zip(texts, sizes):
            if batch and (len(batch) >= self.batch_size or (tpm and tokens + size > tpm)):
                yield batch
                batch, tokens = [], 0
            batch.append(text)
            tokens += size
        if batch:
            yield batch

    def _encode(self, texts: list[str]) -> np.ndarray:
        import httpx

        if not texts:
            return np.empty((0, self.dimension or 0), dtype=np.float32)
        vectors = []
        # Overrides must respect the provider's per-request input limit.
        with httpx.Client(timeout=60.0) as client:
            for number, batch in enumerate(self._batches(texts), 1):
                response = self._request_batch(client, batch, number)
                try:
                    rows = response.json()["data"]
                    if (len(rows) != len(batch)
                            or any(type(row["index"]) is not int for row in rows)
                            or sorted(row["index"] for row in rows) != list(range(len(batch)))):
                        raise ValueError()
                    matrix = np.asarray([row["embedding"] for row in sorted(rows, key=lambda row: row["index"])],
                                        dtype=np.float32)
                    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                        raise ValueError()
                    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
                    if np.any(norms == 0):
                        raise ValueError()
                    if self.dimension is not None and matrix.shape[1] != self.dimension:
                        raise ValueError()
                except (ValueError, TypeError, KeyError, OverflowError):
                    raise ValueError("Remote embedding returned invalid vectors or indices") from None
                self.dimension = matrix.shape[1]
                vectors.append((matrix / norms).astype(np.float32))
        return np.concatenate(vectors, axis=0)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + text for text in texts])


def embedding_options(overrides: dict[str, object] | None = None) -> dict[str, object]:
    """Startup-only provider options; explicit values take precedence over env."""
    overrides = overrides or {}
    return {key: overrides.get("embedding_" + key, os.environ.get("EMBEDDING_" + key.upper(), default))
            for key, default in (("backend", "local"), ("base_url", ""), ("api_key", ""))}


def build_embedder(model_name: str | None = None, device: str | None = None,
                   query_prefix: str | None = None, *, backend: str = "local",
                   base_url: str = "", api_key: str = "") -> Embedder:
    backend = backend.strip().lower()
    if backend == "local":
        return SentenceTransformerEmbedder(model_name=model_name or "BAAI/bge-small-en-v1.5",
                                          device=device, query_prefix=query_prefix)
    if backend != "remote":
        raise ValueError("EMBEDDING_BACKEND must be local or remote")
    for label, value in (("MODEL", model_name), ("BASE_URL", base_url), ("API_KEY", api_key)):
        if not value or not value.strip():
            raise ValueError(f"EMBEDDING_{label} is required for remote embeddings")
    return RemoteEmbedder(model_name.strip(), base_url.strip(), api_key.strip(), query_prefix)
