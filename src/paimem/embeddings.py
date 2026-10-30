from __future__ import annotations

import json
import logging
import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from functools import wraps
from uuid import uuid4
from collections import deque
from email.utils import parsedate_to_datetime
import os
import time
from typing import Protocol

import numpy as np

from .mps_runtime import inference_guard


_timing_context = ContextVar('paimem_timing_context', default=None)
_timing_logger = logging.getLogger('paimem.timing')


@contextmanager
def timing_stage(stage: str, *, started_at: float | None = None, **counts):
    """Opt-in wall time, including failures, with no input/exception text.

    Set MEMORY_TIMING_ENABLED=1 before starting the service. Diagnostic records
    use WARNING so they are visible under the default service logging setup.
    Nested stage durations overlap: do not add parent and child durations.
    trace_id identifies a store operation, not an incoming HTTP request ID.
    Only internal stage names and non-sensitive diagnostic counters may be passed.
    """
    if os.environ.get('MEMORY_TIMING_ENABLED', '').lower() not in {'1', 'true'}:
        yield {}
        return
    token = None
    if _timing_context.get() is None:
        token = _timing_context.set(uuid4().hex)
    record = dict(trace_id=_timing_context.get(), stage=stage, **counts)
    started = time.perf_counter() if started_at is None else started_at
    status = 'ok'
    error_type = None
    try:
        yield record
    except BaseException as exc:
        status, error_type = 'error', type(exc).__name__
        raise
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        if token is not None:
            _timing_context.reset(token)
        record.update(elapsed_ms=round(elapsed, 3),
                      status=record.get('status', 'ok') if status == 'ok' else status)
        if error_type is not None:
            record['error_type'] = error_type
        _timing_logger.warning('paimem_timing %s', json.dumps(record, separators=(',', ':')))


def timed_operation(stage):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with timing_stage(stage):
                return func(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def timed_embedding_lock(lock, stage: str, inputs: int, *, embedder: Embedder | None = None):
    """Serialize local/custom backends; remote calls protect their own state."""
    if isinstance(embedder, RemoteEmbedder):
        with timing_stage(stage + '.encode', inputs=inputs):
            yield
        return
    acquired = False
    try:
        with timing_stage(stage + '.lock_wait', inputs=inputs):
            lock.acquire()
            acquired = True
        with timing_stage(stage + '.encode', inputs=inputs):
            yield
    finally:
        if acquired:
            lock.release()


def _integer_setting(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(eq=False)
class _Reservation:
    timestamp: float
    tokens: int
    active: bool = True


class _RateLimiter:
    """Reserve each attempt in a shared, per-instance rolling minute."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._events: deque[_Reservation] = deque()
        self._tokens = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def _expire(self):
        now = time.monotonic()
        while self._events and now - self._events[0].timestamp >= 60:
            event = self._events.popleft()
            self._tokens -= event.tokens
            event.active = False

    def _wait(self, delay):
        self._condition.wait(delay)

    def acquire(self, tokens: int, timing=None) -> _Reservation | None:
        if self.tpm and tokens > self.tpm:
            raise ValueError("A single embedding input exceeds EMBEDDING_TPM (UTF-8 byte estimate)")
        if not (self.rpm or self.tpm):
            return
        while True:
            with self._condition:
                now = time.monotonic()
                self._expire()
                rpm_blocked = bool(self.rpm and len(self._events) >= self.rpm)
                tpm_blocked = bool(self.tpm and self._tokens + tokens > self.tpm)
                if timing is not None:
                    timing['rpm_blocked'] = timing.get('rpm_blocked', False) or rpm_blocked
                    timing['tpm_blocked'] = timing.get('tpm_blocked', False) or tpm_blocked
                if not (rpm_blocked or tpm_blocked):
                    reservation = _Reservation(now, tokens)
                    self._events.append(reservation)
                    self._tokens += tokens
                    return reservation
                delay = max(0.001, self._events[0].timestamp + 60 - now)
                # Condition.wait releases the lock and wakes on usage corrections.
                self._wait(delay)

    def reconcile(self, reservation: _Reservation | None, actual_tokens: int):
        if reservation is None:
            return
        with self._condition:
            self._expire()
            if reservation.active:
                self._tokens += actual_tokens - reservation.tokens
                reservation.tokens = actual_tokens
                self._condition.notify_all()


def _usage_tokens(body):
    usage = body.get('usage') if isinstance(body, dict) else None
    if isinstance(usage, dict):
        for key in ('total_tokens', 'prompt_tokens', 'input_tokens'):
            value = usage.get(key)
            if type(value) is int and value > 0:
                return value
    return None


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
        self.max_concurrency = _integer_setting("EMBEDDING_MAX_CONCURRENCY", 4, 1)
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrency,
                                            thread_name_prefix='embedding')
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
        self._dimension_lock = threading.Lock()
        self.identity = "remote:" + json.dumps(
            [self.base_url, model_name, self.query_prefix], separators=(",", ":"))

    def _request_batch(self, client, batch: list[str], batch_number: int):
        import httpx

        logger = logging.getLogger(__name__)
        max_attempts = 3
        tokens = sum(len(text.encode("utf-8")) for text in batch)
        for attempt in range(1, max_attempts + 1):
            with timing_stage('embedding.rate_limit_wait', batch=batch_number, attempt=attempt,
                              inputs=len(batch), estimated_bytes=tokens,
                              rpm_limit=self._limiter.rpm, tpm_limit=self._limiter.tpm,
                              rpm_blocked=False, tpm_blocked=False) as quota:
                reservation = self._limiter.acquire(tokens, quota)
            retry_after = 0.0
            try:
                with timing_stage('embedding.http', batch=batch_number, attempt=attempt,
                                  inputs=len(batch)) as timing:
                    response = client.post(
                        self.base_url + "/embeddings",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={"model": self.model_name, "input": batch, "encoding_format": "float"},
                    )
                    timing['http_status'] = response.status_code
                    if response.status_code != 200:
                        timing['status'] = 'error'
            except httpx.RequestError as error:
                # Log only the class: exception text can contain URLs or credentials.
                reason = type(error).__name__
                retryable = isinstance(error, (httpx.TimeoutException, httpx.NetworkError,
                                               httpx.RemoteProtocolError))
            else:
                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError:
                        raise ValueError("Remote embedding returned invalid vectors or indices") from None
                    finally:
                        response.close()
                    actual = _usage_tokens(body)
                    with timing_stage('embedding.usage', batch=batch_number, attempt=attempt,
                                      estimated_bytes=tokens, actual_tokens=actual,
                                      usage_reported=actual is not None):
                        if actual is not None:
                            self._limiter.reconcile(reservation, actual)
                    return body
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
            with timing_stage('embedding.retry_wait', batch=batch_number, attempt=attempt,
                              planned_ms=delay * 1000):
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

    def close(self):
        """Drain outstanding batches and release worker threads."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _encode_batch(self, client, batch, number, submitted):
        with timing_stage('embedding.queue_wait', batch=number, inputs=len(batch), started_at=submitted):
            pass
        body = self._request_batch(client, batch, number)
        try:
            rows = body["data"]
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
            with self._dimension_lock:
                if self.dimension is not None and matrix.shape[1] != self.dimension:
                    raise ValueError()
                self.dimension = matrix.shape[1]
            return (matrix / norms).astype(np.float32)
        except (ValueError, TypeError, KeyError, OverflowError):
            raise ValueError("Remote embedding returned invalid vectors or indices") from None

    @timed_operation('embedding.remote_encode_total')
    def _encode(self, texts: list[str]) -> np.ndarray:
        import httpx

        if not texts:
            with self._dimension_lock:
                return np.empty((0, self.dimension or 0), dtype=np.float32)
        # Validate every input before submitting any work; keep only a bounded
        # number of queued batches per caller. All callers share the same pool.
        batches = list(self._batches(texts))
        vectors = [None] * len(batches)
        pending = {}
        next_batch = 0
        # HTTPX sync clients support concurrent calls; retain this client until
        # every running batch finishes, including when another batch fails.
        with httpx.Client(timeout=60.0) as client:
            try:
                while pending or next_batch < len(batches):
                    while next_batch < len(batches) and len(pending) < self.max_concurrency:
                        i = next_batch
                        future = self._executor.submit(copy_context().run, self._encode_batch,
                                                       client, batches[i], i + 1, time.perf_counter())
                        pending[future] = i
                        next_batch += 1
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        vectors[pending.pop(future)] = future.result()
            finally:
                for future in pending:
                    future.cancel()
                wait(pending)
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
