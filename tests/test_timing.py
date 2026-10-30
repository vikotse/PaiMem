"""Timing diagnostics preserve results, lock ownership and private input data."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from paimem import embeddings as module
from paimem.memory_store import MemoryMessage, SQLiteMemoryStore
from paimem.mempursuit import MemPursuitConfig


def events(caplog):
    return [json.loads(r.getMessage().split('paimem_timing ', 1)[1])
            for r in caplog.records if r.name == 'paimem.timing']


def test_parallel_callers_share_worker_limit_and_preserve_traces(monkeypatch, caplog):
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '1')
    monkeypatch.setenv('EMBEDDING_MAX_CONCURRENCY', '2')
    monkeypatch.setenv('EMBEDDING_BATCH_SIZE', '1')
    monkeypatch.setenv('EMBEDDING_RPM', '100')
    monkeypatch.setenv('EMBEDDING_TPM', '1000')
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active, maximum = 0, 0
    def handler(request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait(timeout=5)
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [3, 4]}],
                                            'usage': {'total_tokens': 1}})
        finally:
            with lock:
                active -= 1
    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    model = module.RemoteEmbedder('m', 'https://example.com/v1', 's')
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(model.encode_queries, ['q'] * 4),
                       pool.submit(model.encode_documents, ['d'] * 4)]
            for future in futures:
                np.testing.assert_allclose(future.result(timeout=10), [[.6, .8]] * 4)
    finally:
        model.close()
    assert maximum == 2
    records = events(caplog)
    ids = {r['trace_id'] for r in records}
    assert len(ids) == 2
    for trace in ids:
        trace_records = [r for r in records if r['trace_id'] == trace]
        for stage in ('embedding.queue_wait', 'embedding.http', 'embedding.usage', 'embedding.rate_limit_wait'):
            assert sum(r['stage'] == stage for r in trace_records) == 4
    assert model._limiter._tokens == 8
    assert len(model._limiter._events) == 8


def test_lock_wait_encode_and_failure_cleanup(monkeypatch, caplog):
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '1')
    clock = [0.0]
    class Lock:
        held = False
        def acquire(self):
            clock[0] += 2
            self.held = True
        def release(self):
            self.held = False
    lock = Lock()
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    with pytest.raises(ValueError, match='private-exception'):
        with module.timing_stage('search.total'):
            with module.timed_embedding_lock(lock, 'query_embedding', 1):
                assert lock.held
                clock[0] += 3
                raise ValueError('private-exception')
    records = events(caplog)
    assert [(r['stage'], r['elapsed_ms'], r['status']) for r in records] == [
        ('query_embedding.lock_wait', 2000, 'ok'),
        ('query_embedding.encode', 3000, 'error'),
        ('search.total', 5000, 'error')]
    assert len({r['trace_id'] for r in records}) == 1
    assert not lock.held and module._timing_context.get() is None
    assert 'private-exception' not in caplog.text


def test_disabled_is_silent_and_does_not_generate_ids(monkeypatch, caplog):
    monkeypatch.delenv('MEMORY_TIMING_ENABLED', raising=False)
    with patch.object(module, 'uuid4', side_effect=AssertionError('disabled')):
        with module.timing_stage('search.total'):
            with module.timed_embedding_lock(threading.RLock(), 'query_embedding', 1):
                pass
    assert events(caplog) == []


def test_concurrent_operations_have_distinct_ids_and_reset(monkeypatch, caplog):
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', 'true')
    barrier = threading.Barrier(2)
    @module.timed_operation('search.total')
    def operation():
        barrier.wait(timeout=5)
        with module.timing_stage('query_embedding.encode'):
            pass
    def worker():
        operation()
        assert module._timing_context.get() is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: worker(), range(2)))
    records = events(caplog)
    ids = {r['trace_id'] for r in records}
    assert len(ids) == 2
    assert all(sum(r['trace_id'] == trace for r in records) == 2 for trace in ids)


def test_remote_retry_and_quota_wait_are_separate(monkeypatch, caplog):
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '1')
    monkeypatch.setenv('EMBEDDING_RPM', '1')
    monkeypatch.setenv('EMBEDDING_TPM', '0')
    clock = [0.0]
    calls = []
    def advance(seconds):
        clock[0] += seconds
    def handler(request):
        calls.append(request)
        advance(.25)
        if len(calls) == 1:
            return httpx.Response(429, text='PRIVATE_RESPONSE')
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [3, 4]}]})
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', advance)
    monkeypatch.setattr(module._RateLimiter, '_wait', lambda self, delay: advance(delay))
    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    embedder = module.RemoteEmbedder('PRIVATE_MODEL', 'https://private-host/v1', 'PRIVATE_KEY')
    np.testing.assert_allclose(embedder.encode_queries(['PRIVATE_QUERY']), [[.6, .8]])
    records = events(caplog)
    assert len({r['trace_id'] for r in records}) == 1
    http = [r for r in records if r['stage'] == 'embedding.http']
    assert [(r['http_status'], r['status'], r['elapsed_ms']) for r in http] == [(429, 'error', 250), (200, 'ok', 250)]
    waits = [r['elapsed_ms'] for r in records if r['stage'] == 'embedding.rate_limit_wait']
    assert waits == [0, 58750]
    assert next(r['elapsed_ms'] for r in records if r['stage'] == 'embedding.retry_wait') == 1000
    assert records[-1]['elapsed_ms'] == 60250
    assert not any(secret in '\n'.join(r.getMessage() for r in caplog.records if r.name == 'paimem.timing')
                   for secret in ['PRIVATE_QUERY', 'PRIVATE_RESPONSE', 'PRIVATE_MODEL', 'PRIVATE_KEY', 'private-host'])


def test_remote_validation_failure_has_total_error(monkeypatch, caplog):
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '1')
    monkeypatch.setenv('EMBEDDING_RPM', '0')
    monkeypatch.setenv('EMBEDDING_TPM', '0')
    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'data': []})), **kw))
    with pytest.raises(ValueError):
        module.RemoteEmbedder('m', 'https://example.com/v1', 'secret').encode_queries(['q'])
    assert events(caplog)[-1]['status'] == 'error'
    assert module._timing_context.get() is None


def test_store_stage_counts_and_results_are_preserved(tmp_path, monkeypatch, caplog):
    class Embedder:
        def encode_documents(self, texts):
            return np.tile([1., 0.], (len(texts), 1)).astype(np.float32)
        encode_queries = encode_documents
    class Reranker:
        def score(self, query, documents):
            return np.arange(len(documents), 0, -1, dtype=np.float32)
    store = SQLiteMemoryStore(tmp_path / 'db.sqlite', Embedder(), reranker=Reranker(),
                              mempursuit_config=MemPursuitConfig(enabled=True, candidates=4),
                              mempursuit_small2big_enabled=True)
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '1')
    store.add('PRIVATE_REQUEST', 'PRIVATE_USER', 'PRIVATE_SESSION', [MemoryMessage('user', 'PRIVATE_DOCUMENT')])
    assert any(r['stage'] == 'document_embedding.encode' and r['inputs'] == 1 for r in events(caplog))
    caplog.clear()
    enabled = store.search('PRIVATE_USER', 'PRIVATE_QUERY', 1)
    records = events(caplog)
    assert len({r['trace_id'] for r in records}) == 1
    stages = {r['stage']: r for r in records}
    for stage in ('query_embedding.encode', 'mempursuit_embedding.encode', 'reranker.score'):
        assert stages[stage]['inputs'] == 1
    assert 'search.total' in stages and 'retrieval.total' in stages
    assert 'PRIVATE_' not in caplog.text
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '0')
    caplog.clear()
    disabled = store.search('PRIVATE_USER', 'PRIVATE_QUERY', 1)
    assert enabled == disabled and not events(caplog)


def test_remote_store_skips_outer_lock_for_all_embedding_stages(tmp_path, monkeypatch, caplog):
    class ForbiddenLock:
        def acquire(self):
            raise AssertionError('remote embedding must not acquire the store lock')
        def release(self):
            raise AssertionError('remote embedding must not release the store lock')

    def handler(request):
        texts = json.loads(request.content)['input']
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [3, 4]} for i in range(len(texts))]})

    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '1')
    monkeypatch.setenv('EMBEDDING_RPM', '0')
    monkeypatch.setenv('EMBEDDING_TPM', '0')
    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    embedder = module.RemoteEmbedder('m', 'https://example.com/v1', 'secret')
    store = SQLiteMemoryStore(tmp_path / 'db.sqlite', embedder,
                              mempursuit_config=MemPursuitConfig(enabled=True, candidates=4),
                              mempursuit_small2big_enabled=True)
    store._embedding_lock = ForbiddenLock()
    store.add('request', 'user', 'session', [MemoryMessage('user', 'test document')])
    assert store.search('user', 'query', 1)
    stages = {r['stage'] for r in events(caplog)}
    for stage in ('document_embedding', 'query_embedding', 'mempursuit_embedding'):
        assert stage + '.encode' in stages
        assert stage + '.lock_wait' not in stages
