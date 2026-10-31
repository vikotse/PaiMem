import importlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np


PACKAGES = ('paimem',)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.failure_path = os.path.join(directory.name, 'failures.jsonl')
        environment = patch.dict(os.environ, {'EMBEDDING_FAILURE_LOG_PATH': self.failure_path})
        environment.start()
        self.addCleanup(environment.stop)

    def test_failure_file_preserves_request_and_redacts_embedding_api_key(self):
        from paimem import embeddings as module
        with patch.dict(os.environ, {'EMBEDDING_MAX_INPUT_TOKENS': '32768'}), patch('httpx.Client') as client:
            post = client.return_value.__enter__.return_value.post
            post.return_value = httpx.Response(400, text='invalid input; key=secret-key')
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 'secret-key', 'query: ')
            self.addCleanup(model.close)
            texts = ['中文\nOptions:\n答案 secret-key', 'x' * 100000]
            with self.assertRaises(RuntimeError):
                model.encode_queries(texts)
            with open(self.failure_path) as stream:
                lines = stream.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            expected = json.loads(json.dumps(post.call_args.kwargs['json']).replace('secret-key', '[REDACTED]'))
            self.assertEqual(record['request'], expected)
            self.assertEqual(record['http_status'], 400)
            self.assertEqual(record['attempts'], 1)
            self.assertEqual(record['response_body'], 'invalid input; key=[REDACTED]')
            self.assertNotIn('secret-key', lines[0])
            self.assertEqual(os.stat(self.failure_path).st_mode & 0o777, 0o600)

    def test_failure_recording_is_disabled_without_nonempty_path(self):
        from paimem import embeddings as module
        for value in (None, '', '  '):
            env = {} if value is None else {'EMBEDDING_FAILURE_LOG_PATH': value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True), \
                    patch('httpx.Client') as client:
                client.return_value.__enter__.return_value.post.return_value = httpx.Response(400, text='bad')
                model = module.RemoteEmbedder('v4', 'https://example.com/v1', 'test-key')
                self.addCleanup(model.close)
                with patch.object(module.os, 'open') as open_file, \
                        patch.object(module.Path, 'mkdir') as mkdir, \
                        self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                    model.encode_queries(['synthetic input'])
                open_file.assert_not_called()
                mkdir.assert_not_called()
        self.assertFalse(os.path.exists(self.failure_path))

    def test_truncation_success_does_not_create_failure_file(self):
        from paimem import embeddings as module
        with patch.dict(os.environ, {'EMBEDDING_MAX_INPUT_TOKENS': '1',
                                    'EMBEDDING_TRUNCATION_STRATEGY': 'head-only'}), patch('httpx.Client') as client:
            post = client.return_value.__enter__.return_value.post
            post.return_value = httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 1]}]})
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 'test-key')
            self.addCleanup(model.close)
            model.encode_documents(['12345'])
            self.assertEqual(post.call_args.kwargs['json']['input'], ['1234'])
        self.assertFalse(os.path.exists(self.failure_path))

    def test_success_does_not_create_failure_file(self):
        from paimem import embeddings as module
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.return_value = httpx.Response(
                200, json={'data': [{'index': 0, 'embedding': [1, 1]}]})
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 'secret-key')
            self.addCleanup(model.close)
            model.encode_queries(['ok'])
        self.assertFalse(os.path.exists(self.failure_path))

    def test_failure_file_error_does_not_replace_original_error(self):
        from paimem import embeddings as module
        with patch('httpx.Client') as client, patch.object(module.os, 'open', side_effect=OSError('private')):
            client.return_value.__enter__.return_value.post.return_value = httpx.Response(400, text='bad')
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 'secret-key')
            self.addCleanup(model.close)
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400'), self.assertLogs(module.__name__) as logs:
                model.encode_queries(['input'])
            self.assertIn('Could not write', ' '.join(logs.output))
            self.assertNotIn('private', ' '.join(logs.output))

    def test_parallel_batches_keep_input_order(self):
        from paimem import embeddings as module
        barrier = threading.Barrier(4)
        gates = [threading.Event() for _ in range(4)]
        completion_order = []

        def handler(request):
            i = int(json.loads(request.content)['input'][0])
            barrier.wait(timeout=5)
            if i < 3:
                self.assertTrue(gates[i + 1].wait(5))
            completion_order.append(i)
            gates[i].set()
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [i + 1, 1]}]})

        real_client = httpx.Client
        with patch.dict(os.environ, {'EMBEDDING_BATCH_SIZE': '1', 'EMBEDDING_MAX_CONCURRENCY': '4'}, clear=True), \
                patch('httpx.Client', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            self.addCleanup(model.close)
            matrix = model.encode_documents(['0', '1', '2', '3'])
        self.assertEqual(completion_order, [3, 2, 1, 0])
        expected = np.array([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(matrix, expected)

    def test_usage_refunds_byte_estimates_before_next_batch(self):
        from paimem import embeddings as module
        real_client = httpx.Client
        def handler(request):
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 1]}],
                                            'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
        with patch.dict(os.environ, {'EMBEDDING_BATCH_SIZE': '1', 'EMBEDDING_MAX_CONCURRENCY': '1',
                                    'EMBEDDING_RPM': '10', 'EMBEDDING_TPM': '6'}, clear=True), \
                patch('httpx.Client', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)), \
                patch.object(module._RateLimiter, '_wait', side_effect=AssertionError('usage should free quota')):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            self.addCleanup(model.close)
            model.encode_documents(['中'] * 4)
        self.assertEqual(model._limiter._tokens, 4)  # Not 12 UTF-8 bytes.
        self.assertEqual(len(model._limiter._events), 4)  # RPM is never refunded.

    def test_quota_reconciliation_wakes_waiters_and_ignores_expired_usage(self):
        from paimem import embeddings as module
        limiter = module._RateLimiter(10, 10)
        first = limiter.acquire(8)
        waiting = threading.Event()
        real_wait = limiter._wait
        def wait_for_quota(delay):
            waiting.set()
            real_wait(delay)
        diagnostics = {}
        with patch.object(limiter, '_wait', side_effect=wait_for_quota), ThreadPoolExecutor(1) as pool:
            future = pool.submit(limiter.acquire, 8, diagnostics)
            try:
                self.assertTrue(waiting.wait(5))
            finally:
                limiter.reconcile(first, 1)
            second = future.result(timeout=5)
        self.assertTrue(diagnostics['tpm_blocked'])
        self.assertFalse(diagnostics['rpm_blocked'])
        self.assertEqual(limiter._tokens, 9)
        with patch.object(module.time, 'monotonic', return_value=second.timestamp + 61):
            limiter.reconcile(first, 5)
            limiter.reconcile(second, 2)
        self.assertEqual(limiter._tokens, 0)
        self.assertEqual(len(limiter._events), 0)

    def test_actual_usage_above_estimate_blocks_until_window_expires(self):
        from paimem import embeddings as module
        clock = [0.0]
        def advance(delay):
            clock[0] += delay
        with patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(module._RateLimiter, '_wait', side_effect=advance):
            limiter = module._RateLimiter(10, 10)
            ticket = limiter.acquire(2)
            limiter.reconcile(ticket, 12)
            diagnostics = {}
            limiter.acquire(1, diagnostics)
        self.assertEqual(clock[0], 60)
        self.assertTrue(diagnostics['tpm_blocked'])
        self.assertEqual(limiter._tokens, 1)

    def test_usage_validation(self):
        from paimem import embeddings as module
        for body in (None, [], {}, {'usage': None}, {'usage': {'total_tokens': True}},
                     {'usage': {'total_tokens': -1}}, {'usage': {'total_tokens': 0}},
                     {'usage': {'total_tokens': '20'}}):
            self.assertIsNone(module._usage_tokens(body))
        self.assertEqual(module._usage_tokens({'usage': {'input_tokens': 4}}), 4)
        self.assertEqual(module._usage_tokens({'usage': {'prompt_tokens': 5}}), 5)
        self.assertEqual(module._usage_tokens({'usage': {'prompt_tokens': 5, 'total_tokens': 6}}), 6)

    def test_parallel_retry_only_resends_failed_batch_and_counts_attempts(self):
        from paimem import embeddings as module
        from collections import Counter
        calls = Counter()
        lock = threading.Lock()
        def handler(request):
            text = json.loads(request.content)['input'][0]
            with lock:
                calls[text] += 1
                fail = text == 'bbb' and calls[text] == 1
            if fail:
                return httpx.Response(429)
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 1]}],
                                            'usage': {'total_tokens': 1}})
        real_client = httpx.Client
        with patch.dict(os.environ, {'EMBEDDING_BATCH_SIZE': '1', 'EMBEDDING_MAX_CONCURRENCY': '2',
                                    'EMBEDDING_RPM': '10', 'EMBEDDING_TPM': '100'}, clear=True), \
                patch('httpx.Client', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)), \
                patch.object(module.time, 'sleep'):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            self.addCleanup(model.close)
            model.encode_documents(['aaa', 'bbb'])
        self.assertEqual(calls, {'aaa': 1, 'bbb': 2})
        self.assertEqual(len(model._limiter._events), 3)
        self.assertEqual(model._limiter._tokens, 5)  # Failed attempt keeps its 3-byte reservation.

    def test_remote_query_and_document_calls_overlap_and_share_quota(self):
        from paimem import embeddings as module
        barrier = threading.Barrier(2)

        def handler(request):
            # Both HTTP calls must enter before either can return. An outer
            # embedding lock (or metadata lock around I/O) breaks this barrier.
            barrier.wait(timeout=5)
            text = json.loads(request.content)['input'][0]
            vector = [3, 4] if text == 'q:query' else [4, 3]
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': vector}]})

        real_client = httpx.Client
        with patch.dict(os.environ, {'EMBEDDING_RPM': '10', 'EMBEDDING_TPM': '1000'}, clear=True), \
                patch('httpx.Client', side_effect=lambda **kw: real_client(
                    transport=httpx.MockTransport(handler), **kw)):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's', query_prefix='q:')
            outer_lock = threading.RLock()

            def encode(query):
                with module.timed_embedding_lock(outer_lock, 'test_embedding', 1, embedder=model):
                    return (model.encode_queries(['query']) if query
                            else model.encode_documents(['document']))

            with ThreadPoolExecutor(max_workers=2) as pool:
                query, document = list(pool.map(encode, [True, False]))
            np.testing.assert_allclose(query, [[.6, .8]])
            np.testing.assert_allclose(document, [[.8, .6]])
            self.assertEqual(model.dimension, 2)
            self.assertEqual(len(model._limiter._events), 2)
            self.assertEqual(model._limiter._tokens, len('q:query') + len('document'))

    def test_concurrent_first_responses_cannot_change_embedding_dimension(self):
        from paimem import embeddings as module
        barrier = threading.Barrier(2)

        def handler(request):
            barrier.wait(timeout=5)
            size = int(json.loads(request.content)['input'][0])
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1] * size}]})

        real_client = httpx.Client
        with patch.dict(os.environ, {}, clear=True), \
                patch('httpx.Client', side_effect=lambda **kw: real_client(
                    transport=httpx.MockTransport(handler), **kw)):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')

            def encode(size):
                try:
                    return model.encode_documents([str(size)])
                except ValueError as exc:
                    return exc

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(encode, [2, 3]))
            successful = [r for r in results if isinstance(r, np.ndarray)]
            errors = [r for r in results if isinstance(r, ValueError)]
            self.assertEqual(len(successful), 1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(successful[0].shape[1], model.dimension)
            self.assertEqual(model.encode_documents([]).shape, (0, model.dimension))

    def test_truncation_strategies_bound_final_http_payload(self):
        from paimem import embeddings as module
        marker = '\n[... truncated ...]\n'
        expected_queries = {
            'head-only': '问:abcdefghijklmnopqrstuvwxyzabcdefghij',
            'tail-only': '问:qrstuvwxyzabcdefghijklmnopqrstuvwxyz',
            'head+tail': '问:abcdefgh' + marker + 'tuvwxyz',
        }
        expected_documents = {
            'head-only': 'abcdefghijklmnopqrstuvwxyzabcdefghijklmn',
            'tail-only': 'mnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz',
            'head+tail': 'abcdefghij' + marker + 'rstuvwxyz',
        }
        for strategy in expected_queries:
            requests = []
            def handler(request):
                batch = json.loads(request.content)['input']
                self.assertTrue(all(len(t.encode('utf-8')) <= 40 for t in batch))
                requests.append(batch)
                return httpx.Response(200, json={'data': [
                    {'index': i, 'embedding': [1, 2]} for i in range(len(batch))]})
            real_client = httpx.Client
            with self.subTest(strategy=strategy), patch.dict(os.environ, {
                    'EMBEDDING_MAX_INPUT_TOKENS': '10', 'EMBEDDING_TPM': '40',
                    'EMBEDDING_TRUNCATION_STRATEGY': strategy}, clear=True), \
                    patch('httpx.Client', side_effect=lambda **kw: real_client(
                        transport=httpx.MockTransport(handler), **kw)):
                model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's', query_prefix='问:')
                self.addCleanup(model.close)
                # Record quota reservations without waiting: reserve bytes for the truncated payload only.
                with patch.object(model._limiter, 'acquire', return_value=None) as acquire:
                    model.encode_queries(['abcdefghijklmnopqrstuvwxyz' * 3])
                    self.assertEqual(requests[-1], [expected_queries[strategy]])
                    self.assertEqual(acquire.call_args.args[0], 40)
                    model.encode_documents(['abcdefghijklmnopqrstuvwxyz' * 3])
                    self.assertEqual(requests[-1], [expected_documents[strategy]])
                    for text in ['short', 'a' * 36, '中' * 12, '😀' * 9]:
                        model.encode_queries([text])
                        self.assertEqual(requests[-1], ['问:' + text])
                    for text in ['中' * 30, '😀' * 30, 'a' + '中😀' * 30 + 'z']:
                        model.encode_queries([text])
                        output = requests[-1][0]
                        self.assertTrue(output.startswith('问:'))
                        self.assertEqual(output.count('问:'), 1)
                        self.assertNotIn('�', output)
                        if strategy == 'head+tail':
                            self.assertEqual(output.count(marker), 1)

    def test_default_truncation_boundary_and_minimum_budget(self):
        from paimem import embeddings as module
        with patch.dict(os.environ, {}, clear=True):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            self.addCleanup(model.close)
            self.assertEqual(model.truncation_strategy, 'head+tail')
            text = 'x' * 32768
            self.assertEqual(model._prepare_input(text, ''), text)
            output = model._prepare_input(text + 'y', '')
            self.assertEqual(len(output.encode()), 32768)
            self.assertIn('\n[... truncated ...]\n', output)
            self.assertTrue(output.endswith('y'))
        with patch.dict(os.environ, {'EMBEDDING_MAX_INPUT_TOKENS': '6'}, clear=True):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's', query_prefix='q:')
            self.addCleanup(model.close)
            # One byte remains for the head; a zero-byte tail must not include the entire input.
            self.assertEqual(model._prepare_input('abcdefghijklmnopqrstuvwxyz', 'q:'),
                             'q:a\n[... truncated ...]\n')

    def test_invalid_truncation_configuration_before_worker_creation(self):
        from paimem import embeddings as module
        cases = [({'EMBEDDING_TRUNCATION_STRATEGY': 'invalid'}, '', 'EMBEDDING_TRUNCATION_STRATEGY'),
                 ({'EMBEDDING_MAX_INPUT_TOKENS': '5'}, '', 'EMBEDDING_MAX_INPUT_TOKENS'),
                 ({'EMBEDDING_MAX_INPUT_TOKENS': '6'}, '问:', 'EMBEDDING_MAX_INPUT_TOKENS'),
                 ({'EMBEDDING_MAX_INPUT_TOKENS': '1', 'EMBEDDING_TRUNCATION_STRATEGY': 'tail-only'},
                  '问:', 'EMBEDDING_MAX_INPUT_TOKENS')]
        for env, prefix, error in cases:
            with self.subTest(env=env, prefix=prefix), patch.dict(os.environ, env, clear=True), \
                    patch.object(module, 'ThreadPoolExecutor') as executor:
                with self.assertRaisesRegex(ValueError, error):
                    module.RemoteEmbedder('v4', 'https://example.com/v1', 's', query_prefix=prefix)
                executor.assert_not_called()

    def test_batch_and_quota_validation(self):
        from paimem import embeddings as module
        for key, values in {
            'EMBEDDING_BATCH_SIZE': ['0', '-1', '1.5', 'bad'],
            'EMBEDDING_MAX_CONCURRENCY': ['0', '-1', '1.5', 'bad'],
            'EMBEDDING_MAX_INPUT_TOKENS': ['0', '-1', '1.5', 'bad'],
            'EMBEDDING_RPM': ['-1', '1.5', 'bad'],
            'EMBEDDING_TPM': ['-1', '1.5', 'bad'],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}, clear=True):
                    with self.assertRaisesRegex(ValueError, key):
                        module.RemoteEmbedder('v4', 'https://example.com/v1', 'secret')
                    if key == 'EMBEDDING_BATCH_SIZE':
                        with self.assertRaisesRegex(ValueError, key):
                            module.SentenceTransformerEmbedder('unused')
        with patch.dict(os.environ, {}, clear=True):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            self.addCleanup(model.close)
            self.assertEqual(model.batch_size, 10)
            self.assertEqual(model.max_concurrency, 4)
            self.assertEqual(model.max_input_tokens, 8192)
        for value, expected in [(None, 64), ('3', 3)]:
            env = {} if value is None else {'EMBEDDING_BATCH_SIZE': value}
            fake_model = SimpleNamespace(device='cpu', encode=lambda texts, **kw: np.ones((len(texts), 2)))
            with patch.dict(os.environ, env, clear=True), patch.dict('sys.modules', {
                'sentence_transformers': SimpleNamespace(SentenceTransformer=lambda *a, **kw: fake_model)
            }), patch.object(module, 'inference_guard'):
                model = module.SentenceTransformerEmbedder('local')
                with patch.object(fake_model, 'encode', wraps=fake_model.encode) as encode:
                    model.encode_documents(['a'])
                    self.assertEqual(encode.call_args.kwargs['batch_size'], expected)

    def test_remote_split_prefix_and_shared_quota(self):
        from paimem import embeddings as module
        clock = [0.0]
        requests = []
        def sleep(delay):
            clock[0] += delay
        def handler(request):
            import json
            batch = json.loads(request.content)['input']
            requests.append((clock[0], batch))
            return httpx.Response(200, json={'data': [
                {'index': i, 'embedding': [i + 1, 2]} for i in range(len(batch))]})
        real_client = httpx.Client
        with patch.dict(os.environ, {'EMBEDDING_MAX_CONCURRENCY': '1', 'EMBEDDING_BATCH_SIZE': '2', 'EMBEDDING_RPM': '2',
                                    'EMBEDDING_TPM': '6'}, clear=True), \
                patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(module._RateLimiter, '_wait', side_effect=sleep), \
                patch.object(module.time, 'sleep', side_effect=sleep), \
                patch('httpx.Client', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's', query_prefix='q:')
            model.encode_documents(['abc', 'de', '中', 'f'])
            model.encode_queries(['x'])
            self.assertEqual(requests, [(0, ['abc', 'de']), (60, ['中', 'f']), (120, ['q:x'])])
            with self.assertRaisesRegex(ValueError, 'EMBEDDING_TPM'):
                model.encode_documents(['ok', 'too long'])
            with self.assertRaisesRegex(ValueError, 'EMBEDDING_TPM'):
                model.encode_queries(['12345'])
            self.assertEqual(len(requests), 3)
        # TPM can split a batch even when the configured batch size allows more.
        with patch.dict(os.environ, {'EMBEDDING_BATCH_SIZE': '10', 'EMBEDDING_TPM': '4'}, clear=True):
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            self.assertEqual(list(model._batches(['abc', 'de', 'f'])), [['abc'], ['de', 'f']])

    def test_retry_reserves_quota_and_honors_retry_after(self):
        from paimem import embeddings as module
        clock = [0.0]
        times = []
        def sleep(delay):
            clock[0] += delay
        def post(*args, **kwargs):
            times.append(clock[0])
            if len(times) == 1:
                return httpx.Response(429, headers={'Retry-After': '75'})
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [3, 4]}]})
        with patch.dict(os.environ, {'EMBEDDING_RPM': '1'}, clear=True), \
                patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(module._RateLimiter, '_wait', side_effect=sleep), \
                patch.object(module.time, 'sleep', side_effect=sleep), patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect = post
            model = module.RemoteEmbedder('v4', 'https://example.com/v1', 's')
            model.encode_documents(['x'])
            model.encode_queries(['y'])
        self.assertEqual(times, [0, 75, 135])
        from email.utils import formatdate
        with patch.object(module.time, 'time', return_value=1000):
            self.assertEqual(module._retry_after(formatdate(1020, usegmt=True)), 20)
        for value in [None, 'bad', 'NaN', 'inf', '-1']:
            self.assertEqual(module._retry_after(value), 0)

    def test_rolling_window_and_concurrent_reservations(self):
        from paimem import embeddings as module
        from concurrent.futures import ThreadPoolExecutor
        clock = [0.0]
        def sleep(delay):
            clock[0] += delay
        with patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(module._RateLimiter, '_wait', side_effect=sleep), \
                patch.object(module.time, 'sleep', side_effect=sleep):
            limiter = module._RateLimiter(2, 0)
            limiter.acquire(2)
            clock[0] = 10
            limiter.acquire(2)
            clock[0] = 20
            limiter.acquire(2)
            self.assertEqual(clock[0], 60)
            limiter.acquire(2)
            self.assertEqual(clock[0], 70)
        # All workers reserve in one window; no increments can be lost.
        with patch.object(module.time, 'monotonic', return_value=0):
            limiter = module._RateLimiter(100, 200)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(limiter.acquire, [2] * 100))
            self.assertEqual(len(limiter._events), 100)
            self.assertEqual(limiter._tokens, 200)


    def test_local_default_and_explicit_remote_settings(self):
        for package in PACKAGES:
            module = importlib.import_module(package + '.embeddings')
            with self.subTest(package=package), patch.object(module, 'SentenceTransformerEmbedder') as local:
                module.build_embedder()
                local.assert_called_once_with(model_name='BAAI/bge-small-en-v1.5', device=None, query_prefix=None)
                local.reset_mock()
                model = module.build_embedder('text-embedding-v4', backend='remote',
                                             base_url='https://example.com/v1/', api_key='secret')
                local.assert_not_called()
                self.assertEqual(model.query_prefix, '')
                self.assertNotIn('secret', model.identity)
                rotated = module.build_embedder('text-embedding-v4', backend='remote',
                                               base_url='https://example.com/v1', api_key='rotated')
                self.assertEqual(model.identity, rotated.identity)

    def test_invalid_configuration(self):
        for package in PACKAGES:
            module = importlib.import_module(package + '.embeddings')
            for field in ('model_name', 'base_url', 'api_key'):
                kwargs = dict(model_name='text-embedding-v4', backend='remote',
                              base_url='https://example.com/v1', api_key='secret')
                kwargs[field] = ''
                with self.subTest(package=package, field=field), self.assertRaises(ValueError):
                    module.build_embedder(**kwargs)
            for url in ('file:///tmp/api', 'https://user:secret@example.com/v1',
                        'https://example.com/v1?key=secret', 'https://example.com/v1/embeddings'):
                with self.assertRaises(ValueError):
                    module.build_embedder('v4', backend='remote', base_url=url, api_key='secret')
            with self.assertRaises(ValueError):
                module.build_embedder('v4', backend='typo')

    def test_batch_order_normalization_query_and_empty_input(self):
        for package in PACKAGES:
            module = importlib.import_module(package + '.embeddings')
            requests = []
            def handler(request):
                import json
                body = json.loads(request.content)
                requests.append(body)
                self.assertEqual(request.url.path, '/v1/embeddings')
                self.assertEqual(request.headers['Authorization'], 'Bearer secret')
                self.assertEqual(body['model'], 'text-embedding-v4')
                return httpx.Response(200, json={'data': [
                    {'index': i, 'embedding': [float(text.removeprefix('query:')) + 1, 2]}
                    for i, text in reversed(list(enumerate(body['input'])))
                ]})
            real_client = httpx.Client
            with self.subTest(package=package), patch('httpx.Client', side_effect=lambda **kw: real_client(
                    transport=httpx.MockTransport(handler), **kw)):
                model = module.build_embedder('text-embedding-v4', backend='remote',
                                             base_url='https://example.com/v1', api_key='secret', query_prefix='query:')
                self.assertEqual(model.encode_documents([]).shape, (0, 0))
                matrix = model.encode_documents([str(i) for i in range(23)])
                self.assertEqual([len(r['input']) for r in requests], [10, 10, 3])
                self.assertEqual(matrix.dtype, np.float32)
                expected = np.array([[i + 1, 2] for i in range(23)], dtype=np.float32)
                expected /= np.linalg.norm(expected, axis=1, keepdims=True)
                np.testing.assert_allclose(matrix, expected, rtol=1e-6)
                model.encode_queries(['1'])
                self.assertEqual(requests[-1]['input'], ['query:1'])
                self.assertEqual(model.encode_documents([]).shape, (0, 2))

    def test_bad_responses_and_errors_are_sanitized(self):
        for package in PACKAGES:
            module = importlib.import_module(package + '.embeddings')
            responses = [
                httpx.Response(401, text='secret'),
                httpx.Response(200, text='not json secret'),
                httpx.Response(200, json={'data': []}),
                httpx.Response(200, json={'data': [{'index': 1, 'embedding': [1, 2]}]}),
                httpx.Response(200, json={'data': [{'index': 0, 'embedding': [0, 0]}]}),
                httpx.Response(200, json={'data': [{'index': 0, 'embedding': ['NaN', 1]}]}),
                httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 2]},
                                                   {'index': 0, 'embedding': [1, 2]}]}),
            ]
            for response in responses:
                with patch('httpx.Client') as client:
                    client.return_value.__enter__.return_value.post.return_value = response
                    model = module.build_embedder('v4', backend='remote', base_url='https://example.com/v1', api_key='secret')
                    with self.assertRaises((RuntimeError, ValueError)) as caught:
                        model.encode_documents(['text'])
                    self.assertNotIn('secret', str(caught.exception))
                    self.assertEqual(client.return_value.__enter__.return_value.post.call_count, 1)
            with patch('httpx.Client') as client, patch.object(module.time, 'sleep'):
                post = client.return_value.__enter__.return_value.post
                post.side_effect = httpx.ReadTimeout('secret')
                with self.assertRaisesRegex(RuntimeError, 'ReadTimeout') as caught:
                    model.encode_queries(['text'])
                self.assertNotIn('secret', str(caught.exception))
                post.side_effect = None
                post.return_value = httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 2]}]})
                model.encode_documents(['text'])
                post.return_value = httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 2, 3]}]})
                with self.assertRaises(ValueError):
                    model.encode_queries(['text'])

    @patch.dict(os.environ, {'EMBEDDING_MAX_CONCURRENCY': '1'})
    def test_transient_failure_retries_only_failed_batch(self):
        for package in PACKAGES:
            module = importlib.import_module(package + '.embeddings')
            for failure in (httpx.ConnectError('secret'), httpx.ReadTimeout('secret'),
                            httpx.RemoteProtocolError('secret'), 429, 502, 503, 504):
                with self.subTest(package=package, failure=type(failure).__name__ if isinstance(failure, Exception) else failure), \
                        patch('httpx.Client') as client, patch.object(module.time, 'sleep') as sleep, \
                        self.assertLogs(module.__name__, level='WARNING') as logs:
                    post = client.return_value.__enter__.return_value.post
                    first = httpx.Response(200, json={'data': [
                        {'index': i, 'embedding': [3, 4]} for i in range(10)]})
                    last = httpx.Response(200, json={'data': [{'index': 0, 'embedding': [4, 3]}]})
                    post.side_effect = [first, failure if isinstance(failure, Exception) else httpx.Response(failure, text='secret'), last]
                    model = module.build_embedder('v4', backend='remote', base_url='https://example.com/v1', api_key='secret')
                    result = model.encode_documents([str(i) for i in range(11)])
                    self.assertEqual([call.kwargs['json']['input'] for call in post.call_args_list],
                                     [[str(i) for i in range(10)], ['10'], ['10']])
                    np.testing.assert_allclose(result[:10], np.tile([0.6, 0.8], (10, 1)), rtol=1e-6)
                    np.testing.assert_allclose(result[-1], [0.8, 0.6], rtol=1e-6)
                    sleep.assert_called_once_with(1.0)
                    self.assertIn('batch=2', logs.output[0])
                    self.assertNotIn('secret', ' '.join(logs.output))

    def test_retry_exhaustion_and_nonretryable_failures(self):
        for package in PACKAGES:
            module = importlib.import_module(package + '.embeddings')
            for failure, attempts in ((httpx.ConnectTimeout('secret'), 3), (503, 3),
                                      (httpx.UnsupportedProtocol('secret'), 1), (400, 1), (401, 1), (403, 1)):
                with self.subTest(package=package, attempts=attempts), patch('httpx.Client') as client, \
                        patch.object(module.time, 'sleep') as sleep, self.assertLogs(module.__name__) as logs:
                    post = client.return_value.__enter__.return_value.post
                    if isinstance(failure, Exception):
                        post.side_effect = failure
                    else:
                        post.return_value = httpx.Response(failure, text='secret')
                    model = module.build_embedder('v4', backend='remote', base_url='https://example.com/v1', api_key='secret')
                    with self.assertRaisesRegex(RuntimeError, f'attempts={attempts}') as caught:
                        model.encode_queries(['private input'])
                    self.assertEqual(post.call_count, attempts)
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0] if attempts == 3 else [])
                    diagnostic = str(caught.exception) + ' '.join(logs.output)
                    self.assertNotIn('secret', diagnostic)
                    self.assertNotIn('private input', diagnostic)
                    expected = type(failure).__name__ if isinstance(failure, Exception) else f'HTTP {failure}'
                    self.assertIn(expected, diagnostic)

    def test_api_environment_and_overrides(self):
        env = {'EMBEDDING_BACKEND': 'remote', 'EMBEDDING_MODEL': 'text-embedding-v4',
               'EMBEDDING_BASE_URL': 'https://example.com/v1', 'EMBEDDING_API_KEY': 'secret'}
        for package in PACKAGES:
            api = importlib.import_module(package + '.api')
            with self.subTest(package=package), patch.dict(os.environ, env, clear=True), \
                    patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), \
                    patch.object(api, 'build_embedder') as build, patch.object(api, 'SQLiteMemoryStore'):
                api.get_store.cache_clear()
                try:
                    api.get_store()
                    self.assertEqual(build.call_args.args[0], 'text-embedding-v4')
                    self.assertEqual(build.call_args.kwargs, dict(backend='remote', base_url=env['EMBEDDING_BASE_URL'], api_key='secret'))
                    api.get_store.cache_clear()
                    api.configure_store(embedding_backend='local', embedding_model='/local/bge', embedding_api_key='other')
                    api.get_store()
                    self.assertEqual(build.call_args.args[0], '/local/bge')
                    self.assertEqual(build.call_args.kwargs['backend'], 'local')
                finally:
                    api.get_store.cache_clear()

    def test_cli_identity_and_environment(self):
        for package in PACKAGES:
            cli = importlib.import_module(package + '.cli')
            with patch.dict(os.environ, {'EMBEDDING_MODEL': 'v4', 'EMBEDDING_BACKEND': 'remote'}, clear=True):
                self.assertEqual(cli.parser().parse_args(['index', '--documents', 'd', '--output', 'o']).embedding_model, 'v4')
                self.assertEqual(cli.parser().parse_args(['index', '--documents', 'd', '--output', 'o', '--embedding-model', 'override']).embedding_model, 'override')
            index = SimpleNamespace(metadata={'embedding_model': 'v4', 'embedding_backend': 'remote', 'embedding_identity': 'saved'})
            with patch.dict(os.environ, {}, clear=True), patch.object(cli.FaissChunkIndex, 'load', return_value=index), \
                    patch.object(cli, 'build_embedder', return_value=SimpleNamespace(identity='saved')) as build:
                cli.load_rag('index', None, None, 'none', '')
                self.assertEqual(build.call_args.kwargs['backend'], 'remote')
                build.return_value.identity = 'different'
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    cli.load_rag('index', None, None, 'none', '')
                index.metadata.pop('embedding_identity')
                with self.assertRaisesRegex(ValueError, 'Legacy index'):
                    cli.load_rag('index', None, None, 'none', '')

    def test_sqlite_model_switch_requires_empty_memories_and_requests(self):
        class StaticEmbedder:
            def __init__(self, identity, dimension):
                self.identity = identity
                self.dimension = dimension

            def encode_documents(self, texts):
                return np.full((len(texts), self.dimension), 1 / np.sqrt(self.dimension), dtype=np.float32)

            encode_queries = encode_documents

        for package in PACKAGES:
            module = importlib.import_module(package + '.memory_store')
            for state in ('empty', 'populated', 'requests_only', 'memories_only', 'cleared'):
                with self.subTest(package=package, state=state), tempfile.TemporaryDirectory() as directory:
                    path = directory + '/memory.db'
                    first = StaticEmbedder('local-bge', 2)
                    second = StaticEmbedder('remote-v4', 3)
                    store = module.SQLiteMemoryStore(path, first)
                    if state != 'empty':
                        store.add('r1', 'u1', 's1', [module.MemoryMessage('user', 'Remember Paris')])
                    if state == 'cleared':
                        store.clear()
                    elif state in ('requests_only', 'memories_only'):
                        # Construct partial cleanup states to test both guards independently.
                        with sqlite3.connect(path) as connection:
                            table = 'memories' if state == 'requests_only' else 'add_requests'
                            connection.execute('DELETE FROM ' + table)
                    with sqlite3.connect(path) as connection:
                        before = tuple(connection.execute(
                            'SELECT (SELECT COUNT(*) FROM memories), (SELECT COUNT(*) FROM add_requests)'
                        ).fetchone())
                    if state in ('empty', 'cleared'):
                        switched = module.SQLiteMemoryStore(path, second)
                        switched.add('r2', 'u1', 's1', [module.MemoryMessage('user', 'Remember London')])
                        restarted = module.SQLiteMemoryStore(path, second)
                        self.assertTrue(restarted.search('u1', 'London', 1))
                        expected_identity = second.identity
                    else:
                        with self.assertRaisesRegex(ValueError, 'embedding model does not match'):
                            module.SQLiteMemoryStore(path, second)
                        expected_identity = first.identity
                    with sqlite3.connect(path) as connection:
                        identity = connection.execute(
                            "SELECT value FROM store_metadata WHERE key = 'embedding_identity'"
                        ).fetchone()[0]
                        self.assertEqual(identity, expected_identity)
                        if state not in ('empty', 'cleared'):
                            after = tuple(connection.execute(
                                'SELECT (SELECT COUNT(*) FROM memories), (SELECT COUNT(*) FROM add_requests)'
                            ).fetchone())
                            self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main()
