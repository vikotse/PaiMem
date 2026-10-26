import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from paimem.long_question import LongQuestionConfig, LLMLongQuestionExtractor
from paimem.memory_store import SQLiteMemoryStore, MemoryMessage
from paimem.near import NearConfig


class Embedder:
    identity = 'long-question-test'

    def __init__(self):
        self.queries = []

    def encode_documents(self, texts):
        return np.asarray([[1., 0.5] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        self.queries.extend(texts)
        return np.asarray([[1., 0.5] for _ in texts], dtype=np.float32)


class LongQuestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.extractor = Mock()
        self.extractor.extract.return_value = 'target'
        self.question = 'headfact ' + 'middlefact ' * 8 + 'tailfact'

    def store(self, enabled=True, **kwargs):
        return SQLiteMemoryStore(
            Path(self.tmp.name) / 'memory.db', Embedder(), chunk_tokens=3, overlap_tokens=1,
            long_question_config=LongQuestionConfig(enabled=enabled, threshold_chars=40, head_chars=10, tail_chars=10),
            long_question_extractor=self.extractor, **kwargs)

    def test_full_question_and_history_share_retrieval_without_persistence(self):
        store = self.store(retrieval_mode='hybrid')
        store.add('r', 'u', 's', [MemoryMessage('user', 'historyfact')])
        before = store.stats()
        hits, detail = store.search('u', self.question, 100, diagnostics=True)
        text = '\n'.join(h.content for h in hits)
        for fact in ('headfact', 'middlefact', 'tailfact', 'historyfact'):
            self.assertIn(fact, text)
        self.extractor.extract.assert_called_once_with(self.question[:10], self.question[-10:])
        self.assertIn('target', store.embedder.queries)
        self.assertEqual(detail['long_question']['evidence_chars'], len(self.question))
        self.assertEqual(detail['persistent_count'], 1)
        self.assertEqual(store.stats(), before)
        self.assertEqual([h.content for h in store.search('u', 'short', 100)], ['[user] historyfact'])
        self.assertEqual(store.search('other', 'short', 100), [])

    def test_default_and_threshold_and_runtime_toggle(self):
        store = self.store(enabled=False)
        self.assertEqual(store.search('u', self.question, 10), [])
        self.extractor.extract.assert_not_called()
        store.update_config(long_question_enabled=True)
        self.assertEqual(store.search('u', 'x' * 40, 10), [])
        self.extractor.extract.assert_not_called()
        self.assertTrue(store.search('u', 'x' * 41, 10))
        store.update_config(long_question_enabled=False)
        self.assertEqual(store.search('u', self.question, 10), [])

    def test_failure_falls_back_without_writes(self):
        store = self.store()
        store.add('r', 'u', 's', [MemoryMessage('user', 'history')])
        for value in (None, '', ' ', {}, 'x' * 21):
            self.extractor.extract.return_value = value
            hits, detail = store.search('u', self.question, 10, diagnostics=True)
            self.assertEqual(detail['long_question']['status'], 'fallback')
            self.assertEqual(len(hits), 1)
        self.extractor.extract.side_effect = TimeoutError('secret')
        _, detail = store.search('u', self.question, 10, diagnostics=True)
        self.assertNotIn('secret', str(detail))
        self.assertEqual(store.stats()['memories'], 1)
        self.extractor.extract.side_effect = None
        self.extractor.extract.return_value = 'target'
        with patch.object(store.embedder, 'encode_documents', side_effect=ValueError('secret')):
            self.assertEqual(len(store.search('u', self.question, 10)), 1)

    def test_same_add_preparation_and_repeated_ids(self):
        store = self.store(context_messages=1, pack_span=3, result_window=1)
        pending, _ = store._prepare_memories('r', [MemoryMessage('user', self.question)], {})
        first = store.search('u', self.question, 100)
        second = store.search('u', self.question, 100)
        self.assertEqual([h.id for h in first], [h.id for h in second])
        text = '\n'.join(h.content for h in first)
        for item in pending:
            self.assertIn(item[2], text)
        self.assertEqual(store.stats()['requests'], 0)

    def test_parallel_requests_do_not_mix_and_near_is_local(self):
        store = self.store(near_config=NearConfig(enabled=True))
        questions = ['alpha ' * 20, 'bravo ' * 20]
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda q: store.search('u', q, 100), questions))
        for hits, own, other in zip(results, ['alpha', 'bravo'], ['bravo', 'alpha']):
            text = '\n'.join(h.content for h in hits)
            self.assertIn(own, text)
            self.assertNotIn(other, text)
        self.assertEqual(store.search('u', 'short', 10), [])

    def test_config_validation_and_atomic_update(self):
        for kwargs in ({'enabled': 1}, {'threshold_chars': 0}, {'head_chars': 9000}, {'timeout': float('nan')}):
            with self.assertRaises(ValueError):
                LongQuestionConfig(**kwargs)
        store = self.store(enabled=False)
        store.long_question_extractor = None
        before = store.get_config()
        for value in (True, 1, 'true'):
            with self.assertRaises(ValueError):
                store.update_config(long_question_enabled=value, result_window=3)
            self.assertEqual(store.get_config(), before)

    def test_api_and_cli_configuration(self):
        from paimem.api import _build_long_question_config, STORE_INIT_OVERRIDES, ConfigPatch
        from paimem.cli import parse_setting
        with patch.dict(STORE_INIT_OVERRIDES, {}, clear=True), patch.dict(os.environ, {
            'MEMORY_LONG_QUESTION_ENABLED': 'true', 'MEMORY_LONG_QUESTION_THRESHOLD_CHARS': '9000'
        }, clear=True):
            config = _build_long_question_config()
            self.assertTrue(config.enabled)
            self.assertEqual(config.threshold_chars, 9000)
        self.assertTrue(ConfigPatch(long_question_enabled=True).long_question_enabled)
        with self.assertRaises(ValueError):
            ConfigPatch(long_question_head_chars=200)
        self.assertEqual(parse_setting('long_question_enabled=true'), ('long_question_enabled', True))

    def test_conflict_and_sira_receive_extracted_query(self):
        from paimem.conflict_strategy import JudgmentQueries
        from paimem.sira import SiraConfig
        strategy = Mock()
        strategy.classify.return_value = JudgmentQueries('positive', 'negative', 'scope')
        strategy.assess.return_value = None
        expander = Mock()
        expander.expand.return_value = ['middlefact']
        store = self.store(conflict_strategy=strategy, retrieval_mode='hybrid',
                           sira_config=SiraConfig(enabled=True), sira_expander=expander)
        hits, detail = store.search('u', self.question, 20, diagnostics=True)
        strategy.classify.assert_called_once_with('target')
        expander.expand.assert_called_once_with('target', store.sira_config)
        self.assertTrue(hits)
        self.assertEqual(detail['persistent_count'], 0)
        self.assertTrue(all(hit.id.startswith('question_') for hit in hits))

    def test_llm_contract(self):
        client = Mock()
        choice = Mock(finish_reason='stop')
        choice.message.content = '{"query":"target"}'
        client.chat.completions.create.return_value.choices = [choice]
        extractor = LLMLongQuestionExtractor(client, 'model')
        self.assertEqual(extractor.extract('head', 'tail'), 'target')
        payload = client.chat.completions.create.call_args.kwargs
        self.assertEqual(json.loads(payload['messages'][1]['content']), {'head': 'head', 'tail': 'tail'})
        choice.finish_reason = 'length'
        with self.assertRaises(ValueError):
            extractor.extract('head', 'tail')


class ExtraBodyTests(unittest.TestCase):
    def test_request_bodies(self):
        for body in (None, {}, {"enable_thinking": False}, {"thinking": {"type": "disabled"}}):
            with self.subTest(body=body):
                client = Mock()
                choice = Mock(finish_reason='stop')
                choice.message.content = '{"query":"target"}'
                client.chat.completions.create.return_value.choices = [choice]
                self.assertEqual(LLMLongQuestionExtractor(client, 'model', body).extract('head', 'tail'), 'target')
                payload = client.chat.completions.create.call_args.kwargs
                if body:
                    self.assertEqual(payload['extra_body'], body)
                else:
                    self.assertNotIn('extra_body', payload)

    def test_configuration_bodies(self):
        from paimem.api import _build_long_question_config, _build_long_question_extractor, STORE_INIT_OVERRIDES, ConfigPatch
        for raw, expected in [('', None), ('{}', {}), ('{"enable_thinking":false}', {'enable_thinking': False}), ('{"thinking":{"type":"disabled"}}', {'thinking': {'type': 'disabled'}})]:
            with self.subTest(raw=raw), patch.dict(STORE_INIT_OVERRIDES, {}, clear=True), patch.dict(os.environ, {
                'MEMORY_LONG_QUESTION_EXTRA_BODY': raw,
                'MEMORY_LONG_QUESTION_MODEL': 'model',
                'MEMORY_LONG_QUESTION_API_KEY': 'test-key',
                'MEMORY_LONG_QUESTION_API_BASE': 'https://example.com/v1',
            }, clear=True), patch('openai.OpenAI'):
                config = _build_long_question_config()
                self.assertEqual(config.extra_body, expected)
                self.assertEqual(config.settings()['long_question_extra_body'], expected)
                self.assertEqual(_build_long_question_extractor(config).extra_body, expected)
        with patch.dict(STORE_INIT_OVERRIDES, {'long_question_extra_body': {'enable_thinking': False}}, clear=True), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_build_long_question_config().extra_body, {'enable_thinking': False})
        with self.assertRaises(ValueError):
            ConfigPatch(long_question_extra_body={})  # startup-only
        for raw in ('invalid', '[]', 'null', 'true', '1', '{"x":NaN}'):
            with self.subTest(invalid=raw), patch.dict(STORE_INIT_OVERRIDES, {}, clear=True), patch.dict(os.environ, {'MEMORY_LONG_QUESTION_EXTRA_BODY': raw}, clear=True):
                with self.assertRaises(ValueError):
                    _build_long_question_config()
        for invalid in ([], True, 1):
            with self.assertRaises(ValueError):
                LongQuestionConfig(extra_body=invalid)
            with self.assertRaises(ValueError):
                LLMLongQuestionExtractor(Mock(), 'model', invalid)
