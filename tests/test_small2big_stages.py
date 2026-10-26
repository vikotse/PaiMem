"""Contracts for independently expanding reranker text and MP vectors."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from paimem import api
from paimem.cli import RUNTIME_SETTINGS
from paimem.memory_store import MemoryMessage, SQLiteMemoryStore
from paimem.mempursuit import MemPursuitConfig, select
from paimem.retrieval import retrieve_chunks


class Embedder:
    def __init__(self):
        self.calls = []

    def encode_documents(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[len(t), sum(map(ord, t)) % 17 + 1] for t in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append(list(documents))
        return np.arange(len(documents), 0, -1, dtype=np.float32)


class Small2BigStagesTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id=str(i), session_id='one' if i < 3 else 'two', role='user',
                          content=t, dimension=2, embedding=np.array([1., 0.], dtype=np.float32).tobytes())
                     for i, t in enumerate(['A', 'B', 'C', 'D'])]
        self.expanded = ['A\nB', 'A\nB\nC', 'B\nC', 'D']
        self.emb, self.rr = Embedder(), Reranker()

    def run_query(self, **kwargs):
        args = dict(embedder=self.emb, embedding_lock=threading.Lock(),
                    render_content=lambda row, names: row['content'], retrieval_mode='dense',
                    reranker=self.rr, rerank_candidates=4,
                    mempursuit_config=MemPursuitConfig(enabled=True, candidates=4), diagnostics=True)
        args.update(kwargs)
        return retrieve_chunks(self.rows, {}, 'query', 2, **args)

    def test_four_combinations_use_correct_text_and_vectors(self):
        for rr in [False, True]:
            for mp in [False, True]:
                with self.subTest(rr=rr, mp=mp), patch('paimem.retrieval.select_memories', wraps=select) as choose:
                    self.emb.calls.clear(); self.rr.calls.clear()
                    result = self.run_query(rerank_small2big_enabled=rr, mempursuit_small2big_enabled=mp)
                    self.assertEqual(self.rr.calls, [self.expanded if rr else ['A', 'B', 'C', 'D']])
                    self.assertEqual(self.emb.calls, [self.expanded] if mp else [])
                    expected = Embedder().encode_documents(self.expanded) if mp else np.tile([1., 0.], (4, 1))
                    np.testing.assert_array_equal(choose.call_args.args[1], expected)
                    self.assertEqual(set(result.positions), set(range(4)))
                    self.assertEqual(result.diagnostics['small2big_stages']['mempursuit_active'], mp)
                    self.assertEqual(result.diagnostics['small2big_stages']['rerank_active'], rr)

    def test_disabled_defaults_are_exactly_compatible(self):
        a = self.run_query(); b = self.run_query(rerank_small2big_enabled=False, mempursuit_small2big_enabled=False)
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.scores, b.scores)
        self.assertEqual(a.diagnostics, b.diagnostics)
        self.assertEqual(self.emb.calls, [])

    def test_inactive_stages_do_not_encode_or_rerank(self):
        r = self.run_query(reranker=None, mempursuit_config=MemPursuitConfig(),
                           rerank_small2big_enabled=True, mempursuit_small2big_enabled=True)
        self.assertFalse(r.diagnostics['small2big_stages']['rerank_active'])
        self.assertFalse(r.diagnostics['small2big_stages']['mempursuit_active'])
        self.assertEqual(self.emb.calls, []); self.assertEqual(self.rr.calls, [])

    def test_only_mp_pool_is_encoded_and_vectors_are_validated(self):
        with patch.object(self.rr, 'score', return_value=[4., 3., -1., -2.]):
            self.run_query(rerank_score_threshold=0., mempursuit_small2big_enabled=True)
        self.assertEqual(self.emb.calls, [self.expanded[:2]])
        for value in [np.zeros((4, 3)), np.full((4, 2), np.nan)]:
            with patch.object(self.emb, 'encode_documents', return_value=value):
                with self.assertRaisesRegex(ValueError, 'small2big document vectors'):
                    self.run_query(mempursuit_small2big_enabled=True)

    def test_store_runtime_switches_and_fresh_neighbors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp)/'memory.db', self.emb, reranker=self.rr,
                                      result_window=0, pack_span=1,
                                      mempursuit_config=MemPursuitConfig(enabled=True, candidates=4))
            store.add('r1', 'u', 's', [MemoryMessage('user', 'OLD')])
            store.add('private', 'other-user', 's', [MemoryMessage('user', 'PRIVATE')])
            before = store.get_config()
            with self.assertRaises(ValueError):
                store.update_config(rerank_small2big_enabled=True, mempursuit_small2big_enabled='true')
            self.assertEqual(before, store.get_config())
            store.update_config(rerank_small2big_enabled=True, mempursuit_small2big_enabled=True)
            store.search('u', 'query', 2)
            store.add('r2', 'u', 's', [MemoryMessage('user', 'NEW')])
            hits = store.search('u', 'query', 2)
            self.assertTrue(all('OLD' in t and 'NEW' in t for t in self.rr.calls[-1]))
            self.assertTrue(all('PRIVATE' not in t for t in self.rr.calls[-1]))
            self.assertTrue(all(not ('OLD' in h.content and 'NEW' in h.content) for h in hits))
            store.update_config(rerank_small2big_enabled=False, mempursuit_small2big_enabled=False)
            self.emb.calls.clear(); store.search('u', 'query', 2)
            self.assertEqual(self.emb.calls, [])
            self.assertTrue(all(not ('OLD' in t and 'NEW' in t) for t in self.rr.calls[-1]))

    def test_api_and_cli_switches_and_strict_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMemoryStore(Path(tmp)/'memory.db', self.emb)
            with patch.dict(api.app.dependency_overrides, {api.get_store: lambda: store}), patch.dict(os.environ, {'MEMORY_API_KEY': 'test', 'SKIP_MODEL_STARTUP': 'true'}):
                with TestClient(api.app, headers={"X-Api-Key": "test"}) as client:
                    response = client.post('/config', json={'rerank_small2big_enabled': True, 'mempursuit_small2big_enabled': True})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(store.get_config()['rerank_small2big_enabled'])
                    response = client.post('/config', json={'mempursuit_small2big_enabled': 'false'})
                    self.assertEqual(response.status_code, 422)
            for key in ['rerank_small2big_enabled', 'mempursuit_small2big_enabled']:
                self.assertIs(RUNTIME_SETTINGS[key], bool)
                self.assertIn(key, api._STORE_SETTING_KEYS)
                with patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True):
                    for raw, expected in [('true', True), ('1', True), ('false', False), ('0', False)]:
                        with patch.dict(os.environ, {'MEMORY_'+key.upper(): raw}):
                            self.assertEqual(api._stage_boolean(key), expected)
                    with patch.dict(os.environ, {'MEMORY_'+key.upper(): 'typo'}):
                        with self.assertRaises(ValueError): api._stage_boolean(key)
                with patch.dict(api.STORE_INIT_OVERRIDES, {key: False}):
                    with patch.dict(os.environ, {'MEMORY_'+key.upper(): 'true'}):
                        self.assertFalse(api._stage_boolean(key))


if __name__ == '__main__':
    unittest.main()
