import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from paimem.memory_store import MemoryMessage, SQLiteMemoryStore


class StaticEmbedder:
    def encode_documents(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    encode_queries = encode_documents


class V2RerankThresholdTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store_count = 0

    def store(self, **settings):
        self.store_count += 1
        config = dict(reranker=Mock(score=Mock(return_value=[0.25, 0.5, 0.75])))
        config.update(settings)
        store = SQLiteMemoryStore(
            Path(self.directory.name) / f"{self.store_count}.db", StaticEmbedder(), **config
        )
        store.add("r", "u", "s", [MemoryMessage("user", text, 1704067200000 + i * 1000)
                                   for i, text in enumerate(["LOW", "BOUNDARY", "HIGH"])])
        return store

    def test_unset_preserves_all_results_and_order(self):
        store = self.store()
        hits = store.search("u", "query", 3)
        self.assertEqual([hit.score for hit in hits], [0.75, 0.5, 0.25])
        self.assertIsNone(store.get_config()["rerank_score_threshold"])

    def test_boundary_is_inclusive_without_backfilling_filtered_candidates(self):
        store = self.store(rerank_score_threshold=0.5)
        hits, detail = store.search("u", "query", 3, diagnostics=True)
        self.assertEqual([hit.score for hit in hits], [0.75, 0.5])
        self.assertEqual(hits, store.search("u", "query", 3))
        self.assertEqual(detail["rerank_candidate_count"], 3)
        self.assertEqual(detail["returned_count"], 2)
        self.assertEqual(detail["rerank_score_threshold"], 0.5)
        low = next(c for c in detail["candidates"] if c["content"].endswith("LOW"))
        self.assertEqual(low["channels"]["reranker"], {"score": 0.25, "rank": 3})
        self.assertIsNone(low["returned_rank"])

    def test_all_filtered_returns_empty_even_with_neighbors_and_packing(self):
        for span in (1, 2):
            with self.subTest(pack_span=span):
                store = self.store(rerank_score_threshold=1.0, result_window=1, pack_span=span)
                hits, detail = store.search("u", "query", 3, diagnostics=True)
                self.assertEqual(hits, [])
                self.assertEqual(detail["returned_count"], 0)
                self.assertEqual(detail["rerank_candidate_count"], 3)
                self.assertTrue(all(c["returned_rank"] is None for c in detail["candidates"]))

    def test_low_scoring_neighbors_survive_in_both_expansion_modes(self):
        for span in (1, 2):
            with self.subTest(pack_span=span):
                store = self.store(rerank_score_threshold=0.75, result_window=1, pack_span=span)
                hits = store.search("u", "query", 3)
                content = "\n".join(hit.content for hit in hits)
                self.assertIn("HIGH", content)
                self.assertIn("BOUNDARY", content)
                self.assertNotIn("LOW", content)
                self.assertEqual(len(hits), 2 if span == 1 else 1)
                self.assertEqual(hits[0].score, 0.75)

    def test_unscored_neighbor_survives_and_remains_unscored_in_diagnostics(self):
        store = self.store(rerank_score_threshold=0.5, rerank_candidates=1,
                           result_window=1, pack_span=2,
                           reranker=Mock(score=Mock(return_value=[0.75])))
        hits, detail = store.search("u", "query", 1, diagnostics=True)
        self.assertEqual(len(hits), 1)
        self.assertIn("BOUNDARY", hits[0].content)
        neighbor = next(c for c in detail["candidates"] if c["content"].endswith("BOUNDARY"))
        self.assertEqual(neighbor["channels"]["reranker"], {"score": None, "rank": None})
        self.assertEqual(neighbor["returned_rank"], 1)

    def test_zero_is_an_active_threshold_and_negative_scores_are_supported(self):
        for threshold, expected in [(None, [0.5, 0.0, -0.5]), (0.0, [0.5, 0.0]),
                                    (-0.5, [0.5, 0.0, -0.5])]:
            with self.subTest(threshold=threshold):
                store = self.store(rerank_score_threshold=threshold,
                                   reranker=Mock(score=Mock(return_value=[-0.5, 0.0, 0.5])))
                self.assertEqual([hit.score for hit in store.search("u", "query", 3)], expected)

    def test_threshold_does_not_filter_dense_or_fusion_scores(self):
        for mode in ("dense", "hybrid"):
            with self.subTest(mode=mode):
                store = self.store(reranker=None, rerank_score_threshold=2.0, retrieval_mode=mode)
                self.assertEqual(len(store.search("u", "query", 3)), 3)

    def test_disabling_reranker_bypasses_threshold(self):
        store = self.store(rerank_score_threshold=1.0)
        self.assertEqual(store.search("u", "query", 3), [])
        store.update_config(reranker_enabled=False)
        self.assertEqual(len(store.search("u", "query", 3)), 3)
        store.update_config(reranker_enabled=True)
        self.assertEqual(store.search("u", "query", 3), [])

    def test_invalid_thresholds_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), "0.5", True):
            with self.subTest(threshold=value), self.assertRaisesRegex(ValueError, "rerank_score_threshold"):
                self.store(rerank_score_threshold=value)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is required")
class V2RerankThresholdEnvironmentTests(unittest.TestCase):
    def setUp(self):
        from paimem import api

        self.api = api
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.enterContext(patch.dict(os.environ, {
            "MEMORY_DB_PATH": str(Path(self.directory.name) / "api.db"),
            "MEMORY_RERANKER_MODEL": "test-reranker",
            "MEMORY_RESULT_WINDOW": "0",
            "ALLOW_UNAUTHENTICATED": "true",
        }, clear=True))
        self.enterContext(patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True))
        self.enterContext(patch.object(api, "build_embedder", return_value=StaticEmbedder()))
        self.enterContext(patch.object(api, "build_reranker", return_value=Mock(
            score=Mock(return_value=[0.25, 0.5, 0.75]))))
        api.get_store.cache_clear()
        self.addCleanup(api.get_store.cache_clear)

    def test_environment_controls_search_end_to_end(self):
        from fastapi.testclient import TestClient

        for raw, expected, count in [(None, None, 3), ("", None, 3), ("  ", None, 3),
                                     (" 0.5 ", 0.5, 2), ("0", 0.0, 3), ("1e-2", 0.01, 3),
                                     ("1", 1.0, 0)]:
            with self.subTest(raw=raw):
                os.environ.pop("MEMORY_RERANK_SCORE_THRESHOLD", None)
                if raw is not None:
                    os.environ["MEMORY_RERANK_SCORE_THRESHOLD"] = raw
                self.api.get_store.cache_clear()
                with TestClient(self.api.app) as client:
                    self.assertEqual(self.api.get_store().get_config()["rerank_score_threshold"], expected)
                    response = client.post("/add", json={
                        "request_id": "r", "user_id": "u", "session_id": "s",
                        "messages": [{"role": "user", "content": text} for text in ("LOW", "BOUNDARY", "HIGH")],
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    response = client.post("/search", json={"query": "query", "user_id": "u", "top_k": 3})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(len(response.json()["data"]), count)

    def test_invalid_environment_fails_before_models_load(self):
        for value in ("not-a-number", "nan", "inf", "-inf", "1e999"):
            with self.subTest(value=value):
                os.environ["MEMORY_RERANK_SCORE_THRESHOLD"] = value
                self.api.get_store.cache_clear()
                with self.assertRaisesRegex(ValueError, "MEMORY_RERANK_SCORE_THRESHOLD"):
                    self.api.get_store()
        self.api.build_embedder.assert_not_called()
        self.api.build_reranker.assert_not_called()

    def test_explicit_startup_override_takes_precedence_over_environment(self):
        os.environ["MEMORY_RERANK_SCORE_THRESHOLD"] = "invalid"
        for value in (None, 0.5):
            with self.subTest(value=value):
                self.api.configure_store(rerank_score_threshold=value)
                self.api.get_store.cache_clear()
                self.assertEqual(self.api.get_store().rerank_score_threshold, value)
