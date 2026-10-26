import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from paimem.near import NearBank, NearCache, NearConfig, neighborhood_references, score_scale, unit
from paimem.memory_store import MemoryMessage, SQLiteMemoryStore, bm25_scores


class NearMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.matrix = unit(rng.normal(size=(31, 9)) + np.array([4.] + [0.] * 8))
        self.query = unit(rng.normal(size=(3, 9)))

    def test_scale_matches_explicit_pairwise_population_variance(self):
        for matrix in (self.matrix, self.matrix * np.linspace(.1, 1, 31)[:, None]):
            expected = max(1e-6, np.sqrt(np.var(matrix @ matrix.T, axis=1).mean()))
            self.assertAlmostEqual(score_scale(matrix), expected, places=12)

    def test_neighbors_exclude_self_and_projection_is_orthogonal(self):
        reference = neighborhood_references(self.matrix, 5)
        order = np.argsort(-(self.matrix @ self.matrix.T), axis=1)[:, 1:6]
        np.testing.assert_allclose(reference, unit(self.matrix[order].mean(axis=1)), atol=1e-12)
        residual = self.matrix - np.sum(self.matrix * reference, axis=1, keepdims=True) * reference
        np.testing.assert_allclose(np.sum(residual * reference, axis=1), 0, atol=1e-12)

    def test_fusion_equals_explicit_concatenated_inner_product(self):
        for normalize in (False, True):
            config = NearConfig(enabled=True, layers=2, normalize=normalize, weight=.5, neighbors=5)
            bank = NearBank(self.matrix, config)
            states = self.query.copy()
            query_views = [states]
            for anchors, directions in bank.tables:
                outputs = []
                for query in states:
                    selected = np.argsort(-(anchors @ query), kind="stable")[:config.routing]
                    logits = config.beta * (anchors[selected] @ query)
                    weights = np.exp(logits - logits.max())
                    weights /= weights.sum()
                    ref = unit(weights @ directions[selected])
                    outputs.append(query - (query @ ref) * ref)
                residuals = np.asarray(outputs)
                states = unit(residuals)
                query_views.append(states if normalize else residuals)
            factor = np.sqrt(config.weight / config.layers)
            keys = np.concatenate([bank.views[0], *[factor * v for v in bank.views[1:]]], axis=1)
            queries = np.concatenate([query_views[0] / bank.scales[0],
                                      *[factor * v / scale for v, scale in zip(query_views[1:], bank.scales[1:])]], axis=1)
            np.testing.assert_allclose(bank.score(self.query), keys @ queries.T, atol=1e-10)

    def test_degenerate_banks_are_finite_and_deterministic(self):
        for matrix in (self.matrix[:0], self.matrix[:1], np.ones((10, 9)), np.eye(9)):
            config = NearConfig(enabled=True, layers=3, normalize=True)
            bank = NearBank(matrix, config)
            self.assertTrue(np.isfinite(bank.score(self.query)).all())
            np.testing.assert_array_equal(bank.score(self.query), NearBank(matrix, config).score(self.query))
        with self.assertRaises(ValueError):
            NearBank(np.zeros((3, 9)), NearConfig(enabled=True))
        with self.assertRaises(ValueError):
            NearBank(self.matrix, NearConfig()).score(np.zeros((1, 9)))

    def test_cache_snapshot_user_and_capacity(self):
        cache = NearCache(NearConfig(enabled=True, cache_max_users=1))
        bank, hit = cache.get("a", list(range(31)), self.matrix)
        self.assertFalse(hit)
        self.assertIs(cache.get("a", list(range(31)), self.matrix)[0], bank)
        self.assertFalse(cache.get("b", list(range(31)), self.matrix)[1])
        self.assertFalse(cache.get("a", list(range(31)), self.matrix)[1])
        self.assertFalse(cache.get("a", list(range(31)), self.matrix[::-1])[1])
        cache.invalidate("a")
        self.assertFalse(cache.get("a", list(range(31)), self.matrix)[1])
        tiny = NearCache(NearConfig(cache_max_mb=.00001))
        tiny.get("a", list(range(31)), self.matrix)
        self.assertFalse(tiny.get("a", list(range(31)), self.matrix)[1])

    def test_invalid_config(self):
        for kwargs in ({"layers": 0}, {"layers": True}, {"weight": -1}, {"weight": float("nan")},
                       {"normalize": "false"}, {"cache_max_mb": 0}, {"routing": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NearConfig(**kwargs)


class StaticEmbedder:
    identity = "near-test"

    def __init__(self):
        rng = np.random.default_rng(14)
        self.matrix = unit(rng.normal(size=(6, 9)) + np.array([4.] + [0.] * 8)).astype(np.float32)
        self.query = unit(rng.normal(size=(1, 9))).astype(np.float32)

    def encode_documents(self, texts):
        return np.asarray([self.matrix[int(text.rsplit(" ", 1)[1]) % 6] for text in texts])

    def encode_queries(self, texts):
        return np.repeat(self.query, len(texts), axis=0)


class NearIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embedder = StaticEmbedder()
        self.path = Path(self.directory.name) / "memory.db"
        self.config = NearConfig(enabled=True, neighbors=2)

    def store(self, config=None, **settings):
        store = SQLiteMemoryStore(self.path, self.embedder, retrieval_mode="hybrid",
                                  near_config=config, **settings)
        store.add("r", "u", "s", [MemoryMessage("user", f"memory {i}", 1704067200000)
                                   for i in range(6)])
        return store

    def test_disabled_and_zero_weight_preserve_full_results(self):
        expected = self.store().search("u", "memory 1", 4)
        for config in (NearConfig(), NearConfig(enabled=True, weight=0)):
            store = self.store(config)
            with patch.object(store._near_cache, "get", side_effect=AssertionError("must bypass")):
                self.assertEqual(store.search("u", "memory 1", 4), expected)

    def test_hybrid_fuses_near_with_bm25_and_preserves_raw_diagnostics(self):
        store = self.store(self.config)
        hits, detail = store.search("u", "memory 1", 6, diagnostics=True)
        near_scores = NearBank(self.embedder.matrix, self.config).score(self.embedder.query)[:, 0]
        raw = self.embedder.matrix @ self.embedder.query[0]
        lexical = bm25_scores([f"memory {i}" for i in range(6)], "memory 1")
        fused = np.zeros(6)
        for order, weight in ((np.argsort(-near_scores), 1), (np.argsort(-lexical), store.lexical_weight)):
            for rank, index in enumerate(order):
                if weight == 1 or lexical[index] > 0:
                    fused[index] += weight / (store.rrf_k + rank + 1)
        expected = np.lexsort((-near_scores, -fused))
        self.assertEqual([int(h.content.rsplit(" ", 1)[1]) for h in hits], expected.tolist())
        for candidate in detail["candidates"]:
            i = int(candidate["content"].rsplit(" ", 1)[1])
            self.assertAlmostEqual(candidate["channels"]["dense"]["score"], raw[i], places=6)
            self.assertAlmostEqual(candidate["channels"]["near"]["score"], near_scores[i], places=6)
        self.assertFalse(detail["near"]["cache_hit"])
        self.assertTrue(store.search("u", "memory 1", 6, diagnostics=True)[1]["near"]["cache_hit"])

    def test_add_clear_and_other_users_do_not_reuse_stale_banks(self):
        store = self.store(self.config)
        store.search("u", "memory", 6)
        store.add("r2", "u", "s", [MemoryMessage("user", "memory 6", 1704067200000)])
        hits, detail = store.search("u", "memory", 100, diagnostics=True)
        self.assertEqual(len(hits), 7)
        self.assertFalse(detail["near"]["cache_hit"])
        store.add("r3", "v", "s", [MemoryMessage("user", "memory 0", 1704067200000)])
        self.assertEqual(len(store.search("v", "memory", 100)), 1)
        self.assertEqual(store.search("missing", "memory", 100), [])
        store.clear()
        self.assertFalse(store._near_cache._entries)
        self.assertEqual(store.search("u", "memory", 100), [])

    def test_reranker_window_options_and_conflict_routes_remain_compatible(self):
        from paimem.conflict_strategy import JudgmentQueries
        strategy = Mock(classify=Mock(return_value=JudgmentQueries("positive", "negative", "scope")),
                        assess=Mock(return_value=None))
        reranker = Mock(score=Mock(side_effect=lambda q, docs: [float(i) for i in range(len(docs))]))
        store = self.store(self.config, reranker=reranker, result_window=1, pack_span=2,
                           conflict_strategy=strategy)
        hits, detail = store.search("u", "memory", 10, diagnostics=True)
        self.assertTrue(hits)
        self.assertEqual(reranker.score.call_count, 4)
        self.assertTrue(all(q["pipeline"]["near"]["active"] for q in detail["conflict_strategy"]["queues"].values()))
        strategy.reset_mock()
        store.update_config(retrieval_mode="enhanced")
        _, detail = store.search("u", "memory", 4, options=["a", "b"], diagnostics=True)
        strategy.classify.assert_not_called()
        self.assertIn("option_dense_1", detail["candidates"][0]["channels"])
        with self.assertRaises(ValueError):
            store.update_config(near_enabled=False)

    def test_startup_env_overrides_and_public_response_schema(self):
        from fastapi.testclient import TestClient
        from paimem import api
        with patch.dict(os.environ, {"MEMORY_NEAR_ENABLED": "true", "MEMORY_NEAR_LAYERS": "2"}), \
             patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True):
            self.assertEqual(api._build_near_config().layers, 2)
            api.configure_store(near_enabled=False)
            self.assertFalse(api._build_near_config().enabled)
            with patch.dict(os.environ, {"MEMORY_NEAR_LAYERS": "1.5"}), self.assertRaises(ValueError):
                api._build_near_config()
        store = self.store(self.config)
        api.app.dependency_overrides[api.get_store] = lambda: store
        self.addCleanup(api.app.dependency_overrides.clear)
        with patch.dict(os.environ, {"SKIP_MODEL_STARTUP": "true", "MEMORY_API_KEY": "test"}):
            with TestClient(api.app) as client:
                response = client.post("/search", headers={"X-Api-Key": "test"},
                                       json={"user_id": "u", "query": "memory", "top_k": 4})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()["data"][0]), {"id", "content", "score", "created_at"})
                config = client.get("/config", headers={"X-Api-Key": "test"}).json()
                self.assertTrue(config["near_enabled"])

    def test_get_store_wires_near_startup_settings(self):
        from paimem import api
        api.get_store.cache_clear()
        self.addCleanup(api.get_store.cache_clear)
        with patch.dict(api.STORE_INIT_OVERRIDES, {
            "database": str(self.path), "embedding_model": "test", "near_enabled": True,
            "near_layers": 2, "near_weight": .5, "near_normalize": True,
        }, clear=True), patch.object(api, "build_embedder", return_value=self.embedder), \
             patch.object(api, "_build_conflict_strategy", return_value=(None, .2, .2, .2)), \
             patch.object(api, "_build_role_name_extractor", return_value=None), \
             patch.dict(os.environ, {"MEMORY_RERANKER_MODEL": ""}):
            store = api.get_store()
            self.assertEqual(store.near_config, NearConfig(enabled=True, layers=2, weight=.5, normalize=True))


class NearExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/run_near_ablation.py"
        spec = importlib.util.spec_from_file_location("near_ablation_tests", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_missing_labels_remain_null_and_completeness_requires_all_units(self):
        from paimem.memory_store import MemoryHit
        hits = [MemoryHit("a", "First fact", 1., "")]
        empty = self.runner.evidence_metrics(hits, [])
        self.assertIsNone(empty["recall_at_100"])
        labeled = self.runner.evidence_metrics(hits, ["First fact", "second fact"])
        self.assertEqual(labeled["recall_at_100"], .5)
        self.assertEqual(labeled["complete_at_100"], 0)

    def test_sampling_is_order_independent_and_preserves_entire_history(self):
        records = [{"id": str(i), "messages": ["full", "history"],
                    "questions": [{"id": str(j), "answer": "label"} for j in range(5)]} for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            self.runner.adapters.write_rows(path, records)
            first = self.runner.select_records(path, 3, 2)
            self.runner.adapters.write_rows(path, list(reversed(records)))
            self.assertEqual(first, self.runner.select_records(path, 3, 2))
            self.assertTrue(all(r["messages"] == ["full", "history"] for r in first))

    def test_validation_rejects_changed_backbone_and_e2e_requires_validation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            selection = folder / "selection.json"
            selection.write_text(json.dumps({"embedding": "different-model", "base_settings": self.runner.BASE_SETTINGS}))
            args = SimpleNamespace(output=folder, phase="validation", selection=selection, embedding_model="model")
            with self.assertRaisesRegex(ValueError, "frozen on dev"):
                self.runner.run_retrieval(args)
            with self.assertRaisesRegex(ValueError, "validation manifest"):
                self.runner.run_e2e(args)


if __name__ == "__main__":
    unittest.main()
