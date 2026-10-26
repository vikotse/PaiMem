"""Numerical and pipeline contracts for the optional MemPursuit selector."""
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from paimem.mempursuit import MemPursuitConfig, select, solve_simplex_qp
from paimem.memory_store import MemoryMessage, SQLiteMemoryStore
from paimem.near import NearBank, NearConfig
from paimem.retrieval import retrieve_chunks


VECTORS = np.asarray([[.8, .6], [.8, .6], [.6, -.8], [0., 1.]], dtype=np.float32)


class CountingEmbedder:
    identity = "mempursuit-test"

    def __init__(self, matrix=VECTORS):
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.document_calls = []
        self.query_calls = []

    def encode_documents(self, texts):
        self.document_calls.append(list(texts))
        return np.asarray([self.matrix[next(i for i in range(len(self.matrix))
                                           if f"MEMORY_{i}" in text)] for text in texts])

    def encode_queries(self, texts):
        self.query_calls.append(list(texts))
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)


class MemPursuitMathTests(unittest.TestCase):
    def test_frozen_official_reference_selection_and_step_values(self):
        # Generated with the unmodified NumPy upstream at
        # FengQi-HITSZ/MemPursuit commit 5138dd0e6f6dbf843692526a44b696a4b9f4bcfb.
        # Frozen values keep this regression test offline and independent of our solver.
        query = [1., .5, -.25]
        matrix = [[1., 1., 0.], [1., 1., 0.], [1., -1., -.5],
                  [.2, 1., -1.], [-1., 0., 1.], [.5, -.2, .7]]
        result = select(query, matrix, 4)
        self.assertEqual(result.selected_indices.tolist(), [0, 2, 3, 1])
        np.testing.assert_allclose(result.weights, [0.7783203628320737, 0.21851785984923253,
                                                    0.003161777318693758, 0.], atol=1e-12)
        np.testing.assert_allclose([step.selection_score for step in result.steps],
                                   [0.9258200997725514, 0.363696483726654,
                                    0.14721333571491857, 0.14562136131435155], atol=1e-12)
        np.testing.assert_allclose([step.squared_error for step in result.steps],
                                   [0.14835980045489694, 0.05249193664253671,
                                    0.05248417434049735, 0.05248417434049757], atol=1e-12)
        self.assertEqual(result.stop_reason, "budget")

    def test_complementary_evidence_beats_duplicate_and_refits_both_weights(self):
        # The analytic simplex projection onto these two unit vectors is (.6, .4).
        matrix = np.asarray([[.8, .6], [.8, .6], [.6, -.8], [0., 1.]])
        result = select([1., 0.], matrix, 2, ridge=0.)
        self.assertEqual(result.selected_indices.tolist(), [0, 2])
        np.testing.assert_allclose(result.weights, [.6, .4], atol=1e-12)
        self.assertAlmostEqual(result.steps[-1].squared_error, .08, places=12)
        self.assertAlmostEqual(result.steps[1].selection_score, .6, places=12)
        self.assertEqual(result.stop_reason, "budget")

    def test_fixed_budget_retains_zero_weight_entries_after_exact_reconstruction(self):
        matrix = np.asarray([[1., 0.], [0., 1.], [-1., 0.]])
        result = select([1., 0.], matrix, 3, ridge=0.)
        self.assertEqual(result.selected_indices.tolist(), [0, 1, 2])
        np.testing.assert_array_equal(result.weights, [1., 0., 0.])
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.stop_reason, "budget")
        np.testing.assert_array_equal(select([1., 0.], matrix, 2, ridge=0.).selected_indices,
                                      result.selected_indices[:2])

    def test_normalization_does_not_mutate_inputs(self):
        query = np.asarray([3., 0.])
        matrix = VECTORS.astype(np.float64) * np.asarray([2., 4., 3., 5.])[:, None]
        before_query, before_matrix = query.copy(), matrix.copy()
        result = select(query, matrix, 2)
        self.assertEqual(result.selected_indices.tolist(), [0, 2])
        np.testing.assert_array_equal(query, before_query)
        np.testing.assert_array_equal(matrix, before_matrix)

    def test_simplex_solver_handles_singular_duplicates_and_boundary_solution(self):
        for gram, linear, ridge in [(np.ones((3, 3)), np.ones(3), 0.),
                                     (np.eye(3), np.asarray([4., 1., 0.]), 1e-10)]:
            with self.subTest(ridge=ridge):
                weights = solve_simplex_qp(gram, linear, ridge=ridge)
                self.assertGreaterEqual(weights.min(), 0.)
                self.assertAlmostEqual(weights.sum(), 1., places=12)
                gradient = (gram + ridge * np.eye(3)) @ weights - linear
                active = weights > 1e-8
                multiplier = gradient[active].mean()
                np.testing.assert_allclose(gradient[active], multiplier, atol=1e-8)
                self.assertTrue(np.all(gradient[~active] >= multiplier - 1e-8))

    def test_random_simplex_solutions_satisfy_kkt_conditions_and_warm_start(self):
        rng = np.random.default_rng(731)
        for count in (3, 8, 15):
            matrix = rng.normal(size=(count, 5))
            gram, linear = matrix @ matrix.T, matrix @ rng.normal(size=5)
            weights = solve_simplex_qp(gram, linear, ridge=1e-5)
            self.assertGreaterEqual(weights.min(), 0.)
            self.assertAlmostEqual(weights.sum(), 1., places=10)
            gradient = (gram + 1e-5 * np.eye(count)) @ weights - linear
            active = weights > 1e-8
            multiplier = gradient[active].mean()
            np.testing.assert_allclose(gradient[active], multiplier, atol=1e-7)
            self.assertTrue(np.all(gradient[~active] >= multiplier - 1e-7))
            warm = solve_simplex_qp(gram, linear, ridge=1e-5, warm_start=weights)
            np.testing.assert_allclose(warm, weights, atol=1e-7)

    def test_invalid_selector_inputs_fail_explicitly(self):
        cases = [([0., 0.], VECTORS, 2), ([1., 0.], [[0., 0.]], 1),
                 ([np.nan, 0.], VECTORS, 1), ([1., 0.], [[np.inf, 1.]], 1),
                 ([1., 0., 0.], VECTORS, 1), ([[1., 0.]], VECTORS, 1),
                 ([1., 0.], [], 1), ([1., 0.], VECTORS, 0),
                 ([1., 0.], VECTORS, 5)]
        for query, matrix, budget in cases:
            with self.subTest(query=query, budget=budget), self.assertRaises(ValueError):
                select(query, matrix, budget)
        for budget in (True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(TypeError):
                select([1., 0.], VECTORS, budget)
        for ridge in (-1., float("nan"), float("inf"), True, "0.1"):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                select([1., 0.], VECTORS, 2, ridge=ridge)

    def test_invalid_solver_inputs_fail_explicitly(self):
        for gram, linear in [(np.zeros((2, 3)), [1., 2.]), (np.eye(2), [1.]),
                             ([[np.nan]], [1.]), ([[1.]], [np.inf])]:
            with self.subTest(gram=gram), self.assertRaises(ValueError):
                solve_simplex_qp(gram, linear)

    def test_config_is_independent_disabled_by_default_and_validated(self):
        self.assertEqual(MemPursuitConfig().settings(), {
            "mempursuit_enabled": False, "mempursuit_candidates": 200,
            "mempursuit_ridge": 1e-10,
        })
        for kwargs in ({"enabled": "false"}, {"candidates": 0}, {"candidates": True},
                       {"candidates": 1.5}, {"ridge": -1.}, {"ridge": float("nan")},
                       {"ridge": float("inf")}, {"ridge": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MemPursuitConfig(**kwargs)


class MemPursuitRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.embedder = CountingEmbedder()
        self.rows = [dict(id=str(i), role="user", content=f"MEMORY_{i}", dimension=2,
                          embedding=vector.tobytes()) for i, vector in enumerate(VECTORS)]

    def retrieve(self, top_k=2, **kwargs):
        settings = dict(embedder=self.embedder, embedding_lock=threading.RLock(),
                        render_content=lambda row, names: row["content"], retrieval_mode="dense",
                        diagnostics=True, mempursuit_config=MemPursuitConfig(enabled=True))
        settings.update(kwargs)
        return retrieve_chunks(self.rows, {}, "question", top_k, **settings)

    def test_disabled_is_exactly_the_existing_order_and_scores_without_selector_calls(self):
        expected = self.retrieve(mempursuit_config=None)
        with patch("paimem.retrieval.select_memories", side_effect=AssertionError("disabled")):
            actual = self.retrieve(mempursuit_config=MemPursuitConfig())
        np.testing.assert_array_equal(actual.positions, expected.positions)
        np.testing.assert_array_equal(actual.scores, expected.scores)
        self.assertEqual(actual.diagnostics, expected.diagnostics)
        self.assertFalse(actual.diagnostics["mempursuit"]["active"])

    def test_selection_changes_order_preserving_scores_and_tail(self):
        result = self.retrieve()
        self.assertEqual(result.positions.tolist(), [0, 2, 1, 3])
        np.testing.assert_array_equal(result.scores, VECTORS[:, 0])
        detail = {row["id"]: row["channels"]["mempursuit"]
                  for row in result.diagnostics["candidates"]}
        self.assertEqual(detail["0"]["rank"], 1)
        self.assertEqual(detail["2"]["rank"], 2)
        self.assertEqual(detail["2"]["score"], .5)
        self.assertAlmostEqual(detail["0"]["coefficient"], .6, places=6)
        self.assertTrue(detail["1"]["backfill_candidate"])
        self.assertIsNone(detail["1"]["rank"])
        self.assertEqual(result.diagnostics["mempursuit"]["selected_count"], 2)
        self.assertEqual(result.diagnostics["mempursuit"]["backfill_count"], 2)

    def test_pool_cap_limits_optimization_and_preserves_eligible_outside_tail(self):
        result = self.retrieve(mempursuit_config=MemPursuitConfig(enabled=True, candidates=2))
        self.assertEqual(result.positions.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.diagnostics["mempursuit"]["pool_count"], 2)
        outside = next(c for c in result.diagnostics["candidates"] if c["id"] == "2")
        self.assertTrue(outside["channels"]["mempursuit"]["backfill_candidate"])
        self.assertIsNone(outside["channels"]["mempursuit"]["pool_rank"])
        bigger = self.retrieve(top_k=3, mempursuit_config=MemPursuitConfig(enabled=True, candidates=1))
        self.assertEqual(bigger.diagnostics["mempursuit"]["pool_count"], 3)
        self.assertEqual(bigger.diagnostics["mempursuit"]["selected_count"], 3)

    def test_threshold_rejections_never_reenter_selected_or_backfill_candidates(self):
        reranker = Mock(score=Mock(return_value=[.9, .8, .7, .1]))
        result = self.retrieve(reranker=reranker, rerank_score_threshold=.8)
        self.assertEqual(result.positions.tolist(), [0, 1])
        detail = {c["id"]: c for c in result.diagnostics["candidates"]}
        self.assertAlmostEqual(detail["2"]["channels"]["reranker"]["score"], .7, places=6)
        self.assertEqual(detail["2"]["channels"]["reranker"]["rank"], 3)
        self.assertFalse(detail["2"]["channels"]["mempursuit"]["backfill_candidate"])
        self.assertIsNone(detail["2"]["channels"]["mempursuit"]["rank"])
        np.testing.assert_allclose(result.scores, [.9, .8, .7, .1], atol=1e-7)
        with patch("paimem.retrieval.select_memories", side_effect=AssertionError("empty")):
            empty = self.retrieve(reranker=reranker, rerank_score_threshold=1.)
        self.assertEqual(empty.positions.tolist(), [])
        self.assertFalse(empty.diagnostics["mempursuit"]["active"])

    def test_diagnostics_do_not_change_results_or_add_model_calls(self):
        with_trace = self.retrieve()
        plain = self.retrieve(diagnostics=False)
        np.testing.assert_array_equal(with_trace.positions, plain.positions)
        np.testing.assert_array_equal(with_trace.scores, plain.scores)
        self.assertIsNone(plain.diagnostics)
        self.assertEqual(self.embedder.query_calls, [["question"], ["question"]])
        self.assertEqual(self.embedder.document_calls, [])

    def test_enhanced_options_reuse_first_vector_from_one_embedding_batch(self):
        result = self.retrieve(retrieval_mode="enhanced", lexical_weight=0., options=["a", "b"])
        self.assertEqual(self.embedder.query_calls, [["question", "question\nCandidate answer: a",
                                                     "question\nCandidate answer: b"]])
        self.assertEqual(result.diagnostics["mempursuit"]["query_view"], "question")
        self.assertEqual(result.positions[:2].tolist(), [0, 2])
        self.assertIn("option_dense_1", result.diagnostics["candidates"][0]["channels"])

    def test_near_changes_candidate_ranking_but_selector_uses_original_vectors(self):
        bank = NearBank(VECTORS, NearConfig(enabled=True, neighbors=2))
        with patch("paimem.retrieval.select_memories", wraps=select) as selector:
            result = self.retrieve(near_context=(bank, False))
        query, matrix, budget = selector.call_args.args
        np.testing.assert_array_equal(query, [1., 0.])
        self.assertEqual(budget, 2)
        self.assertEqual(sorted(map(tuple, matrix)), sorted(map(tuple, VECTORS)))
        self.assertTrue(result.diagnostics["near"]["active"])
        self.assertTrue(result.diagnostics["mempursuit"]["active"])
        np.testing.assert_array_equal(result.scores, bank.score(np.asarray([[1., 0.]]))[:, 0])

    def test_empty_scope_does_not_call_models_or_selector(self):
        self.rows = []
        with patch("paimem.retrieval.select_memories", side_effect=AssertionError("empty")):
            result = self.retrieve()
        self.assertEqual(result.positions.tolist(), [])
        self.assertEqual(self.embedder.query_calls, [])
        self.assertEqual(self.embedder.document_calls, [])
        self.assertFalse(result.diagnostics["mempursuit"]["active"])


class MemPursuitStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embedder = CountingEmbedder()

    def store(self, **kwargs):
        settings = dict(retrieval_mode="dense", mempursuit_config=MemPursuitConfig(enabled=True))
        settings.update(kwargs)
        store = SQLiteMemoryStore(Path(self.directory.name) / "memory.db", self.embedder, **settings)
        store.add("r", "u", "s", [MemoryMessage("user", f"MEMORY_{i}", 1704067200000 + i * 1000)
                                    for i in range(len(self.embedder.matrix))])
        return store

    def test_runtime_toggle_is_independent_and_needs_no_reingestion(self):
        store = self.store(mempursuit_config=MemPursuitConfig(candidates=3, ridge=1e-8))
        before = store.search("u", "question", 2)
        self.assertTrue(before[1].content.endswith("MEMORY_1"))
        document_calls = list(self.embedder.document_calls)
        initial_stats = store.stats()
        store.update_config(mempursuit_enabled=True)
        after = store.search("u", "question", 2)
        self.assertTrue(after[1].content.endswith("MEMORY_2"))
        self.assertFalse(store.get_config()["near_enabled"])
        self.assertFalse(store.get_config()["reranker_enabled"])
        self.assertEqual(store.get_config()["mempursuit_candidates"], 3)
        self.assertEqual(store.get_config()["mempursuit_ridge"], 1e-8)
        self.assertEqual(self.embedder.document_calls, document_calls)
        self.assertEqual(store.stats(), initial_stats)
        store.update_config(mempursuit_enabled=False)
        self.assertEqual(store.search("u", "question", 2), before)
        for invalid in ("true", 1, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                store.update_config(mempursuit_enabled=invalid)
        self.assertFalse(store.get_config()["mempursuit_enabled"])

    def test_add_performs_no_selection_or_additional_embeddings(self):
        with patch("paimem.retrieval.select_memories", side_effect=AssertionError("Add")):
            store = self.store()
        self.assertEqual(len(self.embedder.document_calls), 1)
        self.assertEqual(len(self.embedder.document_calls[0]), 4)
        self.assertEqual(self.embedder.query_calls, [])
        self.assertEqual(store.stats()["memories"], 4)

    def test_packing_backfills_from_eligible_tail_outside_optimization_pool(self):
        self.embedder = CountingEmbedder(np.asarray([[1., 0.]] * 4))
        store = self.store(mempursuit_config=MemPursuitConfig(enabled=True, candidates=2),
                           result_window=1, result_window_seed_k=1, pack_span=2)
        hits, detail = store.search("u", "question", 2, diagnostics=True)
        self.assertEqual(len(hits), 2)
        self.assertIn("MEMORY_0", hits[0].content)
        self.assertIn("MEMORY_1", hits[0].content)
        self.assertTrue(hits[1].content.endswith("MEMORY_2"))
        tail = next(c for c in detail["candidates"] if c["content"].endswith("MEMORY_2"))
        self.assertEqual(tail["returned_rank"], 2)
        self.assertTrue(tail["channels"]["mempursuit"]["backfill_candidate"])
        self.assertIsNone(tail["channels"]["mempursuit"]["pool_rank"])

    def test_conflict_queues_share_one_switch_snapshot_during_runtime_update(self):
        from paimem.conflict_strategy import JudgmentQueries
        strategy = Mock(classify=Mock(return_value=JudgmentQueries("positive", "negative", "scope")),
                        assess=Mock(return_value=None))
        store = self.store(conflict_strategy=strategy, near_config=NearConfig(enabled=True, neighbors=2))
        original_encode = self.embedder.encode_queries

        def update_during_first_query(texts):
            store.update_config(mempursuit_enabled=False)
            return original_encode(texts)

        with patch.object(self.embedder, "encode_queries", side_effect=update_during_first_query):
            hits, detail = store.search("u", "question", 5, diagnostics=True)
        self.assertTrue(hits)
        self.assertFalse(store.get_config()["mempursuit_enabled"])
        self.assertTrue(detail["mempursuit"]["active"])
        for queue in detail["conflict_strategy"]["queues"].values():
            self.assertTrue(queue["pipeline"]["mempursuit"]["active"])
            self.assertTrue(queue["pipeline"]["near"]["active"])
        self.assertFalse(store.search("u", "question", 5, diagnostics=True)[1]["mempursuit"]["active"])


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is required")
class MemPursuitConfigurationTests(unittest.TestCase):
    def setUp(self):
        from paimem import api
        self.api = api
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.enterContext(patch.dict(os.environ, {
            "MEMORY_DB_PATH": str(Path(self.directory.name) / "memory.db"),
            "MEMORY_RERANKER_MODEL": "", "MEMORY_API_KEY": "test",
            "MEMORY_RESULT_WINDOW": "0", "MEMORY_RETRIEVAL_MODE": "dense",
        }, clear=True))
        self.enterContext(patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True))
        self.embedder = self.enterContext(patch.object(api, "build_embedder", return_value=CountingEmbedder()))
        self.reranker = self.enterContext(patch.object(api, "build_reranker"))
        api.get_store.cache_clear()
        self.addCleanup(api.get_store.cache_clear)

    def test_startup_environment_and_overrides(self):
        os.environ.update(MEMORY_MEMPURSUIT_ENABLED="true", MEMORY_MEMPURSUIT_CANDIDATES="17",
                          MEMORY_MEMPURSUIT_RIDGE="1e-7")
        store = self.api.get_store()
        self.assertEqual(store.mempursuit_config, MemPursuitConfig(enabled=True, candidates=17, ridge=1e-7))
        self.api.configure_store(mempursuit_enabled=False, mempursuit_candidates=3, mempursuit_ridge=0.)
        self.api.get_store.cache_clear()
        self.assertEqual(self.api.get_store().mempursuit_config, MemPursuitConfig(candidates=3, ridge=0.))

    def test_invalid_environment_fails_before_model_loading(self):
        for key, values in {"MEMORY_MEMPURSUIT_ENABLED": ["sometimes", "2"],
                            "MEMORY_MEMPURSUIT_CANDIDATES": ["0", "-1", "1.5"],
                            "MEMORY_MEMPURSUIT_RIDGE": ["nan", "inf", "-1"]}.items():
            for value in values:
                with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}):
                    self.api.get_store.cache_clear()
                    with self.assertRaises(ValueError):
                        self.api.get_store()
        self.embedder.assert_not_called()
        self.reranker.assert_not_called()

    def test_http_and_cli_switch_preserve_public_search_schema(self):
        from fastapi.testclient import TestClient
        from paimem.cli import parse_setting
        self.assertEqual(parse_setting("mempursuit_enabled=true"), ("mempursuit_enabled", True))
        self.assertEqual(parse_setting("mempursuit_enabled=false"), ("mempursuit_enabled", False))
        with self.assertRaises(ValueError):
            parse_setting("mempursuit_enabled=1")
        with TestClient(self.api.app, headers={"X-Api-Key": "test"}) as client:
            self.assertFalse(client.get("/config").json()["mempursuit_enabled"])
            response = client.post("/config", json={"mempursuit_enabled": True})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(client.get("/config").json()["mempursuit_enabled"])
            self.assertEqual(client.post("/config", json={"mempursuit_enabled": "true"}).status_code, 422)
            for key, value in (("mempursuit_candidates", 5), ("mempursuit_ridge", .01)):
                self.assertEqual(client.post("/config", json={key: value}).status_code, 422)
            response = client.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s",
                                                 "messages": [{"role": "user", "content": f"MEMORY_{i}"}
                                                              for i in range(4)]})
            self.assertEqual(response.status_code, 200, response.text)
            response = client.post("/search", json={"user_id": "u", "query": "question", "top_k": 2})
            self.assertEqual(response.status_code, 200, response.text)
            hits = response.json()["data"]
            self.assertEqual(len(hits), 2)
            self.assertEqual(set(hits[0]), {"id", "content", "score", "created_at"})
            self.assertTrue(hits[1]["content"].endswith("MEMORY_2"))


if __name__ == "__main__":
    unittest.main()
