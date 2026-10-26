import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from paimem.conflict_strategy import JudgmentQueries, LLMConflictStrategy, validate_ratios
from paimem.memory_store import MemoryHit, MemoryMessage, SQLiteMemoryStore


QUESTION = "Have I ever implemented two-factor authentication for platform access?"
POSITIVE = "I implemented two-factor authentication for platform access."
NEGATIVE = "I have never implemented two-factor authentication for any platform."
SCOPE = "Corrections and scope of my platform two-factor authentication implementation."
QUERIES = JudgmentQueries(POSITIVE, NEGATIVE, SCOPE)


def response(body, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=json.dumps(body)))])


def contradiction(positive_id="p", negative_id="n", **overrides):
    return {"has_conflict": True,
            "positive_evidence": {"id": positive_id, "quote": POSITIVE},
            "negative_evidence": {"id": negative_id, "quote": NEGATIVE},
            "resolution": "The implementation statement is more likely correct, but remains uncertain.",
            "reason": "The affirmative record describes a completed action; neither statement explicitly corrects the other.", **overrides}


class LLMStrategyTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.strategy = LLMConflictStrategy(self.client, "test-model")
        self.positive = [MemoryHit("p", POSITIVE, 1, "2024-01-01")]
        self.negative = [MemoryHit("n", NEGATIVE, 1, "2024-02-01")]

    def test_classification_and_queries_are_semantic_llm_output(self):
        self.client.chat.completions.create.return_value = response({
            "is_judgment": True, "positive_query": POSITIVE, "negative_query": NEGATIVE, "scope_query": SCOPE})
        self.assertEqual(self.strategy.classify(QUESTION), QUERIES)
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(json.loads(payload["messages"][1]["content"]), {"question": QUESTION})
        self.client.chat.completions.create.return_value = response({"is_judgment": False})
        self.assertIsNone(self.strategy.classify("Describe my authentication work"))

    def test_beam_example_discloses_both_claims_and_gives_inference(self):
        self.client.chat.completions.create.return_value = response(contradiction())
        note = self.strategy.assess(QUESTION, self.positive, self.negative, [MemoryHit("s", SCOPE, 1, "2024-03-01")])
        for text in ("There is contradictory information", POSITIVE, NEGATIVE,
                     "Inferred correct description", "more likely correct, but remains uncertain", "Basis for inference"):
            self.assertIn(text, note)
        self.assertNotIn("Which statement is correct?", note)
        payload = json.loads(self.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["positive"][0]["created_at"], "2024-01-01")
        self.assertEqual(payload["scope"][0]["content"], SCOPE)

    def test_resolution_and_no_conflict(self):
        self.client.chat.completions.create.return_value = response(contradiction(
            resolution="The implementation occurred.", reason="An explicit correction confirms it."))
        self.assertIn("Inferred correct description", self.strategy.assess(QUESTION, self.positive, self.negative, [MemoryHit("s", SCOPE, 1, "2024-03-01")]))
        self.client.chat.completions.create.return_value = response({"has_conflict": False})
        self.assertIsNone(self.strategy.assess(QUESTION, self.positive, self.negative, [MemoryHit("s", SCOPE, 1, "2024-03-01")]))

    def test_missing_queue_skips_assessment(self):
        self.assertIsNone(self.strategy.assess(QUESTION, self.positive, [], []))
        self.client.chat.completions.create.assert_not_called()

    def test_rejects_unselected_ids_fabricated_quotes_and_malformed_output(self):
        cases = [[], {}, {"has_conflict": "true"}, contradiction("unknown"),
                 contradiction(positive_evidence={"id": "p", "quote": "invented evidence"}),
                 contradiction(reason=""), contradiction(resolution=123), contradiction(resolution=None),
                 contradiction(resolution="")]
        for body in cases:
            with self.subTest(body=body), self.assertRaises((ValueError, TypeError)):
                self.client.chat.completions.create.return_value = response(body)
                self.strategy.assess(QUESTION, self.positive, self.negative, [MemoryHit("s", SCOPE, 1, "2024-03-01")])

    def test_invalid_classification_and_truncation(self):
        for body in ({}, {"is_judgment": 1}, {"is_judgment": True},
                     {"is_judgment": True, "positive_query": "same", "negative_query": "same", "scope_query": "other"},
                     {"is_judgment": True, "positive_query": POSITIVE, "negative_query": NEGATIVE},
                     {"is_judgment": True, "positive_query": POSITIVE, "negative_query": NEGATIVE, "scope_query": POSITIVE}):
            self.client.chat.completions.create.return_value = response(body)
            with self.assertRaises(ValueError):
                self.strategy.classify(QUESTION)
        self.client.chat.completions.create.return_value = response({"is_judgment": False}, "length")
        with self.assertRaises(ValueError):
            self.strategy.classify(QUESTION)


class RecordingEmbedder:
    identity = "conflict-tests"

    def __init__(self):
        self.queries = []

    def encode_documents(self, texts):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        self.queries.extend(texts)
        return self.encode_documents(texts)


class SearchStrategyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embedder = RecordingEmbedder()
        self.strategy = Mock(classify=Mock(return_value=QUERIES), assess=Mock(return_value=None))
        self.reranker = Mock(score=Mock(side_effect=self.score))
        self.store = SQLiteMemoryStore(
            Path(self.directory.name) / "test.db", self.embedder, retrieval_mode="hybrid",
            reranker=self.reranker, conflict_strategy=self.strategy,
        )
        texts = [f"ordinary memory {i}" for i in range(100)] + [POSITIVE, NEGATIVE, SCOPE]
        texts += [f"additional memory {i}" for i in range(17)]
        self.store.add("r", "u", "s", [MemoryMessage("user", text) for text in texts])

    @staticmethod
    def score(query, documents):
        # The semantic queue omits both contradictory memories at top_k=100;
        # each rewritten query independently promotes its relevant evidence.
        scores = []
        for document in documents:
            if query in (POSITIVE, NEGATIVE, SCOPE) and query in document:
                scores.append(1000.)
            elif "ordinary memory " in document:
                scores.append(100. - int(document.rsplit(" ", 1)[1]))
            else:
                scores.append(-10.)
        return scores

    def test_three_full_pipelines_and_default_20_20_20_order(self):
        hits, detail = self.store.search("u", QUESTION, 100, diagnostics=True)
        self.assertEqual(len(hits), 60)
        self.assertEqual([h.recall_queue for h in hits], ["positive"] * 20 + ["negative"] * 20 + ["scope"] * 20)
        self.assertEqual([r["recall_queue"] for r in detail["final_results"]], [h.recall_queue for h in hits])
        self.assertEqual(self.embedder.queries, [QUESTION, POSITIVE, NEGATIVE, SCOPE])
        self.assertEqual([c.args[0] for c in self.reranker.score.call_args_list], [QUESTION, POSITIVE, NEGATIVE, SCOPE])
        for offset, prefix, text in ((0, "Positive-related", POSITIVE), (20, "Negative-related", NEGATIVE),
                                     (40, "Scope/correction", SCOPE)):
            self.assertTrue(all(h.content.startswith(f"[{prefix} evidence]") for h in hits[offset:offset+20]))
            self.assertIn(text, hits[offset].content)
        question, positive, negative, scope = self.strategy.assess.call_args.args
        self.assertEqual(question, QUESTION)
        self.assertEqual([h.id for h in positive + negative + scope], [h.id for h in hits])
        self.assertEqual(detail["returned_count"], 60)
        for queue in detail["conflict_strategy"]["queues"].values():
            self.assertEqual(queue["pipeline"]["fusion_method"], "rrf")
            self.assertEqual(queue["pipeline"]["rerank_candidate_count"], 120)
        self.embedder.queries.clear()
        self.assertEqual(self.store.search("u", QUESTION, 100), hits)
        self.assertEqual(self.embedder.queries, [POSITIVE, NEGATIVE, SCOPE])

    def test_conflict_explanation_leads_results_and_preserves_three_budgets(self):
        for span in (1, 2):
            with self.subTest(pack_span=span):
                self.store.pack_span = span
                self.store.result_window = 1
                self.strategy.assess.return_value = None
                original, trace = self.store.search("u", QUESTION, 100, diagnostics=True)
                note = "[Conflicting evidence] Inferred correct description: " + POSITIVE
                self.strategy.assess.return_value = note
                hits, detail = self.store.search("u", QUESTION, 100, diagnostics=True)
                self.assertEqual(len(hits), 61)
                self.assertEqual(hits[1:], original)
                self.assertEqual(hits[0].content, note)
                self.assertTrue(hits[0].id.startswith("conflict:"))
                self.assertEqual((hits[0].recall_queue, hits[0].score, hits[0].created_at),
                                 ("conflict_summary", 0.0, ""))
                self.assertEqual(detail["returned_count"], 61)
                self.assertEqual(detail["final_results"][0]["rank"], 1)
                self.assertEqual(detail["final_results"][0]["recall_queue"], "conflict_summary")
                self.assertEqual(detail["conflict_strategy"]["explanation"], note)
                self.assertEqual(detail["conflict_strategy"]["explanation_placement"], "first_result")
                self.assertTrue(detail["conflict_strategy"]["has_conflict"])
                before = {c['id']: c for c in trace['candidates']}
                for candidate in detail['candidates']:
                    old = before[candidate['id']]
                    self.assertEqual(candidate['returned_ranks'], [rank + 1 for rank in old['returned_ranks']])
                    self.assertEqual(candidate['returned_rank'], old['returned_rank'] + 1 if old['returned_rank'] else None)
                    self.assertEqual(candidate['semantic_returned_rank'], old['semantic_returned_rank'])
                self.assertEqual(self.store.search("u", QUESTION, 100), hits)
                self.assertEqual(self.store.stats()["memories"], 120)

    def test_diagnostics_preserve_premerge_ranks_and_separate_stage_outcomes(self):
        self.store.conflict_strategy = None
        ordinary, original_trace = self.store.search("u", QUESTION, 100, diagnostics=True)
        self.assertEqual(original_trace['conflict_strategy']['skip_reason'], 'disabled')
        self.store.conflict_strategy = self.strategy
        self.strategy.assess.side_effect = RuntimeError('failed')
        hits, trace = self.store.search("u", QUESTION, 100, diagnostics=True)
        self.assertEqual(trace['semantic_results'], [{'id': hit.id, 'rank': rank}
                                                   for rank, hit in enumerate(ordinary, 1)])
        before = {c['id']: c['returned_rank'] for c in original_trace['candidates']}
        self.assertEqual({c['id']: c['semantic_returned_rank'] for c in trace['candidates']}, before)
        self.assertTrue(trace['conflict_strategy']['is_judgment'])
        self.assertIsNone(trace['conflict_strategy']['has_conflict'])
        self.assertEqual(trace['conflict_strategy']['assessment_status'], 'failed')
        self.strategy.assess.side_effect = None
        _, trace = self.store.search("u", QUESTION, 100, diagnostics=True)
        self.assertFalse(trace['conflict_strategy']['has_conflict'])
        self.assertEqual(trace['conflict_strategy']['assessment_status'], 'completed')
        self.strategy.assess.return_value = 'conflicting claims'
        _, trace = self.store.search("u", QUESTION, 100, diagnostics=True)
        self.assertTrue(trace['conflict_strategy']['has_conflict'])
        self.strategy.classify.return_value = None
        _, trace = self.store.search("u", QUESTION, 100, diagnostics=True)
        self.assertFalse(trace['conflict_strategy']['is_judgment'])
        self.assertEqual(trace['conflict_strategy']['assessment_status'], 'not_run')

    def test_public_api_omits_queues_and_offline_requests_include_them(self):
        from fastapi.testclient import TestClient
        from paimem import api
        self.strategy.assess.return_value = "[Conflicting evidence] Inferred correct description: " + POSITIVE
        with patch.dict(os.environ, {"MEMORY_API_KEY": "test", "MEMORY_DIAGNOSTICS_DIR": self.directory.name}, clear=True), patch.dict(
                api.app.dependency_overrides, {api.get_store: lambda: self.store}), patch.object(
                    api, "get_store", return_value=self.store):
            with TestClient(api.app, client=("127.0.0.1", 12345), headers={"X-Api-Key": "test"}) as http:
                payload = {"query": QUESTION, "user_id": "u", "top_k": 100}
                ordinary = http.post("/search", json=payload).json()["data"]
                for row in ordinary:
                    self.assertEqual(set(row), {"id", "content", "score", "created_at"})
                for flags in ({"include_recall_queue": True},
                              {"debug_output_path": str((Path(self.directory.name) / "trace.json").resolve())}):
                    result = http.post("/search", json={**payload, **flags})
                    self.assertEqual(result.status_code, 200, result.text)
                    rows = result.json()["data"]
                    self.assertEqual([r["recall_queue"] for r in rows],
                                     ["conflict_summary"] + ["positive"] * 20 + ["negative"] * 20 + ["scope"] * 20)
                    self.assertEqual([{k: v for k, v in r.items() if k != "recall_queue"} for r in rows], ordinary)
                # Offline opt-in must never carry over to subsequent public calls.
                self.assertEqual(http.post("/search", json={**payload, "include_recall_queue": False}).json()["data"], ordinary)
                trace = json.loads((Path(self.directory.name) / "trace.json").read_text())
                self.assertEqual(len(trace["final_results"]), 61)
                self.assertEqual(trace["conflict_strategy"]["explanation"], self.strategy.assess.return_value)

    def test_beam_example_through_search_and_real_strategy_parser(self):
        client = Mock()
        def complete(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            if "positive" not in payload:
                return response({"is_judgment": True, "positive_query": POSITIVE, "negative_query": NEGATIVE, "scope_query": SCOPE})
            return response(contradiction(payload["positive"][0]["id"], payload["negative"][0]["id"]))
        client.chat.completions.create.side_effect = complete
        self.store.conflict_strategy = LLMConflictStrategy(client, "test-model")
        hits, detail = self.store.search("u", QUESTION, 100, diagnostics=True)
        note = detail["conflict_strategy"]["explanation"]
        self.assertIn(POSITIVE, note)
        self.assertIn(NEGATIVE, note)
        self.assertNotIn("Which statement is correct?", note)
        self.assertIn("Inferred correct description", note)
        self.assertEqual(hits[0].content, note)
        self.assertEqual(len(hits), 61)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_disabled_options_and_non_judgment_match_original_pipeline(self):
        for enabled, options, judgment in [(False, None, QUERIES), (True, ["yes", "no"], QUERIES),
                                          (True, None, None)]:
            with self.subTest(enabled=enabled, options=options, judgment=judgment):
                self.store.conflict_strategy = None
                expected = self.store.search("u", QUESTION, 100, options)
                self.strategy.reset_mock()
                self.strategy.classify.return_value = judgment
                self.store.conflict_strategy = self.strategy if enabled else None
                actual = self.store.search("u", QUESTION, 100, options)
                self.assertEqual(actual, expected)
                self.strategy.assess.assert_not_called()
                if options or not enabled:
                    self.strategy.classify.assert_not_called()

    def test_empty_options_allow_classification_and_small_k_floors_to_zero(self):
        for k in (1, 3, 4):
            self.assertEqual(self.store.search("u", QUESTION, k, []), [])
        self.assertEqual(self.strategy.classify.call_count, 3)
        self.strategy.assess.assert_not_called()
        self.assertEqual(self.embedder.queries, [])
        self.assertEqual(len(self.store.search("u", QUESTION, 9)), 3)
        self.assertEqual(self.embedder.queries, [POSITIVE, NEGATIVE, SCOPE])

    def test_search_routes_bypasses_and_classification_failure_only_to_normal(self):
        cases = ((False, None, QUERIES, None, "bypassed", "disabled"),
                 (False, ["yes", "no"], QUERIES, None, "bypassed", "disabled"),
                 (True, ["yes", "no"], QUERIES, None, "bypassed", "options"),
                 (True, None, None, None, "not_judgment", None),
                 (True, None, QUERIES, RuntimeError("timeout"), "classification_failed", None))
        for enabled, options, judgment, error, status, skip_reason in cases:
            for diagnostics in (False, True):
                with self.subTest(enabled=enabled, options=options, status=status, diagnostics=diagnostics):
                    self.strategy.reset_mock()
                    self.strategy.classify.return_value = judgment
                    self.strategy.classify.side_effect = error
                    self.store.conflict_strategy = self.strategy if enabled else None
                    self.embedder.queries.clear()
                    with patch.object(self.store, "_classify_conflict_query", wraps=self.store._classify_conflict_query) as classify, \
                         patch.object(self.store, "_search_normal", wraps=self.store._search_normal) as normal, \
                         patch.object(self.store, "_search_conflict") as conflict:
                        result = self.store.search("u", QUESTION, 20, options, diagnostics)
                    normal.assert_called_once()
                    conflict.assert_not_called()
                    if enabled:
                        classify.assert_called_once()
                    else:
                        classify.assert_not_called()
                    if enabled and not options:
                        self.strategy.classify.assert_called_once_with(QUESTION)
                    else:
                        self.strategy.classify.assert_not_called()
                    self.strategy.assess.assert_not_called()
                    expected_query = QUESTION + ("\nOptions:\n" + "\n".join(options) if options else "")
                    self.assertEqual(self.embedder.queries, [expected_query])
                    hits = result[0] if diagnostics else result
                    self.assertEqual(len(hits), 20)
                    self.assertTrue(all(hit.recall_queue == "semantic" for hit in hits))
                    if diagnostics:
                        self.assertEqual(result[1]["conflict_strategy"]["status"], status)
                        self.assertEqual(result[1]["conflict_strategy"]["skip_reason"], skip_reason)

    def test_judgment_routes_to_conflict_with_normal_search_only_inside_diagnostics(self):
        conflict_active = False
        original_conflict = self.store._search_conflict
        original_normal = self.store._search_normal

        def run_conflict(*args, **kwargs):
            nonlocal conflict_active
            conflict_active = True
            try:
                return original_conflict(*args, **kwargs)
            finally:
                conflict_active = False

        def run_normal(*args, **kwargs):
            self.assertTrue(conflict_active, "Judgment baseline must stay inside conflict diagnostics")
            self.assertTrue(kwargs["diagnostics"])
            return original_normal(*args, **kwargs)

        for diagnostics in (False, True):
            with self.subTest(diagnostics=diagnostics):
                self.strategy.reset_mock()
                self.embedder.queries.clear()
                with patch.object(self.store, "_classify_conflict_query", wraps=self.store._classify_conflict_query) as classify, \
                     patch.object(self.store, "_search_conflict", side_effect=run_conflict) as conflict, \
                     patch.object(self.store, "_search_normal", side_effect=run_normal) as normal:
                    result = self.store.search("u", QUESTION, 20, diagnostics=diagnostics)
                conflict.assert_called_once()
                classify.assert_called_once()
                self.assertEqual(normal.call_count, int(diagnostics))
                self.strategy.classify.assert_called_once_with(QUESTION)
                self.assertEqual(self.embedder.queries,
                                 ([QUESTION] if diagnostics else []) + [POSITIVE, NEGATIVE, SCOPE])
                hits = result[0] if diagnostics else result
                self.assertEqual([hit.recall_queue for hit in hits],
                                 ["positive"] * 4 + ["negative"] * 4 + ["scope"] * 4)

    def test_empty_memories_skip_both_recall_paths_and_classification(self):
        for enabled in (False, True):
            for diagnostics in (False, True):
                with self.subTest(enabled=enabled, diagnostics=diagnostics):
                    self.store.conflict_strategy = self.strategy if enabled else None
                    with patch.object(self.store, "_classify_conflict_query", wraps=self.store._classify_conflict_query) as classify, \
                         patch.object(self.store, "_search_normal") as normal, \
                         patch.object(self.store, "_search_conflict") as conflict:
                        result = self.store.search("missing", QUESTION, 20, diagnostics=diagnostics)
                    normal.assert_not_called()
                    conflict.assert_not_called()
                    classify.assert_not_called()
                    self.strategy.classify.assert_not_called()
                    self.assertEqual(result[0] if diagnostics else result, [])
                    if diagnostics:
                        self.assertEqual(result[1]["conflict_strategy"]["skip_reason"], "empty_memories")

    def test_custom_ratios_and_zero_queue(self):
        self.store.conflict_positive_ratio = .2
        self.store.conflict_negative_ratio = .15
        hits = self.store.search("u", QUESTION, 20)
        self.assertEqual([h.recall_queue for h in hits], ["positive"] * 4 + ["negative"] * 3 + ["scope"] * 4)
        self.store.conflict_negative_ratio = 0
        self.strategy.assess.reset_mock()
        hits, detail = self.store.search("u", QUESTION, 20, diagnostics=True)
        self.assertEqual(len(hits), 8)
        self.assertIsNone(detail["conflict_strategy"]["queues"]["negative"]["pipeline"])
        self.strategy.assess.assert_not_called()

    def test_llm_failures_fall_back_without_losing_evidence(self):
        self.store.conflict_strategy = None
        expected = self.store.search("u", QUESTION, 20)
        self.store.conflict_strategy = self.strategy
        self.strategy.classify.side_effect = RuntimeError("timeout")
        hits, detail = self.store.search("u", QUESTION, 20, diagnostics=True)
        self.assertEqual(hits, expected)
        self.assertEqual(detail["conflict_strategy"]["status"], "classification_failed")
        self.strategy.classify.side_effect = None
        self.strategy.assess.side_effect = RuntimeError("malformed")
        hits, detail = self.store.search("u", QUESTION, 20, diagnostics=True)
        self.assertEqual(len(hits), 12)
        self.assertEqual(detail["conflict_strategy"]["status"], "assessment_failed")
        self.assertFalse(any(h.id.startswith("conflict:") for h in hits))

    def test_retrieval_failure_preserves_other_routes_without_backfill(self):
        def fail_positive(query, docs):
            if query == POSITIVE:
                raise RuntimeError("unavailable")
            return self.score(query, docs)
        self.reranker.score.side_effect = fail_positive
        hits, detail = self.store.search("u", QUESTION, 20, diagnostics=True)
        self.assertEqual([h.recall_queue for h in hits], ["negative"] * 4 + ["scope"] * 4)
        self.assertEqual(detail["conflict_strategy"]["status"], "retrieval_failed")
        self.strategy.assess.assert_not_called()

    def test_threshold_shortfall_never_backfills_semantic_hits(self):
        self.store.rerank_score_threshold = 99
        hits = self.store.search("u", QUESTION, 100)
        self.assertEqual(sum(h.content.startswith("[Positive-related") for h in hits), 3)
        self.assertEqual(sum(h.content.startswith("[Negative-related") for h in hits), 3)
        self.assertEqual(sum(h.recall_queue == "scope" for h in hits), 3)
        self.assertEqual(len(hits), 9)
        self.assertTrue(all(h.score >= 99 for h in hits))
        self.store.rerank_score_threshold = 2000
        self.assertEqual(self.store.search("u", QUESTION, 100), [])

    def test_missing_queue_keeps_only_other_routes(self):
        def empty_negative(query, docs):
            return [-1000.] * len(docs) if query == NEGATIVE else self.score(query, docs)
        self.store.rerank_score_threshold = 0
        self.reranker.score.side_effect = empty_negative
        hits = self.store.search("u", QUESTION, 20)
        self.assertEqual(len(hits), 8)
        self.assertEqual(sum(h.content.startswith("[Positive-related") for h in hits), 4)
        self.assertFalse(any(h.content.startswith("[Negative-related") for h in hits))
        self.strategy.assess.assert_not_called()

    def test_packing_and_expansion_apply_to_each_queue(self):
        for span in (1, 2):
            self.store.pack_span = span
            self.store.result_window = 1
            hits, detail = self.store.search("u", QUESTION, 20, diagnostics=True)
            self.assertLessEqual(len(hits), 12)
            positive = self.strategy.assess.call_args.args[1]
            candidate = next(c for c in detail["candidates"] if NEGATIVE in c["content"])
            expected_ranks = [rank for rank, hit in enumerate(hits, 1) if NEGATIVE in hit.content]
            self.assertEqual(candidate["returned_ranks"], expected_ranks)
            self.assertEqual(hits, self.store.search("u", QUESTION, 20))
            if span == 2:
                self.assertIn(NEGATIVE, positive[0].content)
            else:
                self.assertIn("ordinary memory 99", positive[1].content)

    def test_user_isolation_and_empty_store_skip_llm(self):
        self.assertEqual(self.store.search("missing", QUESTION, 100), [])
        self.strategy.classify.assert_not_called()
        self.store.add("secret", "other", "s", [MemoryMessage("user", "PRIVATE OTHER USER")])
        self.store.search("u", QUESTION, 100)
        for call in self.reranker.score.call_args_list:
            self.assertFalse(any("PRIVATE OTHER USER" in doc for doc in call.args[1]))


class ConfigurationTests(unittest.TestCase):
    def test_invalid_ratios(self):
        for pair in ((-1, .1), (float("nan"), .1), (.1, float("inf")), (.5, .5),
                     (True, .1), (".1", .1)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                validate_ratios(*pair)
        validate_ratios(0, 0)
        validate_ratios(.1, .1)
        validate_ratios(.4, .4, .2)
        with self.assertRaises(ValueError):
            validate_ratios(.2, .2, float("nan"))

    def test_defaults_environment_and_overrides(self):
        from paimem import api
        with patch.dict(os.environ, {}, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True):
            self.assertEqual(api._build_conflict_strategy(), (None, .2, .2, .2))
            api.configure_store(conflict_positive_ratio=.2, conflict_negative_ratio=0)
            os.environ["MEMORY_CONFLICT_POSITIVE_RATIO"] = "invalid"
            self.assertEqual(api._build_conflict_strategy(), (None, .2, 0, .2))
            api.configure_store(conflict_enabled=True)
            with self.assertRaisesRegex(ValueError, "LLM model"):
                api._build_conflict_strategy()
            os.environ.update(ANSWER_MODEL="test", ANSWER_API_KEY="dummy", ANSWER_API_BASE="https://example.com/v1")
            with patch("openai.OpenAI") as client:
                strategy, positive, negative, scope = api._build_conflict_strategy()
                self.assertIsInstance(strategy, LLMConflictStrategy)
                self.assertEqual((positive, negative, scope), (.2, 0, .2))
                self.assertEqual(client.call_args.kwargs["api_key"], "dummy")
                self.assertEqual(client.call_args.kwargs["base_url"], "https://example.com/v1")

    def test_invalid_configuration_fails_before_loading_models(self):
        from paimem import api
        for key, value in (("MEMORY_CONFLICT_ENABLED", "maybe"), ("MEMORY_CONFLICT_POSITIVE_RATIO", "nan"),
                           ("MEMORY_CONFLICT_NEGATIVE_RATIO", "1"), ("MEMORY_CONFLICT_SCOPE_RATIO", "-1")):
            with patch.dict(os.environ, {key: value}, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), patch.object(api, "build_embedder") as embedder:
                api.get_store.cache_clear()
                with self.assertRaises(ValueError):
                    api.get_store()
                embedder.assert_not_called()
        api.get_store.cache_clear()

    def test_api_search_and_config_keep_existing_contract(self):
        from fastapi.testclient import TestClient
        from paimem import api
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MEMORY_API_KEY": "test", "MEMORY_DB_PATH": str(Path(directory) / "api.db"),
            "MEMORY_CONFLICT_ENABLED": "true", "MEMORY_CONFLICT_MODEL": "test",
            "MEMORY_CONFLICT_API_KEY": "dummy", "MEMORY_RESULT_WINDOW": "0",
        }, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), patch.object(api, "build_embedder", return_value=RecordingEmbedder()), patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value = response({"is_judgment": False})
            api.get_store.cache_clear()
            try:
                with TestClient(api.app, headers={"X-Api-Key": "test"}) as http:
                    self.assertTrue(http.get("/config").json()["conflict_enabled"])
                    self.assertEqual(http.post("/add", json={"request_id": "r", "user_id": "u", "session_id": "s",
                        "messages": [{"role": "user", "content": POSITIVE}]}).status_code, 200)
                    result = http.post("/search", json={"query": QUESTION, "user_id": "u", "top_k": 100})
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(set(result.json()["data"][0]), {"id", "content", "score", "created_at"})
            finally:
                api.get_store.cache_clear()


if __name__ == "__main__":
    unittest.main()
