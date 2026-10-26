import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from paimem.memory_store import MemoryMessage, SQLiteMemoryStore


class StaticEmbedder:
    def encode_documents(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class OrderedReranker:
    def __init__(self, order):
        self.order = order
        self.documents = []

    def score(self, query, documents):
        self.documents = list(documents)
        return [next((len(self.order) - rank for rank, marker in enumerate(self.order)
                      if marker in document), 0) for document in documents]


class V2PackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store_count = 0

    def store(self, order, **settings):
        self.store_count += 1
        config = dict(retrieval_mode="dense", pack_span=2, result_window=1,
                      result_window_seed_k=1)
        config.update(settings)
        reranker = OrderedReranker(order)
        store = SQLiteMemoryStore(Path(self.directory.name) / f"{self.store_count}.db",
                                  StaticEmbedder(), reranker=reranker, **config)
        return store

    def add(self, store, markers, session="s1", request="r1"):
        store.add(request, "u1", session,
                  [MemoryMessage("user", marker, 1704067200000 + i * 1000)
                   for i, marker in enumerate(markers)])

    def assert_markers(self, hit, markers):
        self.assertEqual([line.rsplit("] ", 1)[-1] for line in hit.content.splitlines()], markers)

    def test_bidirectional_content_is_chronological_but_identity_is_seed(self):
        store = self.store(["CENTER", "BEFORE", "AFTER", "TAIL"])
        self.add(store, ["BEFORE", "CENTER", "AFTER", "TAIL"])
        hits, detail = store.search("u1", "query", 2, diagnostics=True)
        self.assertEqual(len(hits), 2)
        self.assert_markers(hits[0], ["BEFORE", "CENTER", "AFTER"])
        self.assert_markers(hits[1], ["TAIL"])
        center = next(row for row in detail["candidates"] if row["content"].endswith("CENTER"))
        self.assertEqual(hits[0].id, center["id"])
        self.assertEqual(hits[0].created_at, "2024-01-01T00:00:01Z")
        self.assertEqual(hits[0].score, 4.0)
        self.assertEqual(center["channels"]["reranker"]["rank"], 1)
        for row in detail["candidates"]:
            self.assertEqual(row["returned_rank"], 2 if row["content"].endswith("TAIL") else 1)
        # Reranking still sees individual messages, before any content packing.
        self.assertEqual(len(store.reranker.documents), 4)
        self.assertTrue(all("\n" not in doc for doc in store.reranker.documents))

    def test_window_radius_controls_content_even_with_pack_span_two(self):
        store = self.store(["CENTER"], result_window=2)
        self.add(store, ["LEFT_FAR", "LEFT_NEAR", "CENTER", "RIGHT_NEAR", "RIGHT_FAR"])
        hits = store.search("u1", "query", 1)
        self.assert_markers(hits[0], ["LEFT_FAR", "LEFT_NEAR", "CENTER", "RIGHT_NEAR", "RIGHT_FAR"])

    def test_result_assembly_uses_seed_score_even_if_neighbor_scores_higher(self):
        store = self.store(["BEFORE", "CENTER", "AFTER"])
        self.add(store, ["BEFORE", "CENTER", "AFTER"])
        # Isolate result assembly from the current greedy packing policy, which
        # normally visits higher-scoring rows first and marks them covered.
        with patch.object(store, "_pack_result_positions", return_value=[[1, 0, 2]]):
            hits, detail = store.search("u1", "query", 1, diagnostics=True)
        self.assert_markers(hits[0], ["BEFORE", "CENTER", "AFTER"])
        center = next(row for row in detail["candidates"] if row["content"].endswith("CENTER"))
        self.assertEqual(hits[0].id, center["id"])
        self.assertEqual(hits[0].score, center["channels"]["reranker"]["score"])
        self.assertEqual(hits[0].score, 2.0)

    def test_unscored_neighbor_does_not_override_negative_seed_score(self):
        store = self.store(["CENTER"], rerank_candidates=1)
        store.reranker.score = Mock(return_value=[-2.0])
        self.add(store, ["CENTER", "UNSCORED"])
        hits, detail = store.search("u1", "query", 1, diagnostics=True)
        self.assertEqual(len(hits), 1)
        self.assert_markers(hits[0], ["CENTER", "UNSCORED"])
        self.assertEqual(hits[0].score, -2.0)
        neighbor = next(row for row in detail["candidates"] if row["content"].endswith("UNSCORED"))
        self.assertIsNone(neighbor["channels"]["reranker"]["score"])
        self.assertEqual(neighbor["returned_rank"], 1)

    def test_pack_span_above_two_is_also_an_enable_switch(self):
        for span in (2, 3, 5):
            with self.subTest(span=span):
                store = self.store(["CENTER"], pack_span=span)
                self.add(store, ["BEFORE", "CENTER", "AFTER", "TAIL"])
                self.assert_markers(store.search("u1", "query", 1)[0], ["BEFORE", "CENTER", "AFTER"])

    def test_pack_span_one_keeps_neighbors_as_individual_results(self):
        store = self.store(["CENTER"], pack_span=1)
        self.add(store, ["BEFORE", "CENTER", "AFTER", "TAIL"])
        hits = store.search("u1", "query", 3)
        self.assertEqual(len(hits), 3)
        for hit, marker in zip(hits, ["CENTER", "BEFORE", "AFTER"]):
            self.assert_markers(hit, [marker])

    def test_zero_window_keeps_rerank_order_and_individual_results(self):
        store = self.store(["CENTER", "AFTER", "BEFORE"], result_window=0)
        self.add(store, ["BEFORE", "CENTER", "AFTER"])
        hits = store.search("u1", "query", 3)
        for hit, marker in zip(hits, ["CENTER", "AFTER", "BEFORE"]):
            self.assert_markers(hit, [marker])
        self.assertEqual(len(hits), 3)

    def test_seed_limit_uses_original_ranks_not_number_of_uncovered_results(self):
        store = self.store(["CENTER", "AFTER", "LATER", "END", "BEFORE"],
                           result_window_seed_k=2)
        self.add(store, ["BEFORE", "CENTER", "AFTER", "LATER", "END"])
        hits = store.search("u1", "query", 3)
        self.assertEqual(len(hits), 3)
        self.assert_markers(hits[0], ["BEFORE", "CENTER", "AFTER"])
        self.assert_markers(hits[1], ["LATER"])
        self.assert_markers(hits[2], ["END"])

    def test_overlapping_windows_deduplicate_and_backfill_up_to_top_k(self):
        store = self.store(["SEED_A", "SEED_B", "BEFORE", "SHARED", "AFTER", "TAIL", "END"],
                           result_window_seed_k=2)
        self.add(store, ["BEFORE", "SEED_A", "SHARED", "SEED_B", "AFTER", "TAIL", "END"])
        hits, detail = store.search("u1", "query", 3, diagnostics=True)
        self.assertEqual(len(hits), 3)
        self.assert_markers(hits[0], ["BEFORE", "SEED_A", "SHARED"])
        self.assert_markers(hits[1], ["SEED_B", "AFTER"])
        self.assert_markers(hits[2], ["TAIL"])
        end = next(row for row in detail["candidates"] if row["content"].endswith("END"))
        self.assertIsNone(end["returned_rank"])

    def test_session_checks_match_expansion_in_both_directions(self):
        store = self.store(["CENTER"], result_window=2)
        for i, (marker, session) in enumerate([
            ("LEFT_SAME", "s1"), ("LEFT_OTHER", "s2"), ("CENTER", "s1"),
            ("RIGHT_OTHER", "s2"), ("RIGHT_SAME", "s1"),
        ]):
            self.add(store, [marker], session=session, request=f"r{i}")
        hits = store.search("u1", "query", 1)
        self.assert_markers(hits[0], ["LEFT_SAME", "CENTER", "RIGHT_SAME"])
        store.update_config(pack_span=1)
        expanded = store.search("u1", "query", 3)
        self.assertEqual({line.rsplit("] ", 1)[-1] for line in hits[0].content.splitlines()},
                         {hit.content.rsplit("] ", 1)[-1] for hit in expanded})

    def test_windows_clip_at_both_ends(self):
        for seed in ("FIRST", "LAST"):
            with self.subTest(seed=seed):
                store = self.store([seed], result_window=3)
                self.add(store, ["FIRST", "LAST"])
                hits = store.search("u1", "query", 5)
                self.assertEqual(len(hits), 1)
                self.assert_markers(hits[0], ["FIRST", "LAST"])

    def test_runtime_toggle_changes_packing_without_reingesting(self):
        store = self.store(["CENTER"], pack_span=1)
        self.add(store, ["BEFORE", "CENTER", "AFTER"])
        before = store.search("u1", "query", 1)[0]
        self.assert_markers(before, ["CENTER"])
        store.update_config(pack_span=2)
        after = store.search("u1", "query", 1)[0]
        self.assert_markers(after, ["BEFORE", "CENTER", "AFTER"])
        self.assertEqual(after.id, before.id)


if __name__ == "__main__":
    unittest.main()
