import threading
import unittest
from unittest.mock import Mock

import numpy as np

from paimem.retrieval import retrieve_chunks


class ChunkRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.embedder = Mock()
        self.embedder.encode_queries.return_value = np.asarray([[1., 0.]], dtype=np.float32)
        self.rows = [dict(id=str(i), role="user", content=content, dimension=2,
                          embedding=np.asarray(vector, dtype=np.float32).tobytes())
                     for i, (content, vector) in enumerate([
                         ("neutral first", [1., 0.]),
                         ("neutral second", [.8, .2]),
                         ("hobbies favorites", [.1, .9]),
                         ("unrelated", [0., 1.]),
                     ])]

    def retrieve(self, **kwargs):
        settings = dict(embedder=self.embedder, embedding_lock=threading.RLock(),
                        render_content=lambda row, names: row["content"],
                        lexical_weight=2., rerank_candidates=2, diagnostics=True)
        settings.update(kwargs)
        return retrieve_chunks(self.rows, {}, "hobbies", 1, **settings)

    def test_bm25_can_recall_candidate_outside_dense_top_before_rerank(self):
        reranker = Mock()
        reranker.score.side_effect = lambda q, docs: [1. if "hobbies" in d else 0. for d in docs]
        result = self.retrieve(reranker=reranker)
        self.assertEqual(reranker.score.call_args.args, ("hobbies", ["hobbies favorites", "neutral first"]))
        self.assertEqual(result.positions.tolist(), [2, 0])
        self.assertEqual(result.reranked_count, 2)
        detail = {c["id"]: c for c in result.diagnostics["candidates"]}
        self.assertEqual(detail["2"]["channels"]["dense"]["rank"], 3)
        self.assertEqual(detail["2"]["channels"]["bm25"]["rank"], 1)
        self.assertEqual(detail["1"]["channels"]["reranker"], {"score": None, "rank": None})

    def test_rerank_ties_keep_fusion_order_and_threshold_trace(self):
        reranker = Mock(score=Mock(return_value=[.5, .5]))
        result = self.retrieve(reranker=reranker, rerank_score_threshold=.5)
        self.assertEqual(result.positions.tolist(), [2, 0])
        rejected = self.retrieve(reranker=reranker, rerank_score_threshold=.6)
        self.assertEqual(rejected.positions.tolist(), [])
        self.assertEqual(rejected.diagnostics["rerank_candidate_count"], 2)
        self.assertEqual(sum(c["channels"]["reranker"]["rank"] is not None
                             for c in rejected.diagnostics["candidates"]), 2)

    def test_invalid_rerank_scores_are_rejected(self):
        for scores in ([1.], [float("nan"), 1.], [float("inf"), 1.]):
            with self.subTest(scores=scores), self.assertRaisesRegex(ValueError, "invalid scores"):
                self.retrieve(reranker=Mock(score=Mock(return_value=scores)))

    def test_no_reranker_preserves_dense_order_without_candidate_cap(self):
        result = self.retrieve(retrieval_mode="dense")
        self.assertEqual(result.positions.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.reranked_count, 0)
        plain = self.retrieve(retrieval_mode="dense", diagnostics=False)
        np.testing.assert_array_equal(plain.positions, result.positions)
        np.testing.assert_array_equal(plain.scores, result.scores)
        self.assertIsNone(plain.diagnostics)

    def test_empty_scope_does_not_call_models(self):
        self.rows = []
        reranker = Mock()
        result = self.retrieve(reranker=reranker)
        self.assertEqual(result.positions.tolist(), [])
        self.assertEqual(result.diagnostics["candidates"], [])
        self.embedder.encode_queries.assert_not_called()
        reranker.score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
