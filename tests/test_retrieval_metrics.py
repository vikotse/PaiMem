from pathlib import Path
import sys

import numpy as np
import pytest
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retrieval_metrics import (ranking_metrics, parent_metrics, overlap_groups, paired_summary,
                               answer_turn_gold, answer_turn_metrics, returned_chunk_metrics)
from reference_retrieval import make_bank, fused_order, reconstruct_top100
from paimem.memory_store import SQLiteMemoryStore, MemoryMessage, bm25_scores
from paimem.near import NearBank, NearConfig


def test_capped_recall_and_ndcg_known_values():
    gold = {str(i) for i in range(20)}
    result = ranking_metrics([str(i) for i in range(8)] + ["x", "y"], gold)
    assert result["recall@10"] == .4
    assert result["capped_recall@10"] == .8
    ideal = sum(1 / np.log2(i + 2) for i in range(10))
    expected = sum(1 / np.log2(i + 2) for i in range(8)) / ideal
    np.testing.assert_allclose(result["ndcg@10"], expected)
    assert ranking_metrics(["a", "b"], {"a", "b"})["ndcg@10"] == 1
    np.testing.assert_allclose(ranking_metrics(["b", "a"], {"a"})["ndcg@10"], 1 / np.log2(3))


def test_invalid_labels_and_duplicate_ranking():
    with np.testing.assert_raises(ValueError):
        ranking_metrics(["a"], set())
    with np.testing.assert_raises(ValueError):
        ranking_metrics(["a", "a"], {"a"})


def test_parent_dedup_does_not_hide_chunk_budget():
    result = parent_metrics([0, 1, 2], ["a", "a", "b"], {"b"}, ks=(2,))
    assert result["capped_recall@2"] == 1
    assert result["parent_coverage_in_2_chunks"] == 0


def test_overlap_transitivity_and_single_cluster_no_ci():
    groups = overlap_groups([["a"], ["a", "b"], ["b"], ["c"]])
    assert groups[0] == groups[1] == groups[2] != groups[3]
    row = {"group": "one", "arms": {"base": {"metrics": {"ndcg@10": 0}}, "near": {"metrics": {"ndcg@10": 1}}}}
    result = paired_summary([row, row], "base", "near")
    assert result["metrics"]["ndcg@10"]["cluster_bootstrap_95_ci"] is None
    second = {**row, "group": "two"}
    assert paired_summary([row, second], "base", "near")["metrics"]["ndcg@10"]["cluster_bootstrap_95_ci"] == [1., 1.]


def test_original_source_mapping_and_abstention():
    raw = {"question_id": "q_abs", "question": "Where?", "question_type": "single-session-user",
           "haystack_session_ids": ["s1", "s2"], "haystack_sessions": [[{"content": "A"}], [{"content": "B"}]],
           "answer_session_ids": ["s1"]}
    record = {"id": "q_abs", "messages": [{"content": "A"}, {"content": "B"}],
              "questions": [{"id": "q_abs", "question": "Where?"}]}
    _, parents, _, questions = make_bank("longmemeval_s", record, raw)
    assert parents == ["s1", "s2"]
    assert questions[0]["excluded_reason"] == "abstention_question"
    record["messages"][0]["content"] = "wrong"
    with np.testing.assert_raises_regex(ValueError, "alignment mismatch"):
        make_bank("longmemeval_s", record, raw)


def test_partial_missing_gold_excludes_entire_question():
    raw = {"conversation": {"session_1": [{"dia_id": "D1:1", "text": "A"}]},
           "qa": [{"question": "Where?", "evidence": ["D1:1", "D9:9"]}]}
    record = {"id": "c", "messages": [{"content": "A"}], "questions": [{"id": "q-0", "question": "Where?"}]}
    _, _, _, questions = make_bank("locomo_refined", record, raw)
    assert questions[0]["excluded_reason"] == "gold_source_id_missing_from_bank"
    assert questions[0]["gold"] == ["D1:1", "D9:9"]


class FakeEmbedder:
    identity = "retrieval-metrics-parity"

    def encode_documents(self, texts):
        import hashlib
        matrix = np.array([list(hashlib.sha256(t.encode()).digest())[:8] for t in texts], dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    encode_queries = encode_documents


def test_answer_turn_labels_skip_empty_and_do_not_label_whole_session():
    raw = {"question_id": "q", "haystack_session_ids": ["s"], "answer_session_ids": ["s"],
           "haystack_sessions": [[{"content": "", "has_answer": False}, {"content": "distractor", "has_answer": False},
                                  {"content": "answer", "has_answer": True}, {"content": "more distractors"}]]}
    assert answer_turn_gold(raw) == ["q:message-1"]
    raw["question_id"] = "q_abs"
    with np.testing.assert_raises(ValueError):
        answer_turn_gold(raw)


def test_answer_turn_metrics_use_unique_messages_and_guard_truncation():
    chunks = ["q:message-0:chunk-0", "q:message-0:chunk-1"] + [f"q:message-{i}:chunk-0" for i in range(1, 99)]
    metrics, ranking = answer_turn_metrics(chunks, ["q:message-9"])
    assert metrics["capped_recall@10"] == 1
    assert metrics["parent_coverage_in_10_chunks"] == 0
    assert ranking[9] == "q:message-9"
    with np.testing.assert_raises(ValueError):
        answer_turn_metrics(chunks[:5], ["q:message-9"])


def test_fixed_return_budget_keeps_duplicate_parent_slots():
    parents = {"a0": "a", "a1": "a", "b0": "b", "c0": "c"}
    metrics = returned_chunk_metrics(["a0", "a1"], parents, {"a", "b"}, top_k=2)
    assert metrics["evidence_recall@2"] == .5
    assert metrics["complete_evidence@2"] == 0
    assert metrics["capped_chunk_recall@2"] == 1
    assert metrics["chunk_ndcg@2"] == 1
    # The two chunks of a are relevant chunks, but cover only one of two facts.
    assert metrics["chunk_recall@2"] == 2/3


def test_fixed_return_budget_rejects_partial_or_invalid_lists():
    parents = {f"c{i}": f"m{i}" for i in range(110)}
    with np.testing.assert_raises(ValueError):
        returned_chunk_metrics(list(parents)[:99], parents, {"m0"})
    with np.testing.assert_raises(ValueError):
        returned_chunk_metrics(["c0"] * 100, parents, {"m0"})
    with np.testing.assert_raises(ValueError):
        returned_chunk_metrics(list(parents)[:100], parents, {"missing"})
    small = {"c0": "m0", "c1": "m1"}
    assert returned_chunk_metrics(list(small), small, {"m0"})["evidence_recall@100"] == 1


def test_top100_prefix_exactly_matches_full_expansion():
    rng = np.random.default_rng(20261015)
    store = SQLiteMemoryStore.__new__(SQLiteMemoryStore)
    store.result_window = 1
    store.result_window_seed_k = 20
    for n in (50, 100, 101, 250, 1000):
        bank = [f"chunk-{i}" for i in range(n)]
        sessions = [str(i // 40) for i in range(n)]
        rows = [{"session_id": s} for s in sessions]
        for _ in range(100):
            order = rng.permutation(n)
            expected = [bank[int(i)] for i in store._expand_result_positions(order, rows, min(100, n))]
            actual = reconstruct_top100([bank[int(i)] for i in order[:100]], bank, sessions)
            assert actual == expected
    with np.testing.assert_raises(ValueError):
        reconstruct_top100(["c0"], ["c0", "c1"], ["s", "s"])


def test_top100_reconstruction_matches_serving_search():
    embedder = FakeEmbedder()
    messages = [MemoryMessage("user", f"entry {i} apple pear fruit topic {i % 11}", None) for i in range(160)]
    with tempfile.TemporaryDirectory(prefix="near-top100-parity-") as tmp:
        for mode in ("dense", "hybrid"):
            for enabled in (False, True):
                store = SQLiteMemoryStore(Path(tmp) / f"{mode}-{enabled}.db", embedder,
                                          retrieval_mode=mode, lexical_weight=.5, result_window=1,
                                          result_window_seed_k=20, pack_span=1,
                                          near_config=NearConfig(enabled=enabled, normalize=True))
                store.add("request", "user", "session", messages)
                with store._connect() as db:
                    rows = db.execute("SELECT id, session_id FROM memories ORDER BY rowid").fetchall()
                hits, trace = store.search("user", "apple fruit topic 7", 100, diagnostics=True)
                prefix = [r["id"] for r in sorted(trace["candidates"], key=lambda r:r["channels"]["fusion"]["rank"])[:100]]
                expected = reconstruct_top100(prefix, [r["id"] for r in rows], [r["session_id"] for r in rows])
                assert expected == [h.id for h in hits]
                assert len(hits) == 100


@pytest.mark.parametrize("enabled", [False, True])
def test_offline_ranking_and_text_match_serving_path(tmp_path, enabled):
    embedder = FakeEmbedder()
    config = NearConfig(enabled=enabled, normalize=True, weight=1.)
    messages = [MemoryMessage("user", "apple " * 400, 1700000000000),
                MemoryMessage("assistant", "banana apple", 1700000000000),
                MemoryMessage("user", "pear", None)]
    raw = {"conversation": {"session_1": [{"dia_id": str(i), "text": m.content} for i, m in enumerate(messages)]}, "qa": []}
    record = {"id": "c", "messages": [{"content": m.content, "timestamp": m.timestamp} for m in messages], "questions": []}
    docs, _, _, _ = make_bank("locomo_refined", record, raw)
    store = SQLiteMemoryStore(tmp_path / "memory.db", embedder, retrieval_mode="hybrid", lexical_weight=.5, near_config=config)
    store.add("r", "u", "s", messages)
    with store._connect() as db:
        saved = db.execute("SELECT content, embedding FROM memories ORDER BY rowid").fetchall()
    assert [r["content"] for r in saved] == docs
    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in saved])
    np.testing.assert_array_equal(matrix, embedder.encode_documents(docs))
    query = "apple fruit"
    vector = embedder.encode_queries([query])
    scores = NearBank(matrix, config).score(vector)[:, 0] if enabled else (matrix @ vector.T)[:, 0]
    order = fused_order(scores, bm25_scores(docs, query))
    _, trace = store.search("u", query, 100, diagnostics=True)
    with store._connect() as db:
        row_ids = [r[0] for r in db.execute("SELECT id FROM memories ORDER BY rowid")]
    expected = [row_ids[int(i)] for i in order]
    actual = [c["id"] for c in sorted(trace["candidates"], key=lambda c: c["channels"]["fusion"]["rank"])]
    assert actual == expected


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, fn in globals().items():
        if name.startswith("test_") and name != "test_offline_ranking_and_text_match_serving_path":
            suite.addTest(unittest.FunctionTestCase(fn))
    for enabled in (False, True):
        def parity(enabled=enabled):
            with tempfile.TemporaryDirectory(prefix="near-retrieval-test-") as tmp:
                test_offline_ranking_and_text_match_serving_path(Path(tmp), enabled)
        suite.addTest(unittest.FunctionTestCase(parity, description=f"serving parity NEAR={enabled}"))
    return suite


if __name__ == "__main__":
    unittest.main()
