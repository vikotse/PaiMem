import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from paimem.memory_store import MemoryMessage, RequestConflictError, SQLiteMemoryStore
from paimem.role_names import (
    EvidenceRoleNameExtractor, RoleNames, merge_role_names, parse_evidence, query_role,
)


NAMES = RoleNames({"user": "James", "assistant": "John"})


class RecordingEmbedder:
    identity = "role-name-test"

    def __init__(self):
        self.documents = []

    def encode_documents(self, texts):
        self.documents.extend(texts)
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)


def evidence(index, name, quote, relation="self_introduction"):
    return {"source_turn": index, "target_name": name, "quote": quote, "relation": relation}


def response(items, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=json.dumps({"evidence": items})),
    )])


class EvidenceTests(unittest.TestCase):
    def test_roles_are_derived_from_original_turn_not_model_role_fields(self):
        turns = [{"i": 7, "role": "user", "content": "Hi John!"},
                 {"i": 8, "role": "assistant", "content": "Hi James! My name is John."}]
        items = [evidence(7, "John", "Hi John!", "direct_address"),
                 evidence(8, "James", "Hi James!", "direct_address"),
                 evidence(8, "John", "My name is John.")]
        items[0]["target_role"] = "user"
        self.assertEqual(parse_evidence(json.dumps({"evidence": items}), turns), NAMES)

    def test_invalid_evidence_is_discarded_and_all_invalid_is_error(self):
        turns = [{"i": 4, "role": "user", "content": "My name is James."}]
        invalid = [evidence(True, "James", "My name is James."),
                   evidence(0, "James", "My name is James."),
                   evidence(4, "John", "My name is John."),
                   evidence(4, "Jam", "My name is James."),
                   evidence(4, " James ", "My name is James."),
                   evidence(4, "James", "My name is James.", "third_person"), None]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_evidence(json.dumps({"evidence": [item]}), turns)
        result = parse_evidence(json.dumps({"evidence": invalid + [
            evidence(4, "James", "My name is James.")]}), turns)
        self.assertEqual(result.names, {"user": "James"})

    def test_conflict_clears_both_roles_and_case_variants_do_not_conflict(self):
        turns = [{"i": 1, "role": "user", "content": "My name is James."},
                 {"i": 2, "role": "user", "content": "My name is Robert."}]
        result = parse_evidence(json.dumps({"evidence": [
            evidence(1, "James", turns[0]["content"]),
            evidence(2, "Robert", turns[1]["content"])]}), turns)
        self.assertTrue(result.conflicted)
        self.assertEqual(result.names, {})
        self.assertEqual(merge_role_names(NAMES, RoleNames({"user": "JAMES"})), NAMES)
        self.assertEqual(merge_role_names(NAMES, RoleNames({"assistant": "James"})), NAMES)
        self.assertEqual([item["target_name"] for item in result.evidence], ["James", "Robert"])
        self.assertEqual(result.evidence[1]["quote"], "My name is Robert.")

    def test_empty_and_malformed_responses(self):
        self.assertEqual(parse_evidence('{"evidence":[]}', []), RoleNames())
        for body in ("not json", "[]", "{}", '{"evidence":null}'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                parse_evidence(body, [])

    def test_query_routing_boundaries_ambiguity_and_cjk(self):
        for query, expected in (("James's hobbies?", "user"), ("JOHN?", "assistant"),
                                ("Jameson?", None), ("James and John?", None), ("hobbies?", None)):
            with self.subTest(query=query):
                self.assertEqual(query_role(query, NAMES), expected)
        self.assertEqual(query_role("小明喜欢什么？", RoleNames({"user": "小明"})), "user")
        self.assertIsNone(query_role("Ann Marie?", RoleNames({"user": "Ann", "assistant": "Ann Marie"})))
        self.assertIsNone(query_role("James?", RoleNames(conflicted=True)))

    def test_candidate_selection_keeps_original_indices_context_and_budget(self):
        messages = [MemoryMessage("system", "ignore"), MemoryMessage("user", "boring"),
                    MemoryMessage("assistant", "Hi James! trailing"),
                    MemoryMessage("user", "Hi John! trailing"), MemoryMessage("assistant", "end")]
        reranker = Mock()
        reranker.score.return_value = np.array([0., 3., 1., 0.])
        client = Mock()
        client.chat.completions.create.return_value = response([
            evidence(2, "James", "Hi James!", "direct_address"),
            evidence(3, "John", "Hi John!", "direct_address")])
        extractor = EvidenceRoleNameExtractor(client, "test", reranker, top_k=1, window=1,
                                             context_chars=9, disable_thinking=True)
        self.assertEqual(extractor.extract(messages), NAMES)
        payload = client.chat.completions.create.call_args.kwargs
        turns = json.loads(payload["messages"][1]["content"])["turns"]
        self.assertEqual([turn["i"] for turn in turns], [1, 2, 3])
        self.assertTrue(all(len(turn["content"]) <= 9 for turn in turns))
        self.assertEqual(payload["extra_body"], {"enable_thinking": False})
        self.assertEqual(reranker.score.call_args.args[1], [m.content for m in messages[1:]])

    def test_empty_transcript_invalid_scores_and_truncated_output(self):
        client, reranker = Mock(), Mock()
        extractor = EvidenceRoleNameExtractor(client, "test", reranker)
        self.assertEqual(extractor.extract([MemoryMessage("system", "text")]), RoleNames())
        client.chat.completions.create.assert_not_called()
        for scores in ([], [float("nan")]):
            reranker.score.return_value = scores
            with self.assertRaises(ValueError):
                extractor.extract([MemoryMessage("user", "hello")])
        reranker.score.return_value = [1.]
        client.chat.completions.create.return_value = response([], "length")
        with self.assertRaises(ValueError):
            extractor.extract([MemoryMessage("user", "hello")])


class RoleStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "memory.db"
        self.embedder = RecordingEmbedder()
        self.extractor = Mock()
        self.extractor.extract.return_value = NAMES

    def tearDown(self):
        self.directory.cleanup()

    def store(self, enabled=True, **kwargs):
        return SQLiteMemoryStore(self.path, self.embedder,
                                 role_name_extractor=self.extractor if enabled else None, **kwargs)

    def add(self, store, request="r1", user="u1"):
        store.add(request, user, "s1", [MemoryMessage("user", "I enjoy chess."),
                                       MemoryMessage("assistant", "I enjoy hiking."),
                                       MemoryMessage("user", "I also enjoy swimming.")])

    def state(self, user="u1"):
        with sqlite3.connect(self.path) as connection:
            return connection.execute("SELECT user_name, assistant_name, conflicted "
                                      "FROM user_role_names WHERE user_id = ?", (user,)).fetchone()

    def misses(self, user="u1"):
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT consecutive_no_new_names FROM role_name_extraction_state WHERE user_id = ?",
                (user,),
            ).fetchone()
        return row[0] if row is not None else 0

    def test_two_empty_adds_stop_extraction_across_sessions_and_restart(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames()
        self.add(store)
        self.assertEqual(self.misses(), 1)
        self.add(self.store(), "r2")  # Same session ID, distinct /add batch.
        self.assertEqual(self.misses(), 2)
        self.extractor.extract.return_value = NAMES
        reopened = self.store()
        reopened.add("r3", "u1", "another-session", [MemoryMessage("user", "My name is James.")])
        self.assertEqual(self.extractor.extract.call_count, 2)
        self.assertIsNone(self.state())
        self.assertEqual(self.misses(), 2)
        self.assertEqual(reopened.stats()["memories"], 7)

    def test_new_name_resets_streak_and_repeated_names_do_not(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames()
        self.add(store)
        self.extractor.extract.return_value = RoleNames({"user": "James"})
        self.add(store, "r2")
        self.assertEqual(self.misses(), 0)
        self.extractor.extract.return_value = RoleNames({"user": "JAMES"})
        self.add(store, "r3")
        self.assertEqual(self.misses(), 1)
        self.add(store, "r4")
        self.assertEqual(self.misses(), 2)
        self.extractor.extract.return_value = NAMES
        self.add(store, "r5")
        self.assertEqual(self.extractor.extract.call_count, 4)
        self.assertEqual(self.state(), ("James", None, 0))
        hits = store.search("u1", "James", 100)
        self.assertTrue(all(hit.content.startswith("[James]") for hit in hits))

    def test_rejected_conflicts_count_as_no_new_names(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames({"user": "James"})
        self.add(store)
        # A conflicting batch must not reset the streak even if it includes a missing role.
        self.extractor.extract.return_value = RoleNames({"user": "Robert", "assistant": "John"})
        with self.assertLogs("paimem.memory_store", level="WARNING"):
            self.add(store, "r2")
            self.add(store, "r3")
        self.add(store, "r4")
        self.assertEqual(self.extractor.extract.call_count, 3)
        self.assertEqual(self.misses(), 2)
        self.assertEqual(self.state(), ("James", None, 0))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM role_name_conflicts").fetchone()[0], 2)

    def test_extraction_exceptions_count_as_no_new_names(self):
        store = self.store()
        self.extractor.extract.side_effect = RuntimeError("provider failed")
        with self.assertLogs("paimem.memory_store", level="WARNING"):
            self.add(store)
            self.add(store, "r2")
        self.add(store, "r3")
        self.assertEqual(self.extractor.extract.call_count, 2)
        self.assertEqual(self.misses(), 2)
        self.assertEqual(store.stats()["requests"], 3)

    def test_streak_is_user_scoped_and_clear_resets_it(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames()
        self.add(store)
        self.add(store, "r2")
        self.extractor.extract.return_value = NAMES
        self.add(store, "r3", "u2")
        self.assertEqual(self.state("u2"), ("James", "John", 0))
        self.assertEqual(self.misses("u1"), 2)
        self.assertEqual(self.misses("u2"), 0)
        store.clear()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM role_name_extraction_state").fetchone()[0], 0)
        self.add(store)
        self.assertEqual(self.state(), ("James", "John", 0))
        self.assertEqual(self.extractor.extract.call_count, 4)

    def test_idempotent_conflicting_and_disabled_adds_do_not_advance_streak(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames()
        self.add(store)
        self.add(store)
        with self.assertRaises(RequestConflictError):
            store.add("r1", "u1", "s1", [MemoryMessage("user", "different")])
        self.add(self.store(enabled=False), "r2")
        self.assertEqual(self.misses(), 1)
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.extractor.extract.return_value = NAMES
        self.add(store, "r3")
        self.assertEqual(self.state(), ("James", "John", 0))
        self.assertEqual(self.misses(), 0)

    def test_failed_add_does_not_advance_streak_and_can_be_retried(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames()
        self.add(store)
        save_names = store._save_role_names

        def fail_after_saving(*args):
            save_names(*args)
            raise RuntimeError("write failed")

        with patch.object(store, "_save_role_names", side_effect=fail_after_saving):
            with self.assertRaises(RuntimeError):
                self.add(store, "r2")
        self.assertEqual(self.misses(), 1)
        self.assertEqual(store.stats()["requests"], 1)
        self.extractor.extract.return_value = NAMES
        self.add(store, "r2")
        self.assertEqual(self.state(), ("James", "John", 0))
        self.assertEqual(self.misses(), 0)

    def test_existing_database_starts_with_zero_misses_without_reembedding(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames({"user": "James"})
        self.add(store)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE role_name_extraction_state")
        count = len(self.embedder.documents)
        upgraded = self.store()
        self.assertEqual(self.misses(), 0)
        self.assertEqual(self.state(), ("James", None, 0))
        self.assertEqual(len(self.embedder.documents), count)
        self.extractor.extract.return_value = RoleNames()
        self.add(upgraded, "r2")
        self.add(upgraded, "r3")
        self.add(upgraded, "r4")
        self.assertEqual(self.extractor.extract.call_count, 3)
        self.assertEqual(self.misses(), 2)

    def test_filter_precedes_every_retrieval_stage_and_names_replace_role_labels(self):
        for mode in ("dense", "hybrid", "enhanced"):
            with self.subTest(mode=mode):
                reranker = Mock()
                reranker.score.side_effect = lambda query, docs: np.ones(len(docs))
                store = self.store(retrieval_mode=mode, reranker=reranker,
                                   result_window=2, pack_span=3, temporal_enrichment=True)
                self.add(store)
                for query, role, name, other in (("James hobbies", "user", "James", "hiking"),
                                               ("John hobbies", "assistant", "John", "chess")):
                    hits, detail = store.search("u1", query, 10, ["John", "James"], diagnostics=True)
                    self.assertTrue(hits)
                    self.assertEqual(detail["selected_role"], role)
                    self.assertEqual(detail["stored_count"], 2 if role == "user" else 1)
                    for hit in hits:
                        self.assertNotIn(other, hit.content)
                        self.assertNotIn("[user", hit.content)
                        self.assertNotIn("[assistant", hit.content)
                        self.assertTrue(all(line.startswith(f"[{name}]") for line in hit.content.splitlines()))
                    self.assertTrue(all(other not in doc for doc in reranker.score.call_args.args[1]))
                store.clear()

    def test_no_name_and_disabled_keep_original_behavior(self):
        self.extractor.extract.return_value = RoleNames()
        store = self.store()
        self.add(store)
        self.assertIsNone(self.state())
        hits, detail = store.search("u1", "James", 10, diagnostics=True)
        self.assertEqual(len(hits), 3)
        self.assertIsNone(detail["selected_role"])
        self.assertTrue(hits[0].content.startswith("[user]"))
        self.assertTrue(all("[James]" not in doc for doc in self.embedder.documents))
        disabled = self.store(enabled=False)
        self.add(disabled, "r2")
        self.assertEqual(self.extractor.extract.call_count, 1)

    def test_persistence_user_isolation_and_disabling_with_existing_mapping(self):
        store = self.store()
        self.add(store)
        self.assertEqual(self.state(), ("James", "John", 0))
        self.assertIn("[James]", self.embedder.documents[0])
        self.assertIn("[John]", self.embedder.documents[1])
        self.extractor.extract.return_value = RoleNames({"user": "John", "assistant": "James"})
        self.add(store, "r2", "u2")
        reopened = self.store()
        self.assertIn("hiking", reopened.search("u2", "James", 10)[0].content)
        self.assertEqual(len(reopened.search("u1", "James", 10)), 2)
        self.assertEqual(reopened.search("absent", "James", 10), [])
        disabled = self.store(enabled=False)
        hits = disabled.search("u1", "James", 10)
        self.assertEqual(len(hits), 3)
        self.assertTrue(any(hit.content.startswith("[assistant]") for hit in hits))

    def test_ambiguity_and_names_in_options_do_not_filter(self):
        store = self.store()
        self.add(store)
        for query in ("James and John", "hobbies", "Jameson"):
            hits, detail = store.search("u1", query, 10, ["James"], diagnostics=True)
            self.assertEqual(len(hits), 3)
            self.assertIsNone(detail["selected_role"])

    def test_partial_mapping_can_be_completed_and_then_reused(self):
        self.extractor.extract.return_value = RoleNames({"user": "James"})
        store = self.store()
        self.add(store)
        self.extractor.extract.return_value = RoleNames()
        self.add(store, "r2")
        self.assertEqual(self.state(), ("James", None, 0))
        self.extractor.extract.return_value = RoleNames({"assistant": "John"})
        self.add(store, "r3")
        self.assertEqual(self.state(), ("James", "John", 0))
        self.extractor.extract.return_value = RoleNames({"user": "Robert"})
        self.add(store, "r4")
        self.assertEqual(self.state(), ("James", "John", 0))
        self.add(store, "r5")
        self.assertEqual(self.extractor.extract.call_count, 3)
        hits, detail = store.search("u1", "James", 100, diagnostics=True)
        self.assertEqual(len(hits), 10)
        self.assertFalse(detail["role_names_conflicted"])
        self.assertTrue(hits[0].content.startswith("[James]"))
        store.clear()
        self.assertIsNone(self.state())
        self.add(store)
        self.assertEqual(self.state(), ("Robert", None, 0))

    def test_extraction_failure_is_nonfatal_and_does_not_log_provider_message(self):
        store = self.store()
        self.extractor.extract.side_effect = RuntimeError("secret-token")
        with self.assertLogs("paimem.memory_store", level="WARNING") as logs:
            self.add(store)
        self.assertNotIn("secret-token", " ".join(logs.output))
        self.assertIsNone(self.state())
        self.assertEqual(len(store.search("u1", "James", 10)), 3)
        self.extractor.extract.side_effect = None
        self.extractor.extract.return_value = RoleNames({"user": "James"})
        self.add(store, "r2")
        self.extractor.extract.side_effect = RuntimeError("secret-token")
        with self.assertLogs("paimem.memory_store", level="WARNING"):
            self.add(store, "r3")
        self.assertEqual(self.state(), ("James", None, 0))

    def test_complete_names_skip_third_person_extraction_across_batches_and_restart(self):
        store = self.store()
        self.add(store)
        # A provider might misclassify Jill as the speaker's name. It is never
        # consulted again once this user's participant mapping is complete.
        self.extractor.extract.side_effect = AssertionError("must not reextract")
        reopened = self.store()
        for i in range(1, 35):
            reopened.add(f"r{i + 1}", "u1", "s1", [
                MemoryMessage("user", "Souvenirs for both you and your Jill!"),
                MemoryMessage("assistant", "Thank you very much, Jill will be delighted!")])
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.assertEqual(self.state(), ("James", "John", 0))
        for name in ("James", "John"):
            hits = reopened.search("u1", name, 100)
            self.assertTrue(all(hit.content.startswith(f"[{name}]") for hit in hits))

    def test_conflicting_batch_preserves_partial_mapping_and_saves_evidence(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames({"user": "James"})
        self.add(store)
        turns = [{"i": 0, "role": "assistant", "content": "Hello Robert!"}]
        self.extractor.extract.return_value = parse_evidence(json.dumps({"evidence": [
            evidence(0, "Robert", "Hello Robert!", "direct_address")]}), turns)
        with self.assertLogs("paimem.memory_store", level="WARNING"):
            self.add(store, "r2")
        self.assertEqual(self.state(), ("James", None, 0))
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT request_id, user_id, previous_names, incoming_names, evidence "
                                     "FROM role_name_conflicts").fetchone()
        self.assertEqual(row[:2], ("r2", "u1"))
        self.assertEqual(json.loads(row[2]), {"user": "James"})
        self.assertEqual(json.loads(row[3]), {"user": "Robert"})
        self.assertEqual(json.loads(row[4])[0]["quote"], "Hello Robert!")
        self.add(store, "r2")  # Idempotent replay must not duplicate audit records.
        self.extractor.extract.return_value = RoleNames({"assistant": "John"})
        self.add(store, "r3")
        self.assertEqual(self.state(), ("James", "John", 0))
        store.clear()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM role_name_conflicts").fetchone()[0], 0)

    def test_initial_conflict_does_not_latch_and_legacy_conflict_can_recover(self):
        store = self.store()
        self.extractor.extract.return_value = RoleNames(conflicted=True)
        with self.assertLogs("paimem.memory_store", level="WARNING"):
            self.add(store)
        self.assertIsNone(self.state())
        self.extractor.extract.return_value = NAMES
        self.add(store, "r2")
        self.assertEqual(self.state(), ("James", "John", 0))
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE user_role_names SET user_name=NULL, assistant_name=NULL, conflicted=1")
        self.add(store, "r3")
        self.assertEqual(self.state(), ("James", "John", 0))
        self.assertEqual(self.extractor.extract.call_count, 3)

    def test_concurrent_store_conflict_is_checked_again_in_write_transaction(self):
        store = self.store()
        other_extractor = Mock()
        other_extractor.extract.return_value = NAMES
        other = SQLiteMemoryStore(self.path, RecordingEmbedder(), role_name_extractor=other_extractor)

        def extract(_):
            self.add(other, "other")
            return RoleNames({"user": "Robert"})

        self.extractor.extract.side_effect = extract
        with self.assertLogs("paimem.memory_store", level="WARNING"):
            self.add(store)
        self.assertEqual(self.state(), ("James", "John", 0))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT request_id FROM role_name_conflicts").fetchone()[0], "r1")

    def test_idempotence_and_request_conflicts_do_not_reextract(self):
        store = self.store()
        self.add(store)
        self.add(store)
        with self.assertRaises(RequestConflictError):
            store.add("r1", "u1", "s1", [MemoryMessage("user", "different")])
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.assertEqual(store.stats()["memories"], 3)

    def test_mapping_and_memories_roll_back_together(self):
        store = self.store()
        with patch.object(store, "_save_role_names", side_effect=RuntimeError("write failed")):
            with self.assertRaises(RuntimeError):
                self.add(store)
        self.assertEqual(store.stats(), {"requests": 0, "memories": 0, "users": 0})
        self.assertIsNone(self.state())

    def test_filtered_scope_with_no_matching_memories_is_empty(self):
        store = self.store()
        store.add("r1", "u1", "s1", [MemoryMessage("user", "Hi John!")])
        hits, detail = store.search("u1", "John", 10, diagnostics=True)
        self.assertEqual(hits, [])
        self.assertEqual(detail["selected_role"], "assistant")

    def test_original_transcript_and_named_embedding_context(self):
        store = self.store(context_messages=1, resolve_relative_time_on_add=True)
        messages = [MemoryMessage("user", "My name is James. Yesterday I swam.", 1704153600000),
                    MemoryMessage("assistant", "Hi James!", 1704153600000)]
        store.add("r1", "u1", "s1", messages)
        self.assertEqual(self.extractor.extract.call_args.args[0], messages)
        self.assertTrue(self.embedder.documents[0].startswith("James:"))
        self.assertIn("Next John:", self.embedder.documents[0])
        self.assertIn("January 1, 2024", self.embedder.documents[0])
        self.assertNotIn("Yesterday", self.embedder.documents[0])
        self.assertTrue(store.search("u1", "John", 10)[0].content.startswith("[John |"))

    def test_existing_database_without_role_table_upgrades_without_reembedding(self):
        store = self.store(enabled=False)
        self.add(store)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE user_role_names")
        count = len(self.embedder.documents)
        upgraded = self.store()
        self.assertEqual(len(upgraded.search("u1", "James", 10)), 3)
        self.assertEqual(len(self.embedder.documents), count)


class RoleApiTests(unittest.TestCase):
    def test_startup_switch_and_named_api_results(self):
        from fastapi.testclient import TestClient
        from paimem import api

        for enabled in (None, "false", "true"):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), \
                    patch.object(api, "build_embedder", return_value=RecordingEmbedder()), \
                    patch.object(api, "build_reranker", return_value=Mock()) as build_reranker, \
                    patch("openai.OpenAI") as openai:
                os.environ.update(MEMORY_DB_PATH=str(Path(directory) / "api.db"),
                                  MEMORY_API_KEY="test", MEMORY_ROLE_NAME_MODEL="test-model",
                                  MEMORY_ROLE_NAME_API_KEY="test-key",
                                  MEMORY_ROLE_NAME_RERANKER_MODEL="test-reranker")
                if enabled is not None:
                    os.environ["MEMORY_ROLE_NAME_ENABLED"] = enabled
                build_reranker.return_value.score.return_value = np.array([1., 1.])
                openai.return_value.chat.completions.create.return_value = response([
                    evidence(0, "James", "My name is James."),
                    evidence(1, "John", "My name is John.")])
                api.get_store.cache_clear()
                try:
                    with TestClient(api.app, headers={"X-Api-Key": "test"}) as client:
                        added = client.post("/add", json={
                            "request_id": "r1", "user_id": "u1", "session_id": "s1",
                            "messages": [{"role": "user", "content": "My name is James."},
                                         {"role": "assistant", "content": "My name is John."}]})
                        self.assertEqual(added.status_code, 200, added.text)
                        found = client.post("/search", json={"user_id": "u1", "query": "James", "top_k": 10})
                        self.assertEqual(found.status_code, 200, found.text)
                        self.assertEqual(len(found.json()["data"]), 1 if enabled == "true" else 2)
                        self.assertEqual(client.get("/config").json()["role_name_enabled"], enabled == "true")
                        if enabled == "true":
                            self.assertTrue(found.json()["data"][0]["content"].startswith("[James]"))
                            openai.assert_called_once()
                            self.assertIsNone(api.get_store().reranker)
                        else:
                            openai.assert_not_called()
                            build_reranker.assert_not_called()
                finally:
                    api.get_store.cache_clear()

    def test_enabled_requires_configuration_disabled_does_not(self):
        from paimem import api
        with patch.dict(os.environ, {}, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True):
            self.assertIsNone(api._build_role_name_extractor(None, "", None))
            os.environ["MEMORY_ROLE_NAME_ENABLED"] = "true"
            with self.assertRaises(ValueError):
                api._build_role_name_extractor(None, "", None)

    def test_answer_config_fallback_reuses_search_reranker(self):
        from paimem import api
        with patch.dict(os.environ, {"MEMORY_ROLE_NAME_ENABLED": "true", "ANSWER_MODEL": "answer",
                                    "ANSWER_API_KEY": "key", "ANSWER_API_BASE": "https://example.test/v1"}, clear=True), \
                patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), patch("openai.OpenAI") as openai, \
                patch.object(api, "build_reranker") as build:
            reranker = Mock()
            extractor = api._build_role_name_extractor(reranker, "existing", None)
            self.assertIs(extractor.reranker, reranker)
            self.assertEqual(extractor.model, "answer")
            build.assert_not_called()
            openai.assert_called_once_with(api_key="key", base_url="https://example.test/v1",
                                          timeout=60., max_retries=0)


if __name__ == "__main__":
    unittest.main()
