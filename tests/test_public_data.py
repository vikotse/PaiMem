import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.backfill_longmemeval_timestamps import backfill_record, backfill_splits
from scripts.local_benchmark import chunk_messages
from scripts.prepare_public_data import convert_locomo, convert_longmemeval, timestamp_ms


class PublicDataTests(unittest.TestCase):
    def test_locomo_conversion_resolves_evidence(self):
        source = [
            {
                "sample_id": "sample-1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "I moved to Shanghai."},
                        {"speaker": "Bob", "dia_id": "D1:2", "text": "Thanks for telling me."},
                    ],
                },
                "qa": [{"question": "Where did Alice move?", "answer": "Shanghai", "evidence": ["D1:1"]}],
            }
        ]
        result = convert_locomo(source)[0]
        self.assertEqual(result["messages"][0]["role"], "user")
        expected_timestamp = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual([message["timestamp"] for message in result["messages"]], [expected_timestamp] * 2)
        self.assertEqual(result["questions"][0]["evidence_texts"], ["I moved to Shanghai."])

    def test_timestamp_ms_parses_locomo_session_date(self):
        expected = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(timestamp_ms("1:56 pm on 8 May, 2023"), expected)

    def test_timestamp_ms_parses_longmemeval_session_date(self):
        self.assertEqual(timestamp_ms("2023/04/10 (Mon) 17:50"), 1681149000000)
        self.assertEqual(timestamp_ms("2023/05/20 (Sat) 02:21"), 1684549260000)
        self.assertIsNone(timestamp_ms("2023/02/30 (Thu) 00:00"))

    def test_timestamp_ms_preserves_explicit_timezone(self):
        self.assertEqual(timestamp_ms("2023-04-11T01:50:00+08:00"), 1681149000000)

    def test_longmemeval_turns_inherit_their_own_session_time(self):
        source = [{
            "question_id": "q1", "question": "What color?", "answer": "blue",
            "haystack_session_ids": ["s1", "s2"], "answer_session_ids": ["s2"],
            "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [
                [{"role": "user", "content": "Hello"},
                 {"role": "assistant", "content": "Hi"}],
                [{"role": "user", "content": "  "},
                 {"role": "user", "content": "My favorite color is blue."},
                 {"role": "assistant", "content": "Noted", "timestamp": "2023-05-20T02:22:00Z"}],
            ],
        }]
        result = convert_longmemeval(source)[0]
        self.assertEqual([m["timestamp"] for m in result["messages"]],
                         [1681149000000, 1681149000000, 1684549260000, 1684549320000])
        self.assertEqual(result["questions"][0]["evidence_texts"], ["My favorite color is blue.", "Noted"])

    def test_longmemeval_conversion(self):
        source = [
            {
                "question_id": "q1",
                "question": "What color?",
                "answer": "blue",
                "haystack_session_ids": ["s1"],
                "answer_session_ids": ["s1"],
                "haystack_sessions": [[{"role": "user", "content": "My favorite color is blue."}]],
            }
        ]
        result = convert_longmemeval(source)[0]
        self.assertEqual(result["id"], "q1")
        self.assertEqual(result["questions"][0]["evidence_texts"], ["My favorite color is blue."])

    def test_chunk_messages_splits_large_single_message(self):
        messages = [{"role": "user", "content": " ".join(f"w{i}" for i in range(12))}]
        chunks = chunk_messages(messages, max_messages=20, max_words=5)
        self.assertEqual([len(item[0]["content"].split()) for item in chunks], [5, 5, 2])

    def test_backfill_preserves_other_fields_and_repeated_turn_order(self):
        record = {"id": "q1", "session_id": "q1", "messages": [
            {"role": "user", "content": "same", "extra": "keep"},
            {"role": "user", "content": "same"}],
            "questions": [{"id": "selected", "answer": ["blue"]}], "extra": "keep"}
        original = copy.deepcopy(record)
        source = copy.deepcopy(record)
        for message, timestamp in zip(source["messages"], [1681149000000, 1684549260000]):
            message["timestamp"] = timestamp
        source["questions"].append({"id": "unselected"})
        self.assertEqual(backfill_record(record, {"q1": source}), 2)
        self.assertEqual([m["timestamp"] for m in record["messages"]], [1681149000000, 1684549260000])
        self.assertEqual(backfill_record(record, {"q1": source}), 0)
        for message in record["messages"]:
            del message["timestamp"]
        self.assertEqual(record, original)

    def test_backfill_rejects_mismatches_without_partial_mutation(self):
        source = {"id": "q1", "session_id": "q1", "messages": [
            {"role": "user", "content": "first", "timestamp": 1681149000000},
            {"role": "assistant", "content": "second", "timestamp": 1681149000000}]}
        for mismatch in ("id", "session", "count", "order", "role", "timestamp"):
            with self.subTest(mismatch=mismatch):
                record = copy.deepcopy(source)
                for message in record["messages"]:
                    del message["timestamp"]
                normalized = {"q1": copy.deepcopy(source)}
                if mismatch == "id": record["id"] = "unknown"
                elif mismatch == "session": record["session_id"] = "other"
                elif mismatch == "count": record["messages"].pop()
                elif mismatch == "order": record["messages"].reverse()
                elif mismatch == "role": record["messages"][1]["role"] = "user"
                else: del normalized["q1"]["messages"][1]["timestamp"]
                before = copy.deepcopy(record)
                with self.assertRaises(ValueError):
                    backfill_record(record, normalized)
                self.assertEqual(record, before)

    def test_backfill_validates_all_files_before_writing_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"id": "q1", "session_id": "q1", "messages": [
                {"role": "user", "content": "hello", "timestamp": 1681149000000}], "questions": []}
            target = copy.deepcopy(source)
            del target["messages"][0]["timestamp"]
            normalized, good, bad = [root / name for name in ("normalized.jsonl", "good.jsonl", "bad.jsonl")]
            normalized.write_text(json.dumps(source) + "\n", encoding="utf-8")
            good.write_text(json.dumps(target) + "\n", encoding="utf-8")
            target["id"] = "unknown"
            bad.write_text(json.dumps(target) + "\n", encoding="utf-8")
            before = good.read_bytes()
            with self.assertRaises(ValueError):
                backfill_splits(normalized, [good, bad])
            self.assertEqual(good.read_bytes(), before)
            self.assertEqual(backfill_splits(normalized, [good])[0]["updated"], 1)
            self.assertEqual(json.loads(good.read_text()), source)
            after = good.read_bytes()
            self.assertEqual(backfill_splits(normalized, [good])[0]["updated"], 0)
            self.assertEqual(good.read_bytes(), after)


if __name__ == "__main__":
    unittest.main()
