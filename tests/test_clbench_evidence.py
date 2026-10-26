import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_clbench_evidence import canonicalize, build_output, validate, save_output, read_rows


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id="r", session_id="s", messages=[dict(role="user", content="Rule A: use blue.\nRule B: use red.")],
                        questions=[dict(id="q", question="The sample is blue. What rule applies?", answer="", rubrics=["Apply Rule A"])])
        self.q = self.row["questions"][0]

    def entry(self, text, source="message"):
        return dict(source=source, message_index=0 if source == "message" else None,
                    text=text, kind="rule", rubric_indices=[0], reason="premise")

    def test_question_only_is_not_a_failed_retrieval(self):
        a = canonicalize(self.row, self.q, dict(evidence=[self.entry("The sample is blue.", "question")]))
        self.assertEqual(a["evidence_status"], "question_only")
        self.assertEqual(a["evidence_texts"], [])
        self.assertEqual(a["evidence"][0]["start"], 0)

    def test_mixed_sources_and_deduplication(self):
        a = canonicalize(self.row, self.q, dict(evidence=[self.entry("Rule A: use blue."), self.entry("Rule A: use blue."),
                                                     self.entry("The sample is blue.", "question")]))
        self.assertEqual(len(a["evidence"]), 2)
        self.assertEqual(a["evidence_texts"], ["Rule A: use blue."])
        out = build_output([self.row], {"r/q": a})
        self.assertTrue(validate([self.row], out)["original_fields_unchanged"])

    def test_edited_or_rubric_only_quotes_rejected(self):
        for text in ["Rule A: USE blue.", "Apply Rule A"]:
            with self.assertRaises(ValueError):
                canonicalize(self.row, self.q, dict(evidence=[self.entry(text)]))

    def test_question_quote_also_in_memory_is_recallable(self):
        self.q["question"] += " Rule A: use blue."
        a = canonicalize(self.row, self.q, dict(evidence=[self.entry("Rule A: use blue.", "question")]))
        self.assertEqual(a["evidence_status"], "annotated")
        self.assertEqual(a["evidence"][0]["memory_locations"][0]["message_index"], 0)

    def test_unicode_separators_work_with_existing_benchmark_reader(self):
        self.row["messages"][0]["content"] += "\u2028hello\u0085world\u2029"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            save_output(path, [self.row])
            self.assertEqual([json.loads(x) for x in path.read_text().splitlines()], [self.row])
            self.assertEqual(read_rows(path), [self.row])

    def test_tampering_detected(self):
        a = canonicalize(self.row, self.q, dict(evidence=[self.entry("Rule A: use blue.")]))
        out = build_output([self.row], {"r/q": a})
        tampered = copy.deepcopy(out)
        tampered[0]["questions"][0]["evidence"][0]["start"] = 1
        with self.assertRaises(ValueError):
            validate([self.row], tampered)
        out[0]["questions"][0]["question"] += "changed"
        with self.assertRaises(ValueError):
            validate([self.row], out)

    def test_existing_local_benchmark_recall_denominator(self):
        from scripts import local_benchmark
        a = canonicalize(self.row, self.q, dict(evidence=[self.entry("Rule A: use blue."),
            self.entry("Rule B: use red."), self.entry("The sample is blue.", "question")]))
        out = build_output([self.row], {"r/q": a})
        only_query = copy.deepcopy(out[0]["questions"][0])
        only_query["id"] = "question-only"
        only_query.update(canonicalize(self.row, self.q, dict(evidence=[self.entry("The sample is blue.", "question")])))
        out[0]["questions"].append(only_query)
        def request(method, url, *args):
            if url.endswith("/add"):
                return 200, {"success": True}
            return 200, {"data": [{"content": "Rule A: use blue."}]}
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory) / "in.jsonl", Path(directory) / "out.jsonl"
            save_output(src, out)
            with patch.object(local_benchmark, "request_json", side_effect=request), patch("sys.argv", [
                "local_benchmark", "--input", str(src), "--output", str(dst), "--no-progress"
            ]), contextlib.redirect_stdout(io.StringIO()) as stdout:
                local_benchmark.main()
            results = read_rows(dst)
            self.assertEqual(results[0]["evidence_recall_at_k"], 0.5)
            self.assertIsNone(results[1]["evidence_recall_at_k"])
            self.assertEqual(json.loads(stdout.getvalue())["evidence_recall_at_k"], 0.5)


if __name__ == "__main__":
    unittest.main()
