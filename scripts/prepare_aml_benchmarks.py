"""Convert the remaining AML public benchmarks to the local normalized JSONL format.

Complements scripts/prepare_public_data.py (locomo, longmemeval) so that every
benchmark family in agent-memory-leaderboard/downloads/benchmarks has a local
adapter. Output schema matches the existing normalized files:

    {"id", "session_id", "messages": [{role, content, timestamp?}],
     "questions": [{"id", "question", "answer", ["options"], [...passthrough]}]}

Family notes:
  - scriptmem:   the upstream release withholds the script conversation text for
                 copyright reasons (only a schema example ships), so records carry
                 the 457 questions grouped by movie with EMPTY messages. Retrieval
                 evaluation is impossible locally until the full scripts are
                 obtained; question management/scoring alignment still works.
  - clbench:     one record per task row. The context lives in the row's leading
                 system message (some rows add follow-up assistant/user turns);
                 the final user message is the question. Gold answers do not
                 exist (rubric-judged), so rubrics/categories pass through and
                 evidence_recall stays None.
  - personamem:  one record per 32k chat history (200 files); every CSV row that
                 links to it becomes a question with options [correct]+incorrect.
  - beam:        one record per conversation per scale (100K/1M). Evaluated
                 questions are the parsed probing_questions (question ->
                 ideal_response); plan-specific user_questions are part of the
                 chat itself and are not questions.

Usage:
    uv run python scripts/prepare_aml_benchmarks.py all
    uv run python scripts/prepare_aml_benchmarks.py clbench
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "agent-memory-leaderboard" / "downloads" / "benchmarks"
OUT_DIR = REPO_ROOT / "data" / "normalized"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_literal(value: str) -> Any:
    """Parse a string that may be JSON or a Python literal."""
    text = str(value).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return ast.literal_eval(text)


def normalize_role(role: object, default: str = "user") -> str:
    role = str(role or "").lower()
    return role if role in {"user", "assistant"} else default


def convert_scriptmem() -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    total_q = 0
    for raw in sorted((BENCH / "scriptmem" / "data" / "raw").glob("*.json")):
        source = raw.stem  # angry / enemy / friends / man_earth
        item = json.loads(raw.read_text())[0]
        questions = []
        for index, qa in enumerate(item.get("qa", [])):
            question: dict[str, Any] = {
                # raw files carry no qa_id (the packaged questions.jsonl does); use an index id
                "id": str(qa.get("qa_id") or qa.get("question_id") or f"{source}-q{index:03d}"),
                "question": str(qa["question"]),
                "answer": qa.get("answer", ""),
                "options": qa.get("option"),
                "qa_type": qa.get("qa_type"),
                "answer_letters": qa.get("answer_letters"),
            }
            questions.append({k: v for k, v in question.items() if v is not None})
        records.append(
            {
                "id": source,
                "session_id": source,
                "messages": [],  # script text withheld upstream (copyright); see module docstring
                "questions": questions,
                "note": "script conversation text not public; messages empty",
            }
        )
        total_q += len(questions)
    return "scriptmem", records


def convert_clbench() -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with (BENCH / "clbench" / "CL-bench.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            task_id = str(meta.get("task_id"))
            messages = row["messages"]
            question_turn = messages[-1]
            context_turns = messages[:-1]
            question = {
                "id": task_id,
                "question": str(question_turn.get("content", "")),
                "answer": "",  # rubric-judged: no gold answer upstream
                "rubrics": row.get("rubrics", []),
                "context_category": meta.get("context_category"),
                "sub_category": meta.get("sub_category"),
                "context_id": meta.get("context_id"),
            }
            records.append(
                {
                    "id": task_id,
                    "session_id": str(meta.get("context_id", task_id)),
                    "messages": [
                        {"role": normalize_role(t.get("role")), "content": str(t.get("content", ""))}
                        for t in context_turns
                    ],
                    "questions": [{k: v for k, v in question.items() if v is not None}],
                }
            )
    return "clbench", records


def convert_personamem() -> tuple[str, list[dict[str, Any]]]:
    by_link: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (BENCH / "personamem-v2" / "benchmark.csv").open(encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            by_link[row["chat_history_32k_link"]].append(row)

    records: list[dict[str, Any]] = []
    for link in sorted(by_link):
        history_path = BENCH / "personamem-v2" / link
        history = json.loads(history_path.read_text())
        turns = history.get("chat_history", history if isinstance(history, list) else [])
        record_id = history_path.stem
        questions = []
        for index, row in enumerate(by_link[link]):
            query = parse_literal(row["user_query"])
            content = query.get("content", row["user_query"]) if isinstance(query, dict) else str(query)
            incorrect = parse_literal(row.get("incorrect_answers", "[]"))
            options = [row["correct_answer"]] + [str(v) for v in incorrect]
            questions.append(
                {
                    "id": f"{record_id}-q{index:03d}",
                    "question": str(content),
                    "answer": str(row["correct_answer"]),
                    "options": options,
                }
            )
        records.append(
            {
                "id": record_id,
                "session_id": record_id,
                "messages": [
                    {"role": normalize_role(t.get("role")), "content": str(t.get("content", ""))}
                    for t in turns
                ],
                "questions": questions,
            }
        )
    return "personamem_v2", records


def convert_beam() -> dict[str, list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    out: dict[str, list[dict[str, Any]]] = {}
    for scale, path in [
        ("beam_100k", BENCH / "beam" / "100K-00000-of-00001.parquet"),
        ("beam_1m", BENCH / "beam" / "1M-00000-of-00001.parquet"),
    ]:
        records = []
        for row in pq.read_table(path).to_pylist():
            messages = []
            for session in row["chat"]:
                for turn in session:
                    messages.append(
                        {"role": normalize_role(turn.get("role")), "content": str(turn.get("content", ""))}
                    )
            probing = ast.literal_eval(row["probing_questions"])
            questions = []
            for qtype, items in probing.items():
                for index, item in enumerate(items):
                    questions.append(
                        {
                            "id": f"{row['conversation_id']}#{qtype}#{index}",
                            "question": str(item["question"]),
                            "answer": str(item.get("ideal_response", "")),
                            "qa_type": qtype,
                            # official BEAM evaluate() judges per-rubric via
                            # rubric_items(); without this field the judge stage
                            # has nothing to score against
                            "rubric": item.get("rubric"),
                        }
                    )
            records.append(
                {
                    "id": str(row["conversation_id"]),
                    "session_id": str(row["conversation_id"]),
                    "messages": messages,
                    "questions": questions,
                }
            )
        out[scale] = records
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["scriptmem", "clbench", "personamem", "beam", "all"])
    args = parser.parse_args()

    jobs = {
        "scriptmem": lambda: [convert_scriptmem()],
        "clbench": lambda: [convert_clbench()],
        "personamem": lambda: [convert_personamem()],
        "beam": lambda: list(convert_beam().items()),
    }
    selected = jobs if args.dataset == "all" else {args.dataset: jobs[args.dataset]}
    for name, job in selected.items():
        for key, records in job():
            out = OUT_DIR / f"{key}.jsonl"
            write_jsonl(out, records)
            n_q = sum(len(r["questions"]) for r in records)
            print(json.dumps({"dataset": key, "records": len(records), "questions": n_q, "output": str(out)}))


if __name__ == "__main__":
    main()
