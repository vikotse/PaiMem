from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    for key in ("data", "records", "samples"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError("Expected a JSON array, JSONL records, or an object containing data/records/samples")


def timestamp_ms(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number > 10_000_000_000 else number * 1000
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for date_format in (
            "%Y/%m/%d (%a) %H:%M",  # LongMemEval haystack_dates (session time)
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %b, %Y",
        ):
            try:
                parsed = datetime.strptime(text, date_format)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def write_jsonl(records: list[dict[str, Any]], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_locomo(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(records):
        conversation = sample.get("conversation", sample)
        speaker_a = str(conversation.get("speaker_a", ""))
        messages: list[dict[str, Any]] = []
        evidence_by_id: dict[str, str] = {}
        session_keys = sorted(
            (key for key, value in conversation.items() if re.fullmatch(r"session_\d+", key) and isinstance(value, list)),
            key=lambda key: int(key.split("_")[-1]),
        )
        for session_key in session_keys:
            session_timestamp = timestamp_ms(conversation.get(f"{session_key}_date_time"))
            for turn in conversation[session_key]:
                content = str(turn.get("text", turn.get("content", ""))).strip()
                if not content:
                    continue
                speaker = str(turn.get("speaker", ""))
                role = str(turn.get("role", ""))
                if role not in {"user", "assistant"}:
                    role = "user" if speaker_a and speaker == speaker_a else "assistant"
                message = {"role": role, "content": content}
                turn_timestamp = timestamp_ms(turn.get("timestamp")) or session_timestamp
                if turn_timestamp is not None:
                    message["timestamp"] = turn_timestamp
                messages.append(message)
                evidence_id = turn.get("dia_id", turn.get("id"))
                if evidence_id is not None:
                    evidence_by_id[str(evidence_id)] = content

        questions = []
        for question_index, qa in enumerate(sample.get("qa", sample.get("questions", []))):
            evidence_ids = [str(value) for value in qa.get("evidence", [])]
            question = {
                "id": str(qa.get("id", qa.get("question_id", f"q-{question_index}"))),
                "question": str(qa["question"]),
                "answer": str(qa.get("answer", "")),
                "evidence_texts": [evidence_by_id[value] for value in evidence_ids if value in evidence_by_id],
            }
            if qa.get("options"):
                question["options"] = qa["options"]
            questions.append(question)
        converted.append(
            {
                "id": str(sample.get("sample_id", sample.get("id", f"locomo-{sample_index}"))),
                "session_id": str(sample.get("sample_id", sample.get("id", f"locomo-{sample_index}"))),
                "messages": messages,
                "questions": questions,
            }
        )
    return converted


def convert_longmemeval(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(records):
        sessions = sample.get("haystack_sessions", sample.get("sessions", []))
        dates = sample.get("haystack_dates", [])
        messages: list[dict[str, Any]] = []
        evidence_texts: list[str] = []
        answer_sessions = {str(value) for value in sample.get("answer_session_ids", [])}
        session_ids = sample.get("haystack_session_ids", [])
        for session_index, session in enumerate(sessions):
            session_timestamp = timestamp_ms(dates[session_index]) if session_index < len(dates) else None
            session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
            for turn in session:
                content = str(turn.get("content", turn.get("text", ""))).strip()
                if not content:
                    continue
                role = str(turn.get("role", "user"))
                if role not in {"user", "assistant"}:
                    role = "user"
                message = {"role": role, "content": content}
                turn_timestamp = timestamp_ms(turn.get("timestamp")) or session_timestamp
                if turn_timestamp is not None:
                    message["timestamp"] = turn_timestamp
                messages.append(message)
                if session_id in answer_sessions:
                    evidence_texts.append(content)
        question_id = str(sample.get("question_id", sample.get("id", f"longmemeval-{sample_index}")))
        converted.append(
            {
                "id": question_id,
                "session_id": question_id,
                "messages": messages,
                "questions": [
                    {
                        "id": question_id,
                        "question": str(sample["question"]),
                        "answer": str(sample.get("answer", "")),
                        "evidence_texts": evidence_texts,
                    }
                ],
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public memory datasets to local benchmark JSONL")
    parser.add_argument("dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = read_records(args.input)
    converted = convert_locomo(source) if args.dataset == "locomo" else convert_longmemeval(source)
    write_jsonl(converted, args.output)
    print(json.dumps({"dataset": args.dataset, "records": len(converted), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
