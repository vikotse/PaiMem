"""Map normalized LongMemEval timestamps onto existing splits without repartitioning.

Example:
    python scripts/backfill_longmemeval_timestamps.py \
      --normalized data/normalized/longmemeval_oracle.jsonl \
      --splits data/splits/longmemeval_oracle.dev.jsonl \
               data/splits/longmemeval_oracle.validation.jsonl

All targets are validated before writing. Only messages[*].timestamp changes;
record order, question selection, and other fields are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterator


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_timestamp_map(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for record in read_rows(path):
        record_id = record["id"]
        if record_id in records:
            raise ValueError(f"Duplicate normalized record id: {record_id}")
        for index, message in enumerate(record["messages"]):
            if type(message.get("timestamp")) is not int:
                raise ValueError(f"{record_id}: message {index} has no integer timestamp")
        records[record_id] = record
    if not records:
        raise ValueError(f"No normalized records in {path}")
    return records


def backfill_record(record: dict[str, Any], normalized: dict[str, dict[str, Any]]) -> int:
    """Validate message identity, then update only timestamps in place."""
    record_id = record["id"]
    source = normalized.get(record_id)
    if source is None:
        raise ValueError(f"Unknown split record id: {record_id}")
    if record["session_id"] != source["session_id"]:
        raise ValueError(f"{record_id}: session_id mismatch")
    messages = record["messages"]
    expected = source["messages"]
    if len(messages) != len(expected):
        raise ValueError(f"{record_id}: message count mismatch")
    # Check every turn before changing any, including repeated message content.
    for index, (message, original) in enumerate(zip(messages, expected)):
        if any(message.get(key) != original.get(key) for key in ("role", "content")):
            raise ValueError(f"{record_id}: message {index} role/content mismatch")
        if type(original.get("timestamp")) is not int:
            raise ValueError(f"{record_id}: message {index} has no integer timestamp")
    changed = 0
    for message, original in zip(messages, expected):
        changed += type(message.get("timestamp")) is not int or message["timestamp"] != original["timestamp"]
        message["timestamp"] = original["timestamp"]
    return changed


def backfill_splits(normalized_path: Path, paths: list[Path]) -> list[dict[str, Any]]:
    normalized = load_timestamp_map(normalized_path)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate split paths")
    if normalized_path.resolve() in resolved:
        raise ValueError("Normalized source cannot also be a split target")
    reports = []
    # A missing id or mismatched turn in any target prevents all writes.
    for path in paths:
        seen = set()
        messages = changed = 0
        for record in read_rows(path):
            if record["id"] in seen:
                raise ValueError(f"{path}: duplicate split record id {record['id']}")
            seen.add(record["id"])
            changed += backfill_record(record, normalized)
            messages += len(record["messages"])
        reports.append({"path": str(path), "records": len(seen), "messages": messages, "updated": changed})
    for path, report in zip(paths, reports):
        if not report["updated"]:
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                for record in read_rows(path):
                    backfill_record(record, normalized)
                    # Match the line-safe encoding used by the existing splits.
                    stream.write(json.dumps(record, ensure_ascii=True) + "\n")
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normalized", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    for report in backfill_splits(args.normalized, args.splits):
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
