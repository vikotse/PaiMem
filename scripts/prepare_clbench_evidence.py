"""Build source-verified CLBench evidence from direct assistant annotations.

No model APIs or network calls. Original inputs are preserved. Evidence is
manually selected by the assistant and stored in the audit file; this script
only validates positions and builds the local_benchmark-compatible JSONL.
--review prints lexical candidate passages as a navigation aid, not gold labels.
Only reviewed records are emitted; audit explicitly counts unannotated records.

  python -B scripts/prepare_clbench_evidence.py --review 0,1 --candidates 8
  python -B scripts/prepare_clbench_evidence.py
  python -B scripts/prepare_clbench_evidence.py --validate-only
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
VERSION = "clbench-evidence-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    # Iterate physical JSONL lines: str.splitlines() also splits U+2028/U+0085.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def source_text(record, question, source, index):
    if source == "question" and index is None:
        return question["question"]
    if source == "message" and type(index) is int and 0 <= index < len(record["messages"]):
        return record["messages"][index]["content"]
    raise ValueError("Invalid source or message_index")


def canonicalize(record, question, annotation):
    """Reject invented/edited quotes; attach exact Python character offsets."""
    if not isinstance(annotation, dict) or not isinstance(annotation.get("evidence"), list):
        raise ValueError("Missing evidence array")
    result, seen = [], set()
    for entry in annotation["evidence"]:
        source, index = entry.get("source"), entry.get("message_index")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty evidence text")
        original = source_text(record, question, source, index)
        start = original.find(text)
        if start < 0:
            raise ValueError("Evidence is not an exact source substring: " + repr(text[:90]))
        key = (source, index, text)
        if key in seen:
            continue
        seen.add(key)
        rubric_indices = entry.get("rubric_indices", [])
        if not isinstance(rubric_indices, list) or any(
            type(i) is not int or not 0 <= i < len(question.get("rubrics", [])) for i in rubric_indices
        ):
            raise ValueError("Invalid rubric index")
        if entry.get("kind") not in {"fact", "rule", "procedure", "conversation", "response_constraint"}:
            raise ValueError("Invalid evidence kind")
        memory_locations = []
        for mi, message in enumerate(record["messages"]):
            pos = message["content"].find(text)
            if pos >= 0:
                memory_locations.append(dict(message_index=mi, start=pos, end=pos + len(text)))
        result.append(dict(source=source, message_index=index, start=start,
                           end=start + len(text), text=text, kind=entry["kind"],
                           rubric_indices=sorted(set(rubric_indices)), reason=str(entry.get("reason", "")),
                           memory_locations=memory_locations))
    texts = list(dict.fromkeys(e["text"] for e in result if e["memory_locations"]))
    status = "annotated" if texts else "question_only" if result else "no_source_evidence"
    return dict(evidence=result, evidence_texts=texts, evidence_status=status,
                evidence_note=str(annotation.get("note", "")))


def build_output(rows, annotations):
    output = copy.deepcopy(rows)
    for record in output:
        for question in record["questions"]:
            key = record["id"] + "/" + question["id"]
            item = annotations.get(key)
            if item is None:
                item = dict(evidence=[], evidence_texts=[], evidence_status="pending",
                            evidence_note="Not annotated yet")
            question.update({k: v for k, v in item.items() if k.startswith("evidence")})
    return output


def validate(source, output):
    if len(source) != len(output):
        raise ValueError("Record count changed")
    counts, kinds, total, memory, query = Counter(), Counter(), 0, 0, 0
    for original, record in zip(source, output, strict=True):
        if {k: v for k, v in original.items() if k != "questions"} != {k: v for k, v in record.items() if k != "questions"}:
            raise ValueError("Original record/messages changed")
        if len(original["questions"]) != len(record["questions"]):
            raise ValueError("Question count changed")
        for old, question in zip(original["questions"], record["questions"], strict=True):
            if any(question.get(k) != v for k, v in old.items()):
                raise ValueError("Original question field changed")
            expected = canonicalize(record, question, dict(evidence=question["evidence"], note=question["evidence_note"]))
            if expected["evidence"] != question["evidence"] or expected["evidence_texts"] != question["evidence_texts"]:
                raise ValueError("Evidence offsets or recall targets changed")
            if question["evidence_status"] not in {"pending", "unresolved"} and question["evidence_status"] != expected["evidence_status"]:
                raise ValueError("Evidence status is inconsistent")
            counts[question["evidence_status"]] += 1
            for entry in question["evidence"]:
                total += 1
                kinds[entry["kind"]] += 1
                memory += bool(entry["memory_locations"])
                query += entry["source"] == "question"
    return dict(records=len(output), questions=sum(counts.values()), statuses=dict(counts),
                evidence_spans=total, memory_available_spans=memory, question_source_spans=query,
                kinds=dict(kinds), original_fields_unchanged=True, exact_source_spans_verified=True)


def save_output(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in rows:
            # Escape Unicode separators so unmodified local_benchmark.splitlines works.
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
    temporary.replace(path)


def candidates(record, question, count=10):
    """Read-only navigation aid; lexical ranking is NOT an evidence annotation."""
    stop = set("the a an of in on and or to is are be it that this for with should response include provide explain state would must from as by at not can which how what your you have has its their when use using than any all such if also".split())
    def terms(text):
        return set(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)) - stop
    blocks = []
    for source, mi, text in [("message", i, m["content"]) for i, m in enumerate(record["messages"])] + [("question", None, question["question"])]:
        # Contiguous spans, with exact offsets. Paragraphs capped at 100 words.
        for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
            words = list(re.finditer(r"\S+", paragraph[0]))
            for start in range(0, len(words), 80):
                end = min(start + 100, len(words))
                lo = paragraph.start() + words[start].start()
                hi = paragraph.start() + words[end-1].end()
                fragment = text[lo:hi]
                blocks.append(dict(source=source, message_index=mi, start=lo, end=hi, text=fragment))
    docs = [terms(b["text"]) for b in blocks]
    frequency = Counter(w for d in docs for w in d)
    queries = question.get("rubrics", [])
    ranked = []
    for rubric_index, query in enumerate(queries):
        ts = terms(query)
        scores = [sum(1 / (1 + frequency[w]) ** .5 for w in ts & d) / (max(15, len(d)) ** .25) for d in docs]
        for index in sorted(range(len(blocks)), key=lambda i: -scores[i])[:2]:
            ranked.append((scores[index], index, rubric_index))
    picks = {}
    for score, index, ri in sorted(ranked, reverse=True):
        if index not in picks and len(picks) >= count:
            continue
        picks.setdefault(index, dict(**blocks[index], rubric_hints=[]))["rubric_hints"].append(ri)
    return list(picks.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/normalized/clbench.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/normalized/clbench_evidence.jsonl")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/normalized/clbench_evidence.audit.json")
    parser.add_argument("--review", help="Comma-separated zero-based source row numbers; read only")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.review:
        for index in map(int, args.review.split(",")):
            record = rows[index]
            q = record["questions"][0]
            print(json.dumps(dict(row=index, id=record["id"], category=q["context_category"],
                  question_tail=q["question"][-900:], rubrics=q.get("rubrics", []),
                  candidates=candidates(record, q, args.candidates)), ensure_ascii=False))
        return
    audit = json.loads(args.audit.read_text())
    if audit["source_sha256"] != hashlib.sha256(args.input.read_bytes()).hexdigest():
        raise ValueError("Source hash changed")
    selected = [r for r in rows if all(r["id"] + "/" + q["id"] in audit["annotations"] for q in r["questions"])]
    if args.validate_only:
        print(json.dumps(validate(selected, read_rows(args.output))))
        return
    output = build_output(selected, audit["annotations"])
    audit["validation"] = validate(selected, output)
    audit["unannotated_records"] = len(rows) - len(selected)
    save_output(args.output, output)
    atomic_json(args.audit, audit)
    print(json.dumps(audit["validation"]))


if __name__ == "__main__":
    main()
