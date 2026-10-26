"""Normalize a benchmark JSONL for downstream readers and recover list gold answers.

Two fixes applied on top of the raw normalized files:

1. Recovers list-valued gold answers that prepare_public_data.py stringified.
   LoCoMo answers were stored with str(qa["answer"]), so the JSONL carries the
   Python repr of upstream answer lists ("['7 May 2023']") instead of lists.
   Answer lists hold acceptable answer variants, so the faithful fix stores
   them as JSON arrays:
     - scripts/local_benchmark.py sees an identical str() view (unchanged);
     - scripts/e2e_benchmark.py consumes arrays natively.
   Conversion is conservative: a string is only rewritten when ast.literal_eval
   yields a scalar list whose repr round-trips exactly.

2. Line-safety re-encoding. Some upstream text contains  -class
   characters that str.splitlines() (used by local_benchmark.py) breaks into
   invalid JSON fragments. Output is re-written with ensure_ascii=True so every
   line is one JSON object for any reader.

Running the script on an already-fixed file is a no-op.

Examples:
    uv run python scripts/fix_normalized_answers.py \
      --input data/normalized/locomo_refined.jsonl \
      --output data/normalized/locomo_refined.fixed.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, object]]:
    # split('\n') not splitlines(): some upstream text contains  -class
    # characters that splitlines() would break into invalid JSON fragments.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def recover_answer(answer: object) -> tuple[object, str]:
    """Return (fixed answer, disposition) with disposition in
    {converted, already_list, kept_string, skipped_non_roundtrip}.

    Element types are preserved exactly ([2022] stays [2022]) so that
    str(answer) — the view scripts/local_benchmark.py sees — is byte-identical
    before and after the fix.
    """
    if isinstance(answer, list):
        return answer, "already_list"
    text = str(answer or "")
    if not text.lstrip().startswith("["):
        return answer, "kept_string"
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return answer, "kept_string"
    if not isinstance(parsed, list) or not parsed or not all(isinstance(value, (str, int, float)) for value in parsed):
        return answer, "kept_string"
    if str(parsed) != text:
        # Only touch values that were literally produced by str(list).
        return answer, "skipped_non_roundtrip"
    return parsed, "converted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Normalized JSONL benchmark file")
    parser.add_argument("--output", required=True, help="Destination for the fixed JSONL")
    args = parser.parse_args()
    records = read_rows(Path(args.input))
    counts = {"converted": 0, "already_list": 0, "kept_string": 0, "skipped_non_roundtrip": 0}
    questions = 0
    for record in records:
        for question in record.get("questions", []):
            questions += 1
            fixed, disposition = recover_answer(question.get("answer", ""))
            counts[disposition] += 1
            question["answer"] = fixed
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            # ensure_ascii=True escapes  -class characters, making every
            # line safe for splitlines()-based readers (e.g. local_benchmark.py).
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(
        json.dumps(
            {"records": len(records), "questions": questions, **counts, "output": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
