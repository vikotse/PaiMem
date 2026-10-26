"""Data adapters for the public AML pipelines (no execution/retry logic).

The retrieved-context mapping is a local evaluation convention, not a claim
about the leaderboard's private memory adapter. Prompts/scoring stay official.
"""
from __future__ import annotations
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
try:
    from .local_benchmark import normalize
except ImportError:
    from local_benchmark import normalize


CONTRACT_BINARY = "binary_judge"

CONTRACT_BEAM = "beam_rubric"

CONTRACT_CLBENCH = "clbench_rubric"

CONTRACT_PERSONA_V2 = "personamem_v2"

CONTRACT_PERSONA_V1 = "personamem_v1"

CONTRACT_SCRIPTMEM = "scriptmem_rule"

CONTRACTS = {
    CONTRACT_BINARY: "LoCoMo-Refined / LongMemEval — CORRECT/WRONG LLM judge (accuracy)",
    CONTRACT_BEAM: "BEAM — rubric batch LLM judge, 0/0.5/1 per rubric + event_ordering alignment",
    CONTRACT_CLBENCH: "CL-Bench — all-or-nothing rubric LLM judge + requirement ratio",
    CONTRACT_PERSONA_V2: "PersonaMem v2 — MCQ letter match (rule, no judge API); "
    "generative mode needs a preference field for the narrow judge",
    CONTRACT_PERSONA_V1: "PersonaMem v1 — option exact match (rule, no judge API)",
    CONTRACT_SCRIPTMEM: "ScriptMem — official exact option-letter scorer (rule, no judge API)",
}

LLM_JUDGE_CONTRACTS = {CONTRACT_BINARY, CONTRACT_BEAM, CONTRACT_CLBENCH}

QUESTION_RESERVED_KEYS = {"id", "question", "answer", "evidence_texts"}

def read_rows(path: Path) -> list[dict[str, object]]:
    # split('\n') not splitlines(): see fix_normalized_answers.py for why
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]

def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    # ensure_ascii keeps every record on one physical line for readers that use
    # splitlines() (the official pipelines do)
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )

def load_pipeline(path: Path) -> object:
    """Import the public AML pipeline module so its contracts are reused verbatim."""
    spec = importlib.util.spec_from_file_location("aml_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def detect_contract(pipeline: object) -> str:
    """Identify which official evaluation contract the pipeline module implements."""

    def has(*names: str) -> bool:
        return all(hasattr(pipeline, name) for name in names)

    if has("render_answer_prompt", "complete", "render_accuracy_prompt", "parse_judge_label"):
        return CONTRACT_BINARY
    if has("render_answer_prompt", "call_model", "render_batch_judge_prompt", "parse_rubric_scores", "rubric_items"):
        return CONTRACT_BEAM
    if has("build_answer_prompt", "post_chat", "evaluate_rubric_clbench", "official_rubrics"):
        return CONTRACT_CLBENCH
    if has("official_mcq_messages", "official_generative_messages", "extract_final_letter"):
        return CONTRACT_PERSONA_V2
    if has("official_messages", "official_extract_answer"):
        return CONTRACT_PERSONA_V1
    if has("render_answer_prompt", "post_chat", "predicted_letters", "score_item", "gold_letters"):
        return CONTRACT_SCRIPTMEM
    raise SystemExit(
        "unsupported pipeline: no known AML evaluation contract matches this module.\n"
        "Supported contracts:\n"
        + "\n".join(f"  - {name}: {desc}" for name, desc in CONTRACTS.items())
    )

def parse_literal(value: object) -> object:
    """Normalize str(list) artifacts the JSONL carries ("['A. x', 'B. y']") to lists."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text.startswith("["):
        return value
    try:
        return json.loads(text)
    except ValueError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value

class _AsyncFriendlyFile:
    """File wrapper that also satisfies `async with`.

    The public pipelines open their output files as the second manager of an
    `async with` (`async with httpx.AsyncClient(...) as client, output.open(...)
    as out:`), which raises TypeError on the plain TextIOWrapper (no
    __aenter__). Wrapping keeps the vendored pipeline files byte-identical
    while letting their official loops run unchanged."""

    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *exc_info):
        self._stream.close()
        return False

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc_info):
        self._stream.close()
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)

class _AsyncFriendlyPath(type(Path())):
    def open(self, *args, **kwargs):
        return _AsyncFriendlyFile(super().open(*args, **kwargs))

def install_path_shim(pipeline: object) -> None:
    """Replace the pipeline module's Path with the async-friendly subclass.
    Idempotent; everything except open() behaves exactly like pathlib.Path."""
    if getattr(pipeline, "Path", None) is not _AsyncFriendlyPath:
        pipeline.Path = lambda *args, **kwargs: _AsyncFriendlyPath(*args, **kwargs)

def evidence_recall(context: str, evidence: list[str]) -> float | None:
    if not evidence:
        return None
    normalized_context = normalize(context)
    return sum(normalize(value) in normalized_context for value in evidence) / len(evidence)

def recover_gold(answer: object) -> str | list[str]:
    """Undo prepare_public_data's str() on list-valued gold answers.

    AML's memory_text renders list gold as newline-joined items, which is what
    the private adapters feed the judge; the normalized JSONL carries the Python
    repr instead ("['7 May 2023']"). Recover the list when possible.
    """
    text = str(answer or "")
    if text.lstrip().startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list) and parsed and all(isinstance(value, (str, int, float)) for value in parsed):
            return [str(value) for value in parsed]
    return text

def memory_block(context: str) -> str:
    return f"<memories>\n{context}\n</memories>" if context.strip() else "(no memories)"

def build_item(
    question: dict[str, object],
    record_id: str,
    question_id: str,
    context: str,
    retrieved: list[dict[str, object]],
    contract: str,
) -> dict[str, object]:
    """Bridge row carrying both the shared diagnostics and the fields each
    contract's official answer()/evaluate() reads."""
    gold = recover_gold(question.get("answer", ""))
    evidence = [str(value) for value in question.get("evidence_texts", []) if str(value)]
    if isinstance(gold, list):
        in_context = any(normalize(value) in normalize(context) for value in gold) if gold else None
    elif gold:
        in_context = normalize(gold) in normalize(context)
    else:
        in_context = None
    item: dict[str, object] = {
        "id": question_id,
        "question": question["question"],
        "gold_answer": gold,
        # AML binary/beam/scriptmem renderers read the speaker fields; x_* keys
        # are local diagnostics that the public pipelines ignore.
        "speaker_1_name": "speaker 1",
        "speaker_1_memories": context,
        "speaker_2_name": "speaker 2",
        "speaker_2_memories": "",
        "retrieved_context": context,
        "x_record_id": record_id,
        "x_question_id": question.get("id"),
        "x_n_retrieved": len(retrieved),
        "x_evidence_recall_at_k": evidence_recall(context, evidence),
        "x_answer_in_context": in_context,
    }
    for key, value in question.items():
        if key in QUESTION_RESERVED_KEYS or key in item:
            continue
        item[key] = parse_literal(value) if isinstance(value, str) else value

    if contract == CONTRACT_CLBENCH:
        item['idx'] = question_id
        # official build_answer_prompt reads item["retrieval"]["selected"]
        item["retrieval"] = {
            "selected": [
                {"created_at": str(row.get("created_at") or ""), "text": str(row.get("content") or "")}
                for row in retrieved
            ]
        }
    elif contract == CONTRACT_PERSONA_V2:
        item["chat_history"] = [{"role": "user", "content": memory_block(context)}]
        item["user_query"] = str(question["question"])
        item["correct_answer"] = str(question.get("answer", ""))
        options = [str(option) for option in (question.get("options") or []) if str(option) != item["correct_answer"]]
        item["incorrect_answers"] = options
    elif contract == CONTRACT_PERSONA_V1:
        item["context_messages"] = [{"role": "user", "content": memory_block(context)}]
        options = [str(option) for option in (question.get("options") or [])]
        item["all_options"] = "\n".join(f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options))
        item["correct_answer"] = str(question.get("answer", ""))
    return item

def scriptmem_gold_map(pipeline: object, data_dir: Path) -> dict[tuple[str, int], str]:
    """Map (dataset source, qa index) -> official qa_id so bridge ids line up
    with load_gold_records() at scoring time."""
    mapping: dict[tuple[str, int], str] = {}
    for record in pipeline.load_gold_records(data_dir):
        mapping[(str(record["dataset"]), int(record["qa_index"]))] = str(record["qa_id"])
    return mapping

def validate_items(contract: str, items: list[dict[str, object]], args: argparse.Namespace) -> None:
    """Fail fast when the bridge file lacks the fields the official code needs."""
    if not items:
        raise SystemExit("replay produced no items")
    if contract == CONTRACT_BEAM:
        missing = [item["id"] for item in items if not isinstance(parse_literal(item.get("rubric")), list)]
        if missing:
            raise SystemExit(
                f"{len(missing)}/{len(items)} beam items have no rubric list (first: {missing[0]}). "
                "Regenerate the data: uv run python scripts/prepare_aml_benchmarks.py beam "
                "&& uv run python scripts/make_splits.py --dataset beam_100k "
                "&& uv run python scripts/make_splits.py --dataset beam_1m, then re-run replay."
            )
    elif contract == CONTRACT_CLBENCH:
        if not all(isinstance(item.get("retrieval"), dict) for item in items):
            raise SystemExit(
                "clbench items lack the retrieval block; re-run replay without --skip-replay "
                "(the .input.jsonl predates contract-aware bridging)."
            )
        rubric_rows = pipeline_rubrics_missing(items)
        if rubric_rows:
            raise SystemExit(f"{len(rubric_rows)}/{len(items)} clbench items have no rubrics (first: {rubric_rows[0]}).")
    elif contract == CONTRACT_PERSONA_V2:
        if args.personamem_mode == "generative":
            raise SystemExit(
                "personamem generative mode needs a `preference` field for the official narrow judge; "
                "the local normalized data does not carry one. Use --personamem-mode mcq."
            )
        bad = [item["id"] for item in items if len(item.get("incorrect_answers") or []) < 1]
        if bad:
            raise SystemExit(f"{len(bad)}/{len(items)} personamem items lack usable options (first: {bad[0]}).")
    elif contract == CONTRACT_PERSONA_V1:
        bad = [item["id"] for item in items if not item.get("all_options")]
        if bad:
            raise SystemExit(f"{len(bad)}/{len(items)} personamem v1 items lack options (first: {bad[0]}).")
    elif contract == CONTRACT_BINARY:
        missing = [item["id"] for item in items if item.get("gold_answer") in ("", None)]
        if missing:
            print(f"[warn] {len(missing)}/{len(items)} items have an empty gold answer", file=sys.stderr)

def pipeline_rubrics_missing(items: list[dict[str, object]]) -> list[object]:
    missing = []
    for item in items:
        rubrics = item.get("rubrics")
        if rubrics is None and isinstance(item.get("metadata"), dict):
            rubrics = item["metadata"].get("rubrics")
        if not isinstance(parse_literal(rubrics), list):
            missing.append(item["id"])
    return missing

def official_ns(**overrides) -> SimpleNamespace:
    """argparse.Namespace superset covering every field the official
    answer()/evaluate() entry points may read (extra fields are ignored)."""
    values = dict(
        input="",
        output="",
        answers="",
        model="",
        base_url="",
        api_key="",
        api_key_env="API_KEY",
        max_tokens=None,
        judge_max_tokens=None,
        temperature=None,
        timeout=180.0,
        mode="mcq",
        max_retries=3,
        data_dir="",
        submission="",
        details="",
        command="answer",
    )
    values.update(overrides)
    return SimpleNamespace(**values)

def row_id_of(row: dict[str, object], contract: str) -> str:
    for key in (("idx", "id") if contract == CONTRACT_CLBENCH else ("id", "idx", "qa_id")):
        if row.get(key) is not None:
            return str(row[key])
    return ""

def row_score(row: dict[str, object], contract: str) -> float | None:
    """Map any contract's judged row onto a 0..1 score for aggregation."""
    if contract == CONTRACT_BINARY:
        if row.get("is_correct") is None:
            return None
        return 1.0 if row.get("is_correct") else 0.0
    if contract == CONTRACT_BEAM:
        value = row.get("llm_judge_score")
        return float(value) if value is not None else None
    if contract == CONTRACT_CLBENCH:
        value = row.get("rubric_clbench_score")
        return float(value) if value is not None else None
    if contract in (CONTRACT_PERSONA_V1, CONTRACT_PERSONA_V2):
        if row.get("is_correct") is None:
            return None
        return 1.0 if row.get("is_correct") else 0.0
    if contract == CONTRACT_SCRIPTMEM:
        value = row.get("score")
        return float(value) if value is not None else None
    return None
