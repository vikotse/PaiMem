# Reproducible evaluation

[English](evaluation.md) | [简体中文](evaluation.zh-CN.md) | [README](../README.md)

Run all commands from the PaiMem repository root. Configure models and start a dedicated evaluation memory service as described in the README. Data, model weights, and generated results are not distributed with this repository.

## 1. Obtain the external evaluator and data

```bash
git clone https://github.com/AML-memory/agent-memory-leaderboard.git
# Pin the evaluator revision
git -C agent-memory-leaderboard checkout 1b8142bfe0f20f1c5218d6b554aa0012de34e504
```

Follow the [upstream instructions](https://github.com/AML-memory/agent-memory-leaderboard) for that revision to download data into `agent-memory-leaderboard/downloads/benchmarks/`. Upstream maintains download scripts and access requirements. Record dataset versions separately.

## 2. Preprocess data

LoCoMo:

```bash
uv run python scripts/prepare_public_data.py locomo \
  --input agent-memory-leaderboard/downloads/benchmarks/locomo-refined/locomo_refined.json \
  --output data/normalized/locomo_refined.jsonl
uv run python scripts/fix_normalized_answers.py \
  --input data/normalized/locomo_refined.jsonl \
  --output data/normalized/locomo_refined.fixed.jsonl
```

LongMemEval-S; substitute the corresponding filenames for the oracle variant:

```bash
uv run python scripts/prepare_public_data.py longmemeval \
  --input agent-memory-leaderboard/downloads/benchmarks/longmemeval/longmemeval_s_cleaned.json \
  --output data/normalized/longmemeval_s.jsonl
uv run python scripts/fix_normalized_answers.py \
  --input data/normalized/longmemeval_s.jsonl \
  --output data/normalized/longmemeval_s.fixed.jsonl
```

Other public datasets:

```bash
uv run python scripts/prepare_aml_benchmarks.py personamem
uv run python scripts/prepare_aml_benchmarks.py beam
uv run python scripts/prepare_aml_benchmarks.py clbench
uv run python scripts/fix_normalized_answers.py \
  --input data/normalized/clbench.jsonl --output data/normalized/clbench.fixed.jsonl
```

Reading PersonaMem parquet requires pyarrow, installed through the datasets dependency. ScriptMem cannot support meaningful memory-retrieval evaluation without the original script text.

The normalized format contains one memory unit per line:

```json
{"id":"record-1","session_id":"session-1","messages":[{"role":"user","content":"I like tea."}],"questions":[{"id":"q1","question":"What do I like?","answer":"tea","evidence_texts":["I like tea."]}]}
```

For smaller samples, select complete records and retain all their messages. Save selected IDs, input hashes, and the sampling seed. Use the same split for reproduction and record cache settings.

## 3. Retrieval recall

```bash
export MEMORY_BASE_URL='http://127.0.0.1:8000'
export MEMORY_API_KEY='same-key-as-server'
uv run python scripts/local_benchmark.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/recall/locomo-k100.jsonl \
  --base-url "$MEMORY_BASE_URL" --run-label locomo-k100-001 --top-k 100
```

Per-question JSONL is written to the output path. Summary JSON is printed to the terminal and can be redirected to a file. Use `--top-k 20`, for example, to evaluate another budget, and a distinct run-label to avoid mixing existing memories. The script does not clear the database.

- `evidence_recall_at_k`: for questions with evidence_texts, the fraction of evidence substrings covered by returned content, averaged across questions.
- `answer_in_context_rate`: a proxy measuring whether the answer string occurs in the context; it is not answer accuracy.
- Recall is null when evidence labels are absent, rather than zero. Chunk concatenation, punctuation, and speaker prefixes can affect substring matching.
- Retrieval evaluation does not generate answers or run judges, but remote embedding, reranking, and enabled LLM strategies may still incur model calls.

CLBench rubrics are not evidence labels. The optional `scripts/prepare_clbench_evidence.py` converts only source-verifiable direct annotations and produces an audit. This conversion produces evidence labels only for qualifying examples.

## 4. End-to-end evaluation

Configure answer and judge services; they may use different models:

```bash
export ANSWER_API_BASE='https://your-answer-provider.example/v1'
export ANSWER_MODEL='your-answer-model'
export ANSWER_API_KEY='your-answer-key'
export JUDGE_API_BASE='https://your-judge-provider.example/v1'
export JUDGE_MODEL='your-judge-model'
export JUDGE_API_KEY='your-judge-key'

PYTHONHASHSEED=0 uv run python scripts/run_aml_e2e.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/aml-e2e/run-001/locomo_refined \
  --base-url "$MEMORY_BASE_URL" \
  --pipeline agent-memory-leaderboard/data/locomo-refined/pipeline.py \
  --limit 3 --top-k 100 --concurrency 2 \
  --call-timeout 180 --item-timeout 600 --retries 1 \
  --save-search-results
```

Start with `--limit 3` to check the workflow. Remove limit and use a new output directory for a full run. The sequence is Add → Search → upstream answer → upstream judge or rule-based scoring. `--dry-run` checks input and configuration but still requires the pipeline file.

| Normalized data | Upstream pipeline, relative to agent-memory-leaderboard |
|---|---|
| locomo_refined | `data/locomo-refined/pipeline.py` |
| longmemeval_s / longmemeval_oracle | `data/longmemeval-s/pipeline.py` |
| personamem_v2 | `data/personamem/pipeline_v2.py` |
| beam_100k / beam_1m | `data/beam/pipeline.py` |
| clbench | `data/clbench/pipeline.py` |

`--concurrency` supplies the default for both replay and model stages. Override them individually with `--replay-concurrency` and `--model-concurrency`. Saved reports, answers, and judge outputs support tracing results.

Resume with the same configuration and output directory. Use a new directory when changing data, models, strategies, top_k, or limit. **Do not clear memory before resuming**: completed Add checkpoints would skip ingestion. If cleanup is needed, explicitly run `uv run python scripts/clear_memory.py --base-url "$MEMORY_BASE_URL"` against a dedicated test instance, then start a new run.

Supply provider prices through `--answer-price-in/out` and `--judge-price-in/out`, per million tokens. Complete costs cannot be calculated when usage or prices are missing.

## 5. Summaries and diagnostics

```bash
uv run python scripts/summarize_aml_e2e.py summarize artifacts/aml-e2e/run-001
uv run python scripts/summarize_aml_e2e.py serve artifacts/aml-e2e --port 8765
```

Open `http://127.0.0.1:8765` in a browser. For full retrieval-channel diagnostics, add `--save-retrieval-diagnostics` when the local service shares the same filesystem. For ordinary remote deployments, use only `--save-search-results`.

For `--save-retrieval-diagnostics`, create `<output>/search_diagnostics` and set the service's `[service].diagnostics_dir` to its absolute path before starting PaiMem. The runner and service must share this directory; the runner sends its service key with requests. Existing diagnostic JSON files are replaced during retries.

### Optional question categories

The report page supports English and Chinese. Version selections survive report refreshes while those versions remain available. Initial comparison defaults use the first and last versions in the displayed list, without naming requirements.

Category statistics are optional. Place labels in `data/splits/question_categories.<suffix>.json` under the PaiMem repository. For example, `question_categories.demo.json` applies to report directories named `baseline_demo` and `experiment_demo`. Matching requires the version name to end with `_<suffix>`; when several files match, the longest suffix takes precedence. Category files are independent of the NEAR `--data-dir` option.

```json
{
  "questions": [
    {
      "source_dataset": "locomo_refined",
      "item_id": "record-1:q1",
      "type": "temporal",
      "subtype": "relative-time"
    }
  ]
}
```

`source_dataset` must match the dataset output directory name, and `item_id` must match the saved evaluation item ID (for normalized input, typically `<record_id>:<question_id>`). `type` is required; `subtype` is optional. Each `(source_dataset, item_id)` pair must be unique. The example uses `temporal` as a user-defined label, not a required taxonomy.

Without a matching file, category statistics and category filters are omitted; overall, dataset, and case views remain available. Invalid matched files produce warnings. Unmapped items appear as `未分类` (uncategorized); conflicting labels across compared versions appear as `分类不一致` (inconsistent categories). Category counts include mapped items without results, but score averages use only valid scored items. Refresh the page after changing label files.

## 6. Reproducibility records

Save the PaiMem commit, configuration files and non-sensitive overrides, model revisions, hardware, Python/dependency versions, dataset hashes, complete command, upstream pipeline commit, and run outputs. Do not record API keys. LLM answers and judges can vary; identical configuration does not guarantee identical scores on every run.

PaiMem provides the workflow from preprocessing to result aggregation. Report scores, rankings, or improvements together with reproducible model settings, data splits, and run records.

## 7. Local tests

```bash
uv sync --group dev
uv run python -m pytest tests -q
```

Tests use temporary databases, fixed vectors, and mocked HTTP. No model downloads or paid-service keys are needed.

AML integration tests require the external evaluator described above and skip when it is absent. Alternatively, export `AML_PIPELINE_ROOT=/absolute/agent-memory-leaderboard/data` to use an existing checkout at the same revision.

## 8. NEAR ablation

`scripts/run_near_ablation.py` compares Hybrid with Hybrid + NEAR using its own fixed `BASE_SETTINGS`, independently of the service TOML profiles. It loads a local embedding model and SQLite stores directly; no running memory service is needed. Prepare the local model first using the local embedding setup. The commands below assume `models/bge-small-en-v1.5` is already available; `--device cuda:0` selects a compatible GPU instead of CPU.

### Prepare development and validation data

Use the normalized, answer-corrected LoCoMo-refined and LongMemEval-S files from section 2. Each JSONL row contains a complete record with `id`, `messages`, and `questions`; questions contain `id`, `question`, `answer`, and evidence labels in `evidence_texts`. Development selection requires questions with usable evidence labels in both datasets. Split complete records into disjoint development and validation sets; if a session spans multiple records, group those records together to prevent overlap.

The input directory must contain:

```text
locomo_refined.dev.jsonl
longmemeval_s.dev.jsonl
locomo_refined.validation.jsonl
longmemeval_s.validation.jsonl
```

For datasets with one independent memory record per row, this example assigns approximately 20% of records to development and the rest to validation. It is an example split, not an official benchmark split; record the selected IDs and file hashes. Use the same files when comparing runs.

```bash
uv run python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

source = Path("data/normalized")
target = Path("data/near-splits")
target.mkdir(parents=True, exist_ok=True)
for dataset in ("locomo_refined", "longmemeval_s"):
    rows = [json.loads(line) for line in
            (source / f"{dataset}.fixed.jsonl").read_text().split("\n") if line.strip()]
    ids = [str(row["id"]) for row in rows]
    assert len(rows) >= 2 and len(set(ids)) == len(ids), "Need at least two unique records"
    rows.sort(key=lambda row: hashlib.sha256(
        ("near-20261015:" + str(row["id"])).encode()).hexdigest())
    boundary = min(len(rows) - 1, max(1, len(rows) // 5))
    for phase, subset in (("dev", rows[:boundary]), ("validation", rows[boundary:])):
        (target / f"{dataset}.{phase}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in subset))
PYTHON
```

### Run retrieval and end-to-end evaluation

```bash
uv run python scripts/run_near_ablation.py --phase dev \
  --data-dir data/near-splits --output artifacts/near/dev \
  --embedding-model models/bge-small-en-v1.5 --device cpu

uv run python scripts/run_near_ablation.py --phase validation \
  --data-dir data/near-splits --output artifacts/near/validation \
  --selection artifacts/near/dev/selection.json \
  --embedding-model models/bge-small-en-v1.5 --device cpu
```

`--data-dir` defaults to the repository's `data/splits`; an explicit relative path is resolved from the current working directory. Each retrieval phase checks that both dataset files exist before creating outputs or loading the model. The same resolved inputs are used for ingestion and manifest hashes.

Development uses deterministic sampling (up to 3 LoCoMo records, 8 questions per record, and 16 LongMemEval records by default) to compare six single-layer NEAR settings. It writes `selection.json`; validation freezes that selection and evaluates all records and questions in the validation files. Keep the embedding model identical across phases. Retrieval timings exclude query encoding in both arms; token-budget recall uses the local embedding tokenizer. These timings are not end-to-end service latency.

Outputs include `manifest.json`, `retrieval-summary.json`, and per-dataset selected records and retrieval results. Validation also writes `official-input.jsonl` for each arm. After configuring `ANSWER_*` and `JUDGE_*` as in section 4 and obtaining the pinned AML checkout at `agent-memory-leaderboard`, run:

```bash
uv run python scripts/run_near_ablation.py --phase e2e \
  --output artifacts/near/validation
```

The `e2e` phase consumes the saved validation inputs and does not read `--data-dir`; it invokes answer and judge services and may incur costs. Use a new output directory after changing inputs, models, or implementation. The fixed sampling seed and algorithm settings remain unchanged.
