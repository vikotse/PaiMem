"""Paired Hybrid / Hybrid+NEAR evaluation using PaiMem and official AML pipelines.

dev: fixed hash sample, six one-layer settings, no answer/judge calls.
validation: frozen dev selection, all questions, saves official answer inputs.
e2e: calls the unchanged AML runner on those saved inputs; credentials via env.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from paimem.embeddings import build_embedder
from paimem.memory_store import MemoryMessage, SQLiteMemoryStore
from paimem.near import NearConfig
import aml_pipeline_adapters as adapters
from local_benchmark import chunk_messages, normalize


DATASETS = {"locomo_refined": "locomo-refined", "longmemeval_s": "longmemeval-s"}
BASE_SETTINGS = dict(retrieval_mode="hybrid", lexical_weight=.5, rrf_k=60,
                     option_view_weight=0, context_messages=0, result_window=1,
                     result_window_seed_k=20, pack_span=1)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_order(value):
    return hashlib.sha256(("near-20261015:" + str(value)).encode()).hexdigest()


def select_records(path, record_limit=None, question_limit=None):
    records = adapters.read_rows(path)
    if record_limit is not None:
        records = sorted(records, key=lambda r: hash_order(r["id"]))[:record_limit]
    if question_limit is not None:
        records = [{**r, "questions": sorted(r["questions"], key=lambda q: hash_order(q["id"]))[:question_limit]}
                   for r in records]
    return records


class QueryCachedEmbedder:
    """Encoding is excluded from paired search timings in BOTH arms."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.identity = wrapped.identity
        self.queries = {}

    def encode_documents(self, texts):
        return self.wrapped.encode_documents(texts)

    def prepare(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.queries))
        if missing:
            vectors = self.wrapped.encode_queries(missing)
            self.queries.update(zip(missing, vectors))

    def encode_queries(self, texts):
        self.prepare(texts)
        return np.asarray([self.queries[t] for t in texts], dtype=np.float32)


def evidence_metrics(hits, evidence, tokenizer=None):
    evidence = list(dict.fromkeys(normalize(str(v)) for v in evidence if str(v).strip()))
    result = {"returned": len(hits), "context_chars": sum(len(h.content) for h in hits),
              "unique_contents": len({h.content for h in hits}), "evidence_units": len(evidence)}
    for k in (10, 20, 100):
        context = normalize("\n".join(h.content for h in hits[:k]))
        found = sum(text in context for text in evidence)
        result[f"recall_at_{k}"] = found / len(evidence) if evidence else None
        result[f"complete_at_{k}"] = float(found == len(evidence)) if evidence else None
    # Fixed token budget uses the same local BGE tokenizer in both arms; this is
    # a retrieval proxy, not the answering model's token accounting.
    if tokenizer is not None:
        context = "\n".join(h.content for h in hits)
        tokens = tokenizer.encode(context, add_special_tokens=False, truncation=False, verbose=False)
        result["bge_context_tokens"] = len(tokens)
        for budget in (2048, 4096):
            clipped = normalize(tokenizer.decode(tokens[:budget], skip_special_tokens=True))
            # Decode evidence by the same tokenizer to avoid punctuation artifacts.
            normalized_evidence = [normalize(tokenizer.decode(tokenizer.encode(t, add_special_tokens=False),
                                                             skip_special_tokens=True)) for t in evidence]
            found = sum(t in clipped for t in normalized_evidence)
            result[f"recall_budget_{budget}"] = found / len(evidence) if evidence else None
    return result


def summarize(rows):
    keys = sorted({k for row in rows for k in row["metrics"]})
    result = {"questions": len(rows)}
    for key in keys:
        values = [r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
        result[key] = statistics.mean(values) if values else None
    times = [r["search_seconds"] for r in rows]
    warm_times = [r["warm_search_seconds"] for r in rows]
    result.update(search_median_seconds=statistics.median(times) if times else None,
                  search_p95_seconds=float(np.percentile(times, 95)) if times else None,
                  warm_search_median_seconds=statistics.median(warm_times) if warm_times else None,
                  max_bank_bytes=max((r["near"].get("bank_bytes", 0) for r in rows), default=0),
                  max_build_seconds=max((r["near"].get("build_seconds", 0) for r in rows), default=0))
    return result


def paired_rows(base, near):
    left = {row["id"]: row for row in base}
    right = {row["id"]: row for row in near}
    if left.keys() != right.keys():
        raise ValueError("Paired arms have different question IDs")
    result = {}
    for metric in ("recall_at_10", "recall_at_20", "recall_at_100", "complete_at_100", "recall_budget_4096"):
        deltas = [right[k]["metrics"][metric] - left[k]["metrics"][metric]
                  for k in left if left[k]["metrics"].get(metric) is not None]
        result[metric] = dict(mean_delta=statistics.mean(deltas) if deltas else None,
                              improved=sum(d > 1e-10 for d in deltas),
                              regressed=sum(d < -1e-10 for d in deltas), unchanged=sum(abs(d) <= 1e-10 for d in deltas))
    return result


def retrieval_inputs(data_dir: Path, phase: str) -> dict[str, Path]:
    """Validate the complete input set before creating outputs or loading models."""
    directory = data_dir.expanduser().resolve()
    paths = {dataset: directory / f"{dataset}.{phase}.jsonl" for dataset in DATASETS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Missing NEAR input files: " + ", ".join(missing)
                         + ". Set --data-dir to the prepared split directory; see docs/evaluation.md.")
    return paths


def run_retrieval(args):
    output = args.output.resolve()
    if args.phase == "dev":
        configs = {"hybrid": NearConfig()}
        configs.update({f"near_l1_{'unit' if norm else 'raw'}_a{weight:g}":
                        NearConfig(enabled=True, layers=1, normalize=norm, weight=weight)
                        for norm in (False, True) for weight in (.5, 1., 2.)})
    else:
        if args.selection is None:
            raise ValueError("validation requires --selection from dev")
        selection = json.loads(args.selection.read_text())
        if selection.get("embedding") != args.embedding_model or selection.get("base_settings") != BASE_SETTINGS:
            raise ValueError("Validation must use the embedding and Hybrid settings frozen on dev")
        configs = {"hybrid": NearConfig(), "near": NearConfig(**selection["near_config"])}
    inputs = retrieval_inputs(args.data_dir, args.phase)
    manifest = {"phase": args.phase, "embedding": args.embedding_model, "device": args.device,
                "base_settings": BASE_SETTINGS, "configs": {k: asdict(v) for k, v in configs.items()},
                "sampling": {"locomo_records": args.locomo_dev_records, "locomo_questions_per_record": args.locomo_dev_questions,
                             "longmemeval_records": args.longmemeval_dev_records, "seed": "near-20261015"},
                "input_sha256": {dataset: digest(path) for dataset, path in inputs.items()},
                "source_sha256": {name: digest(ROOT / name) for name in (
                    "src/paimem/near.py", "src/paimem/memory_store.py", "scripts/run_near_ablation.py")},
                "selection_sha256": digest(args.selection) if args.selection else None,
                "timing_note": "Search timings exclude query encoding in both arms; exact NumPy, not ANN. Warm calls omit diagnostics."}
    existing = output / "manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError("Configuration/source/input changed; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(existing, manifest)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    embedder = QueryCachedEmbedder(build_embedder(args.embedding_model, args.device))
    summaries = {}
    for dataset in DATASETS:
        path = inputs[dataset]
        limits = ((args.locomo_dev_records, args.locomo_dev_questions) if dataset == "locomo_refined"
                  else (args.longmemeval_dev_records, None)) if args.phase == "dev" else (None, None)
        records = select_records(path, *limits)
        folder = output / dataset
        folder.mkdir(exist_ok=True)
        adapters.write_rows(folder / "selected.jsonl", records)
        stores = {name: SQLiteMemoryStore(folder / "memory.db", embedder, near_config=config, **BASE_SETTINGS)
                  for name, config in configs.items()}
        rows_by_arm = {name: [] for name in configs}
        official_by_arm = {name: [] for name in configs}
        for record_number, record in enumerate(records):
            uid = f"near:{dataset}:{record['id']}"
            started = time.perf_counter()
            for index, messages in enumerate(chunk_messages(record["messages"])):
                stores["hybrid"].add(f"{uid}:{index}", uid, str(record.get("session_id", record["id"])),
                                     [MemoryMessage(str(m["role"]), str(m["content"]), m.get("timestamp")) for m in messages])
            queries = [q["question"] + ("\nOptions:\n" + "\n".join(q["options"]) if q.get("options") else "")
                       for q in record["questions"]]
            embedder.prepare(queries)
            encode_seconds = time.perf_counter() - started
            print(json.dumps({"event": "record_ready", "dataset": dataset, "record": record_number + 1,
                              "records": len(records), "add_and_query_encode_seconds": encode_seconds}), flush=True)
            for question in record["questions"]:
                ident = f"{record['id']}:{question['id']}"
                for name, store in stores.items():
                    started = time.perf_counter()
                    hits, trace = store.search(uid, question["question"], 100, question.get("options"), diagnostics=True)
                    elapsed = time.perf_counter() - started
                    started = time.perf_counter()
                    warm_hits = store.search(uid, question["question"], 100, question.get("options"))
                    warm_elapsed = time.perf_counter() - started
                    if hits != warm_hits:
                        raise AssertionError("Warm and diagnostic retrieval differ")
                    entry = dict(id=ident, record_id=str(record["id"]), question=question["question"],
                                 metrics=evidence_metrics(hits, question.get("evidence_texts", []), embedder.wrapped.model.tokenizer),
                                 search_seconds=elapsed, warm_search_seconds=warm_elapsed,
                                 near=trace["near"], retrieved=[asdict(hit) for hit in hits])
                    rows_by_arm[name].append(entry)
                    if args.phase == "validation":
                        key = hashlib.sha256(ident.encode()).hexdigest()
                        write_json(folder / name / "diagnostics" / f"{key}.json", trace)
                        context = "\n".join(h.content for h in hits)
                        official_by_arm[name].append(adapters.build_item(question, str(record["id"]), ident,
                                                                        context, entry["retrieved"], adapters.CONTRACT_BINARY))
            # Per-record checkpoints allow inspecting progress without incomplete summaries.
            for name, rows in rows_by_arm.items():
                target = folder / name
                target.mkdir(exist_ok=True)
                adapters.write_rows(target / "retrieval.jsonl", rows)
        summaries[dataset] = {name: summarize(rows) for name, rows in rows_by_arm.items()}
        summaries[dataset]["paired"] = {name: paired_rows(rows_by_arm["hybrid"], rows)
                                         for name, rows in rows_by_arm.items() if name != "hybrid"}
        for name, items in official_by_arm.items():
            if items:
                adapters.write_rows(folder / name / "official-input.jsonl", items)
        write_json(output / "retrieval-summary.json", summaries)
    if args.phase == "dev":
        # Equal dataset weights; primary is recall@20, then recall@100, then @10.
        # This objective is fixed before validation and does not use answer labels.
        candidates = [name for name in configs if name != "hybrid"]
        def objective(name):
            return tuple(statistics.mean(summaries[d][name][metric] for d in DATASETS)
                         for metric in ("recall_at_20", "recall_at_100", "recall_at_10"))
        best = max(candidates, key=objective)
        write_json(output / "selection.json", {
            "selected": best, "near_config": asdict(configs[best]),
            "selection_rule": "Macro dataset recall@20; ties recall@100 then recall@10; only dev",
            "selected_objective": objective(best), "baseline_objective": objective("hybrid"),
            "positive_dev_net_gain": objective(best) > objective("hybrid"),
            "limitation": "Small deterministic dev subset; six one-layer settings, not the full paper grid.",
            "embedding": args.embedding_model, "base_settings": BASE_SETTINGS})
    print(json.dumps({"event": "retrieval_complete", "output": str(output)}), flush=True)


def run_e2e(args):
    """Delegate official generation/judging, preserving runner accounting/resume."""
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("phase") != "validation":
        raise ValueError("e2e requires a validation manifest")
    if not (output / "retrieval-summary.json").exists():
        raise ValueError("e2e requires a completed validation retrieval directory")
    for stage in ("ANSWER", "JUDGE"):
        if not all(os.environ.get(f"{stage}_{field}") for field in ("API_BASE", "API_KEY", "MODEL")):
            raise ValueError(f"Configure {stage}_API_BASE / API_KEY / MODEL in the environment")
    reports = {}
    for dataset, pipeline_dir in DATASETS.items():
        reports[dataset] = {}
        for arm in ("hybrid", "near"):
            folder = output / dataset / arm
            command = [sys.executable, str(ROOT / "scripts/run_aml_e2e.py"),
                       "--input", str(folder / "official-input.jsonl"), "--input-format", "official",
                       "--output", str(folder / "e2e"), "--pipeline",
                       str(ROOT / "agent-memory-leaderboard/data" / pipeline_dir / "pipeline.py"),
                       "--concurrency", "4", "--call-timeout", "120", "--item-timeout", "300",
                       "--retries", "1", "--heartbeat", "20"]
            with (folder / "e2e.log").open("a") as log:
                result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                        stdout=log, stderr=log)
            report_path = folder / "e2e/report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {"complete": False}
            reports[dataset][arm] = {"returncode": result.returncode, **report}
            print(json.dumps({"event": "e2e_arm_complete", "dataset": dataset, "arm": arm,
                              "complete": report.get("complete"), "score": report.get("score_mean")}), flush=True)
            write_json(output / "e2e-summary.json", reports)
            if result.returncode:
                raise RuntimeError(f"{dataset}/{arm} incomplete; see its e2e.log")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "validation", "e2e"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/splits",
                        help="Normalized dev/validation JSONL directory (default: repository data/splits); unused for e2e")
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--locomo-dev-records", type=int, default=3)
    parser.add_argument("--locomo-dev-questions", type=int, default=8)
    parser.add_argument("--longmemeval-dev-records", type=int, default=16)
    args = parser.parse_args(argv)
    if min(args.locomo_dev_records, args.locomo_dev_questions, args.longmemeval_dev_records) <= 0:
        parser.error("dev sample sizes must be positive")
    try:
        if args.phase == "e2e":
            run_e2e(args)
        else:
            run_retrieval(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
