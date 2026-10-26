r"""Measure the paimem reranker, with one fresh process per configuration.

Examples (run from the repository root):
  python scripts/benchmark_rerankers.py \
    --models /models/bge-reranker-large /models/MemReranker-4B \
    --device cuda:2 --batch-sizes 1 4 8 16 32 --max-lengths 512 1024 \
    --candidates 200 --warmup 2 --repeats 10 --output /tmp/reranker-benchmark

  CUDA_VISIBLE_DEVICES=2 python scripts/benchmark_rerankers.py \
    --models /models/MemReranker-4B --device cuda:0 \
    --input /tmp/rerank-input.jsonl --queries 3 --output /tmp/real-input

  python scripts/benchmark_rerankers.py --backend vllm \
    --models bge MemReranker-4B \
    --base-urls http://127.0.0.1:8088 http://127.0.0.1:8089 \
    --gpu-ids 4 5 --input /tmp/rerank-input.jsonl \
    --queries 3 --candidates 200 --warmup 1 --repeats 10 \
    --output /tmp/vllm-comparison

  CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_rerankers.py \
    --backends cross-encoder vllm --models /models/bge MemReranker-4B \
    --base-urls - http://127.0.0.1:8089 --device cuda:0 --gpu-ids 5 5 \
    --batch-sizes 4 16 --max-lengths 512 --pause-before-model \
    --input /tmp/rerank-input.jsonl --queries 1 --candidates 200 \
    --warmup 1 --repeats 3 --output /tmp/mixed-comparison

vLLM mode uses existing servers' /score endpoints, sequentially, without loading
models. Models, URLs and optional GPU groups correspond by position. GPU groups
are physical nvidia-smi indices/UUIDs (e.g. --gpu-ids 4 5 or 4,5 6,7);
use "-" to skip a server. Monitoring is only allowed for loopback URLs: run this
script on the server machine, or omit --gpu-ids for remote latency-only tests.
Memory is sampled whole-card usage, including other resident processes, NOT
PyTorch allocated/reserved. Sampling can miss peaks. Shared GPUs are flagged.
--pause-before-model allows manually replacing a server between model tests.
The script never starts/stops servers. Server load time is unavailable.
HTTP latency includes serialization, network, server queueing and preprocessing.
No HTTP client truncation is performed: choose inputs fitting BOTH models'
complete query/document/template limits. --max-lengths controls local models only.
--batch-sizes controls local inference batches AND sequential HTTP request sizes.
Without --batch-sizes, all-vLLM runs send one request per query as before.
For mixed runs, use --backends cross-encoder vllm, --base-urls - URL,
and --gpu-ids 5 5. Mixed runs accept a single local --max-lengths value (default
512); this does NOT set the remote length limit. Comparisons use identical raw
texts/batch sizes, not guaranteed identical token counts. --pause-before-model
allows stopping remote services before local tests and starting them afterwards.
Without input, vLLM defaults to 128 words in the longest synthetic document;
this is not token-length matching. Use the same JSONL for cross-backend tests.

JSONL input: one {"query": "...", "documents": ["...", "..."]} per line.
The first --queries records are used, each with at least max(--candidates)
documents. Smaller candidate budgets use a prefix of the same documents.

Without --input, fixed English memory-like synthetic texts are used. They test
performance, not retrieval quality. All models receive identical texts; token
counts differ by tokenizer. By default the longest synthetic documents contain
2 * max(--max-lengths) words, so the matrix actually exercises truncation.

Each measured call scores one query's entire candidate pool, including
tokenization, batching and score transfer to CPU. Loading, token diagnostics and
warmup are excluded. GPU work is synchronized before and after each timed call.
CUDA figures are PyTorch allocated/reserved bytes, NOT nvidia-smi process usage
or whole-card usage. Peaks include weights and are measured after warmup, without
emptying the allocator cache between calls. CPU/MPS do not report CUDA metrics.

Models must already be local or cached unless --allow-download is supplied.
Loading time includes any permitted download and filesystem-cache effects; it
is not a cold-disk benchmark. Local mode uses the PaiMem loader (FP16 on
CUDA) and its device placement; this script does not add quantization.
--dry-run validates the input/matrix without loading models or writing reports.
--output is a filename prefix for .json and .csv; existing reports require
--overwrite. Partial reports are saved after each configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone


GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]


def http_json(base_url: str, path: str, timeout: float, payload=None, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {exc.code} on {path}: {detail}") from exc


def validate_scores(response, count: int) -> list[float]:
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or len(data) != count:
        raise ValueError(f"/score must return exactly {count} data items")
    scores = [None] * count
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("invalid /score item")
        index, score = item.get("index"), item.get("score")
        if (type(index) is not int or not 0 <= index < count or scores[index] is not None
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError("/score returned invalid/duplicate indices or non-finite scores")
        scores[index] = float(score)
    return scores


def gpu_snapshot(gpu_ids: list[str]) -> list[dict]:
    completed = subprocess.run(
        ["nvidia-smi", "--id=" + ",".join(gpu_ids),
         "--query-gpu=index,uuid,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    rows = []
    for fields in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 5:
            raise ValueError("unexpected nvidia-smi response")
        index, uuid, name, total, used = (field.strip() for field in fields)
        total, used = float(total), float(used)
        if not all(math.isfinite(value) and value >= 0 for value in (total, used)):
            raise ValueError("nvidia-smi memory values unavailable")
        rows.append(dict(index=index, uuid=uuid, name=name,
                         total_bytes=int(total * 1024**2), used_bytes=int(used * 1024**2)))
    if len(rows) != len(gpu_ids) or len({row["uuid"] for row in rows}) != len(rows):
        raise ValueError("nvidia-smi did not return each requested GPU exactly once")
    return rows


class GpuSampler:
    """Sample whole-card usage; failures are explicit, never represented as zero."""

    def __init__(self, gpu_ids: list[str], interval: float):
        self.gpu_ids, self.interval = gpu_ids, interval
        self.samples, self.errors = [], []
        self.stop_event = threading.Event()
        self.started = time.perf_counter()
        self.thread = None

    def sample(self):
        try:
            rows = gpu_snapshot(self.gpu_ids)
            self.samples.append({
                "seconds": time.perf_counter() - self.started, "gpus": rows,
                "total_used_bytes": sum(row["used_bytes"] for row in rows),
            })
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self):
        self.sample()
        if not self.errors:
            def run():
                while not self.stop_event.wait(self.interval):
                    self.sample()
                    if self.errors:
                        break
            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
        return {
            "status": "ok" if not self.errors else "partial" if self.samples else "unavailable",
            "scope": "whole_gpu_nvidia_smi_sampled",
            "interval_seconds": self.interval, "errors": self.errors,
            "samples": self.samples,
            "sampled_peak_total_bytes": max(
                (sample["total_used_bytes"] for sample in self.samples), default=None),
        }


def vllm_worker(spec: dict) -> dict:
    result = {"status": "error", "stage": "server_check", "load_seconds": None}
    sampler = None
    try:
        api_key = os.environ.get(spec.get("api_key_env") or "")
        if spec.get("api_key_env") and not api_key:
            raise ValueError(f"API key environment variable {spec['api_key_env']} is empty")
        def request(path, payload=None):
            return http_json(spec["base_url"], path, spec["request_timeout"], payload, api_key)

        models = request("/v1/models")
        if spec["model"] not in [item["id"] for item in models["data"]]:
            raise ValueError(f"server does not advertise model {spec['model']!r}")
        result["server_model"] = next(item for item in models["data"] if item["id"] == spec["model"])
        records = json.loads(Path(spec["input_path"]).read_text(encoding="utf-8"))
        records = [dict(query=row["query"], documents=row["documents"][:spec["candidates"]])
                   for row in records]
        if spec["gpu_ids"]:
            try:
                result["idle_gpu_memory"] = gpu_snapshot(spec["gpu_ids"])
            except Exception as exc:
                result["idle_gpu_memory_error"] = f"{type(exc).__name__}: {exc}"

        def score(row):
            scores = []
            batch_size = spec["batch_size"] or len(row["documents"])
            for start in range(0, len(row["documents"]), batch_size):
                documents = row["documents"][start:start + batch_size]
                response = request("/score", {
                    "model": spec["model"], "queries": [row["query"]] * len(documents),
                    "documents": documents,
                })
                scores.extend(validate_scores(response, len(documents)))
            return scores

        result["stage"] = "warmup"
        for repeat in range(spec["warmup"]):
            for index, row in enumerate(records):
                score(row)
                print(f"  warmup {repeat + 1}/{spec['warmup']} query {index + 1}/{len(records)}",
                      flush=True)
        if spec["gpu_ids"]:
            sampler = GpuSampler(spec["gpu_ids"], spec["memory_interval"])
            sampler.start()
        result["stage"] = "measure"
        samples = []
        started_all = time.perf_counter()
        for repeat in range(spec["repeats"]):
            for index, row in enumerate(records):
                started = time.perf_counter()
                score(row)
                elapsed = time.perf_counter() - started
                samples.append(dict(repeat=repeat, query_index=index, seconds=elapsed))
                print(f"  timed {len(samples)}/{spec['repeats'] * len(records)}: "
                      f"{elapsed * 1000:.2f}ms", flush=True)
        wall_seconds = time.perf_counter() - started_all
        times = [sample["seconds"] for sample in samples]
        result.update(
            samples=samples, latency_ms=describe([seconds * 1000 for seconds in times]),
            documents_per_second=len(times) * spec["candidates"] / sum(times),
            wall_documents_per_second=len(times) * spec["candidates"] / wall_seconds,
            amortized_ms_per_document=sum(times) * 1000 / (len(times) * spec["candidates"]),
            measured_total_seconds=sum(times), measured_wall_seconds=wall_seconds,
            status="ok", stage="complete",
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sampler is not None:
            result["service_memory"] = sampler.finish()
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def percentile(values: list[float], quantile: float) -> float:
    """Linearly interpolate sorted observations (including single observations)."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(values), "mean": statistics.mean(values),
        "min": min(values), "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95), "max": max(values),
    }


def synthetic_inputs(count: int, candidates: int, longest_words: int) -> list[dict]:
    topics = [
        ("When did Alice move to Berlin?",
         "Alice moved to Berlin in March after finishing her research project in Paris."),
        ("What food does Bob prefer when travelling?",
         "Bob prefers vegetarian meals and avoids peanuts when travelling with his colleagues."),
        ("What changed in Carol's deployment plan?",
         "Carol postponed the deployment until Friday after reviewing the failed integration tests."),
    ]
    records = []
    for query_index in range(count):
        query, _ = topics[query_index % len(topics)]
        documents = []
        for index in range(candidates):
            sentence = topics[(query_index + index) % len(topics)][1]
            words = sentence.split()
            target = max(1, longest_words * (4 - index % 4) // 4)
            body = " ".join((words * ((target + len(words) - 1) // len(words)))[:target])
            documents.append(
                f"[user | 2026-01-{index % 28 + 1:02d}] "
                f"Session {query_index}, memory {index}. {body}"
            )
        records.append({"query": query, "documents": documents})
    return records


def read_inputs(path: Path, query_count: int, candidates: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
                raise ValueError(f"{path}:{line_number}: query must be a nonempty string")
            documents = row.get("documents")
            if (not isinstance(documents, list) or len(documents) < candidates
                    or any(not isinstance(doc, str) or not doc.strip() for doc in documents)):
                raise ValueError(
                    f"{path}:{line_number}: documents must contain at least "
                    f"{candidates} nonempty strings"
                )
            records.append({"query": row["query"], "documents": documents[:candidates]})
            if len(records) == query_count:
                break
    if len(records) != query_count:
        raise ValueError(f"{path}: requested {query_count} queries, found {len(records)}")
    return records


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(torch, device) -> dict | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def reset_peaks(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def token_diagnostics(model, records: list[dict], batch_size: int) -> dict:
    """Use the model's inference preprocessing, including its default prompt."""
    lengths = []
    modern = callable(getattr(model, "preprocess", None))
    prompt = model._resolve_prompt(None, None) if modern else None
    for row in records:
        docs = row["documents"]
        for start in range(0, len(docs), min(batch_size, 16)):
            batch = docs[start:start + min(batch_size, 16)]
            if modern:
                features = model.preprocess([(row["query"], doc) for doc in batch], prompt=prompt)
            else:
                # Legacy CrossEncoder tokenizes plain query/document pairs.
                features = model.tokenizer(
                    [row["query"]] * len(batch), batch, padding=True,
                    truncation="longest_first", max_length=model.max_length,
                    return_tensors="pt",
                )
            mask = features.get("attention_mask")
            if mask is None:
                raise ValueError("model preprocessing returned no attention_mask")
            lengths.extend(int(value) for value in mask.sum(dim=-1).tolist())
    return {
        "effective_pair_tokens": describe(lengths),
        "pairs_at_length_limit": sum(length >= model.max_length for length in lengths),
        "effective_max_length": model.max_length,
        "method": "CrossEncoder.preprocess with default prompt" if modern else "legacy pair tokenizer",
        "note": "Non-padding tokens after truncation, including prompt/special tokens. "
                "At-limit count is not an exact truncation count.",
    }


def worker(spec_path: str, result_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("backend") == "vllm":
        result = vllm_worker(spec)
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return 0 if result["status"] == "ok" else 1
    result = {"status": "error", "stage": "imports"}
    torch = device = None
    sampler = None
    try:
        if not spec["allow_download"]:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import torch
        from paimem.reranking import build_reranker

        # The PaiMem factory imports this lazily; keep that import outside load timing.
        importlib.import_module("sentence_transformers")
        result["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sentence-transformers", "numpy")
        }
        result["stage"] = "device"
        device = torch.device(spec["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            result["gpu"] = {
                "name": properties.name, "total_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(device)),
                "cuda_runtime": torch.version.cuda,
            }
        elif device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; no CPU fallback")
        elif device.type not in {"cpu", "mps"}:
            raise ValueError("device must be cpu, mps, cuda or cuda:N")
        result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        synchronize(torch, device)
        result["before_load_memory"] = memory_snapshot(torch, device)
        reset_peaks(torch, device)
        result["stage"] = "load"
        started = time.perf_counter()
        reranker = build_reranker(
            spec["model"], device=str(device), batch_size=spec["batch_size"],
            max_length=spec["max_length"],
        )
        synchronize(torch, device)
        result["load_seconds"] = time.perf_counter() - started
        result["loaded_memory"] = memory_snapshot(torch, device)
        result["parameter_dtype"] = str(next(reranker.model.parameters()).dtype)
        result["actual_device"] = str(reranker.model.device)
        result["model_commit"] = getattr(reranker.model.config, "_commit_hash", None)
        if spec.get("gpu_ids"):
            try:
                result["idle_gpu_memory"] = gpu_snapshot(spec["gpu_ids"])
            except Exception as exc:
                result["idle_gpu_memory_error"] = f"{type(exc).__name__}: {exc}"
        records = json.loads(Path(spec["input_path"]).read_text(encoding="utf-8"))
        records = [
            {"query": row["query"], "documents": row["documents"][:spec["candidates"]]}
            for row in records
        ]
        result["stage"] = "token_diagnostics"
        result["tokens"] = token_diagnostics(reranker.model, records, spec["batch_size"])

        def score(row):
            values = reranker.score(row["query"], row["documents"])
            return np.asarray(values)

        def check(values, row):
            if values.shape != (len(row["documents"]),) or not np.isfinite(values).all():
                raise ValueError("reranker returned an incorrect score shape or non-finite scores")

        result["stage"] = "warmup"
        for repeat in range(spec["warmup"]):
            for index, row in enumerate(records):
                values = score(row)
                check(values, row)
                print(f"  warmup {repeat + 1}/{spec['warmup']} query {index + 1}/{len(records)}",
                      flush=True)
        synchronize(torch, device)
        result["warmed_memory"] = memory_snapshot(torch, device)
        reset_peaks(torch, device)
        if spec.get("gpu_ids"):
            sampler = GpuSampler(spec["gpu_ids"], spec["memory_interval"])
            sampler.start()
        result["stage"] = "measure"
        samples = []
        started_all = time.perf_counter()
        for repeat in range(spec["repeats"]):
            for query_index, row in enumerate(records):
                synchronize(torch, device)
                started = time.perf_counter()
                values = score(row)
                synchronize(torch, device)
                elapsed = time.perf_counter() - started
                check(values, row)
                samples.append({"repeat": repeat, "query_index": query_index, "seconds": elapsed})
                print(f"  timed {len(samples)}/{spec['repeats'] * len(records)}: "
                      f"{elapsed * 1000:.2f}ms", flush=True)
        wall_seconds = time.perf_counter() - started_all
        result["inference_memory"] = memory_snapshot(torch, device)
        result["samples"] = samples
        times = [sample["seconds"] for sample in samples]
        result["latency_ms"] = describe([seconds * 1000 for seconds in times])
        result["documents_per_second"] = len(times) * spec["candidates"] / sum(times)
        result["wall_documents_per_second"] = len(times) * spec["candidates"] / wall_seconds
        result["measured_wall_seconds"] = wall_seconds
        result["amortized_ms_per_document"] = sum(times) * 1000 / (len(times) * spec["candidates"])
        result["measured_total_seconds"] = sum(times)
        result["status"], result["stage"] = "ok", "complete"
    except Exception as exc:
        # CUDA OutOfMemoryError is a RuntimeError subclass; MPS/host OOM errors
        # need text matching. A SIGKILL without a result is NOT assumed to be OOM.
        oom_type = getattr(torch, "OutOfMemoryError", ()) if torch is not None else ()
        result["status"] = "oom" if isinstance(exc, oom_type) or "out of memory" in str(exc).lower() else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        if torch is not None and device is not None:
            try:
                result["failure_memory"] = memory_snapshot(torch, device)
            except Exception:
                pass
    finally:
        if sampler is not None:
            result["service_memory"] = sampler.finish()
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


def execute_case(spec: dict, directory: Path, timeout: float) -> dict:
    spec_path, result_path, log_path = (directory / name for name in ("spec.json", "result.json", "worker.log"))
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    # Every case gets its own directory; never read a previous worker's result.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    started = time.perf_counter()
    with log_path.open("wb") as log:
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(spec_path), str(result_path)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                last_progress = time.perf_counter()
                while True:
                    # Stream worker diagnostics/progress, including while loading.
                    for line in reader:
                        print(line, end="", flush=True)
                        last_progress = time.perf_counter()
                    if process.poll() is not None:
                        for line in reader:
                            print(line, end="", flush=True)
                        break
                    if time.perf_counter() - started >= timeout:
                        raise subprocess.TimeoutExpired(process.args, timeout)
                    if time.perf_counter() - last_progress >= 30:
                        print(f"  still running ({time.perf_counter() - started:.0f}s elapsed)",
                              flush=True)
                        last_progress = time.perf_counter()
                    time.sleep(0.1)
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if not isinstance(result, dict) or result.get("status") not in {"ok", "oom", "error"}:
                        raise ValueError("invalid worker result schema")
                except (OSError, ValueError) as exc:
                    result = {"status": "error", "stage": "worker", "error": f"invalid worker result: {exc}"}
                if process.returncode != 0 and result.get("status") == "ok":
                    result = {"status": "error", "stage": "worker", "error": "worker exited after reporting success"}
            else:
                result = {"status": "error", "stage": "worker", "error": "worker exited without a result"}
            result["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            result = {"status": "timeout", "stage": "worker", "error": f"exceeded {timeout:g} seconds"}
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
    result["worker_wall_seconds"] = time.perf_counter() - started
    if result.get("status") != "ok":
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 12000))
            result["worker_log_tail"] = log.read().decode("utf-8", errors="replace")
    return result


def flatten(row: dict) -> dict:
    output = {name: row.get(name) for name in (
        "backend", "model", "base_url", "device", "batch_size", "max_length", "candidates", "status", "stage",
        "load_seconds", "worker_wall_seconds", "parameter_dtype", "actual_device",
        "documents_per_second", "wall_documents_per_second", "amortized_ms_per_document", "measured_wall_seconds", "error",
        "requests_per_query", "length_policy", "baseline_backend", "baseline_max_length",
        "baseline_model", "baseline_base_url", "latency_speedup_vs_baseline",
        "throughput_change_pct_vs_baseline", "sampled_peak_change_gib_vs_baseline",
        "allocated_peak_change_gib_vs_baseline",
    )}
    output["gpu_ids"] = ",".join(row.get("gpu_ids", []))
    idle = row.get("idle_gpu_memory")
    output["idle_whole_gpu_gib"] = sum(gpu["used_bytes"] for gpu in idle) / GIB if idle else None
    memory = row.get("service_memory", {})
    peak = memory.get("sampled_peak_total_bytes")
    output["sampled_peak_whole_gpu_gib"] = peak / GIB if peak is not None else None
    output["memory_status"] = memory.get("status", "not_monitored")
    output["memory_errors"] = "; ".join(memory.get("errors", []))
    output["idle_memory_error"] = row.get("idle_gpu_memory_error")
    for name in ("count", "mean", "p50", "p95", "min", "max"):
        output[f"latency_{name}" + ("" if name == "count" else "_ms")] = row.get("latency_ms", {}).get(name)
    for phase in ("loaded", "warmed", "inference"):
        memory = row.get(f"{phase}_memory") or {}
        for metric in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
            value = memory.get(f"{metric}_bytes")
            output[f"{phase}_{metric}_gib"] = value / GIB if value is not None else None
    tokens = row.get("tokens", {})
    for name in ("min", "mean", "p95", "max"):
        output[f"effective_pair_tokens_{name}"] = tokens.get("effective_pair_tokens", {}).get(name)
    output["pairs_at_length_limit"] = tokens.get("pairs_at_length_limit")
    return output


def save_reports(paths: list[Path], report: dict) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after the new report is fully written.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                if path.suffix == ".json":
                    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.write("\n")
                else:
                    rows = [flatten(row) for row in report["results"]]
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def print_row(row: dict) -> None:
    flat = flatten(row)

    def number(key):
        value = flat.get(key)
        return f"{value:.2f}" if isinstance(value, (float, int)) else "-"

    if row.get("backend") == "vllm":
        print(
            f"{row['model']} @ {row['base_url']} | b={row['batch_size'] or 'all'} "
            f"n={row['candidates']} | {row['status']} | "
            f"mean={number('latency_mean_ms')}ms p95={number('latency_p95_ms')}ms | "
            f"docs/s={number('documents_per_second')} | "
            f"whole-GPU idle/sampled peak={number('idle_whole_gpu_gib')}/"
            f"{number('sampled_peak_whole_gpu_gib')} GiB ({flat['memory_status']})",
            flush=True,
        )
        if flat["memory_errors"] or flat["idle_memory_error"]:
            print(f"  memory warning: {flat['idle_memory_error'] or ''} {flat['memory_errors']}", flush=True)
    else:
        print(
            f"{row['model']} | b={row['batch_size']} len={row['max_length']} n={row['candidates']} | "
            f"{row['status']} | load={number('load_seconds')}s | "
            f"mean={number('latency_mean_ms')}ms p95={number('latency_p95_ms')}ms | "
            f"docs/s={number('documents_per_second')} | "
            f"peak alloc/reserved={number('inference_peak_allocated_gib')}/"
            f"{number('inference_peak_reserved_gib')} GiB",
            flush=True,
        )
        if row.get("gpu_ids"):
            print(f"  whole-GPU idle/sampled peak={number('idle_whole_gpu_gib')}/"
                  f"{number('sampled_peak_whole_gpu_gib')} GiB ({flat['memory_status']})",
                  flush=True)
            if flat["memory_errors"] or flat["idle_memory_error"]:
                print(f"  memory warning: {flat['idle_memory_error'] or ''} {flat['memory_errors']}", flush=True)
    if row.get("latency_speedup_vs_baseline") is not None:
        memory_delta = ("sampled_peak_change_gib_vs_baseline" if row.get("gpu_ids")
                        else "allocated_peak_change_gib_vs_baseline")
        memory_label = "sampled whole-GPU" if row.get("gpu_ids") else "allocated"
        print(f"  vs {row['baseline_model']}: latency speedup "
              f"{number('latency_speedup_vs_baseline')}x, "
              f"throughput change {number('throughput_change_pct_vs_baseline')}%, "
              f"{memory_label} peak delta {number(memory_delta)} GiB",
              flush=True)
    if row.get("error"):
        print(f"  {row.get('stage')}: {row['error']}", flush=True)


def add_comparison(row: dict, previous: list[dict]) -> None:
    """Compare only matching workloads; the first successful model is baseline."""
    if row["status"] != "ok":
        return
    keys = ("batch_size", "candidates")
    baseline = next((other for other in previous if other["status"] == "ok"
                     and other.get("server_index") != row.get("server_index")
                     and all(other.get(key) == row.get(key) for key in keys)
                     and (other["backend"] != row["backend"]
                          or other.get("max_length") == row.get("max_length"))), None)
    if baseline is None:
        return
    row.update(
        baseline_model=baseline["model"], baseline_base_url=baseline.get("base_url"),
        baseline_backend=baseline["backend"], baseline_max_length=baseline.get("max_length"),
        latency_speedup_vs_baseline=baseline["latency_ms"]["mean"] / row["latency_ms"]["mean"],
        throughput_change_pct_vs_baseline=(
            row["documents_per_second"] / baseline["documents_per_second"] - 1) * 100,
    )
    if all(item.get("service_memory", {}).get("status") == "ok" for item in (row, baseline)):
        row["sampled_peak_change_gib_vs_baseline"] = (
            row["service_memory"]["sampled_peak_total_bytes"]
            - baseline["service_memory"]["sampled_peak_total_bytes"]) / GIB
    current = (row.get("inference_memory") or {}).get("peak_allocated_bytes")
    prior = (baseline.get("inference_memory") or {}).get("peak_allocated_bytes")
    if current is not None and prior is not None:
        row["allocated_peak_change_gib_vs_baseline"] = (current - prior) / GIB


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--backend", choices=["cross-encoder", "vllm"], help="one backend for all models (default: cross-encoder)")
    backend_group.add_argument("--backends", nargs="+", choices=["cross-encoder", "vllm"], help="one backend per model")
    parser.add_argument("--models", nargs="+", required=True, help="local paths/HF IDs, or vLLM served model names")
    parser.add_argument("--base-urls", nargs="+", help="one server root per model; use '-' for local models")
    parser.add_argument("--gpu-ids", nargs="+", help="one physical GPU group per model (4 or 4,5); '-' skips monitoring")
    parser.add_argument("--memory-interval", type=positive_seconds, default=0.1, help="GPU sampling interval in seconds")
    parser.add_argument("--request-timeout", type=positive_seconds, default=120, help="HTTP request timeout in seconds")
    parser.add_argument("--api-key-env", help="environment variable containing the vLLM API key (value never saved)")
    parser.add_argument("--pause-before-model", action="store_true", help="wait for Enter before each model; manually start/stop services")
    parser.add_argument("--device", default="cuda:0", help="local backend device: cuda:N, cpu or mps")
    parser.add_argument("--batch-sizes", nargs="+", type=positive_int, help="local batch / serial HTTP request size; default: 1 4 8 (all-vLLM: all candidates)")
    parser.add_argument("--max-lengths", nargs="+", type=positive_int, help="local only; default: 512 1024")
    parser.add_argument("--candidates", nargs="+", type=positive_int, default=[200])
    parser.add_argument("--input", type=Path, help="query/documents JSONL; omitted for synthetic inputs")
    parser.add_argument("--queries", type=positive_int, default=3, help="number of input queries (default: 3)")
    parser.add_argument("--synthetic-words", type=positive_int, help="longest synthetic document word count")
    parser.add_argument("--warmup", type=positive_int, default=2, help="warmup rounds over all queries")
    parser.add_argument("--repeats", type=positive_int, default=10, help="timed rounds over all queries")
    parser.add_argument("--timeout", type=positive_seconds, default=1800, help="seconds per configuration, including load")
    parser.add_argument("--allow-download", action="store_true", help="allow HF downloads; default is offline")
    parser.add_argument("--output", type=Path, help="report prefix; writes PREFIX.json and PREFIX.csv")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing output reports")
    parser.add_argument("--dry-run", action="store_true", help="validate and print matrix only; no report files")
    return parser


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    backends = args.backends or [args.backend or "cross-encoder"] * len(args.models)
    if len(backends) != len(args.models):
        parser.error("--backends must supply one backend per model")
    has_http, has_local = "vllm" in backends, "cross-encoder" in backends
    mixed = has_http and has_local
    args.backend = "mixed" if mixed else backends[0]
    if has_http and (not args.base_urls or len(args.base_urls) != len(args.models)):
        parser.error("--base-urls must supply one entry per model; use '-' for local models")
    if args.base_urls and len(args.base_urls) != len(args.models):
        parser.error("--base-urls must supply one entry per model")
    if args.gpu_ids and len(args.gpu_ids) != len(args.models):
        parser.error("--gpu-ids must supply one GPU group per model; use '-' to skip")
    urls = args.base_urls or ["-"] * len(args.models)
    gpu_groups = []
    for backend, url, group in zip(backends, urls, args.gpu_ids or ["-"] * len(args.models)):
        ids = [] if group == "-" else group.split(",")
        if any(not (value.isdigit() or value.startswith("GPU-")) for value in ids) or len(set(ids)) != len(ids):
            parser.error("GPU groups must contain distinct physical indices or GPU UUIDs")
        if backend == "vllm":
            parsed = urllib.parse.urlsplit(url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in {"", "/"}):
                parser.error("vLLM URLs must be HTTP(S) server roots without credentials, query or path")
            if ids and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                parser.error("GPU monitoring requires a loopback URL on the server host; use '-' for remote servers")
        elif url != "-":
            parser.error("use '-' as the URL for a local model")
        gpu_groups.append(ids)
    if not has_local and args.max_lengths is not None:
        parser.error("--max-lengths controls local models only")
    if not has_http and args.api_key_env:
        parser.error("--api-key-env requires a vLLM backend")
    args.batch_sizes = args.batch_sizes or ([1, 4, 8] if has_local else [None])
    args.max_lengths = args.max_lengths or ([512] if mixed else [512, 1024])
    if mixed and len(set(args.max_lengths)) != 1:
        parser.error("mixed comparisons require one local --max-lengths value; remote limits remain server-controlled")
    if not (args.device in {"cpu", "mps", "cuda"} or
            (args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("--device must be cpu, mps, cuda or cuda:N (single-device PaiMem loader)")
    if args.input and args.synthetic_words:
        parser.error("--synthetic-words cannot be used with --input")
    longest_words = args.synthetic_words or (128 if has_http else 2 * max(args.max_lengths))
    paths = [Path(str(args.output) + suffix) for suffix in (".json", ".csv")] if args.output else []
    if not args.dry_run and not args.overwrite:
        for path in paths:
            if path.exists():
                parser.error(f"report exists: {path}; use a new prefix or --overwrite")
    try:
        records = (read_inputs(args.input, args.queries, max(args.candidates)) if args.input else
                   synthetic_inputs(args.queries, max(args.candidates),
                                    longest_words))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(records, ensure_ascii=False)
    matrix = []
    for index, (model, backend, url, ids) in enumerate(zip(args.models, backends, urls, gpu_groups)):
        lengths = args.max_lengths if backend == "cross-encoder" else [None]
        for batch, length, count in itertools.product(
                dict.fromkeys(args.batch_sizes), dict.fromkeys(lengths), dict.fromkeys(args.candidates)):
            matrix.append(dict(
                backend=backend, model=model, base_url=url.rstrip("/") if backend == "vllm" else None,
                gpu_ids=ids, device=args.device if backend == "cross-encoder" else None,
                batch_size=batch, max_length=length, candidates=count, server_index=index,
                requests_per_query=math.ceil(count / (batch or count)) if backend == "vllm" else None,
                length_policy="server-controlled; no client truncation" if backend == "vllm"
                              else f"CrossEncoder truncates to {length} tokens",
            ))
    warnings = []
    if has_http or any(gpu_groups):
        warnings.append("Whole-GPU memory includes all resident processes; sampled peaks are not allocator peaks. "
                        "GPU names/totals are recorded, but hardware and server settings must be matched manually.")
        warnings.append("HTTP inputs are sent without truncation; local inputs may be truncated. Match templates, dtype, token limits and cache "
                        "settings before comparing models/backends; repeated inputs may benefit from server caching.")
        if any(set(a) & set(b) for i, a in enumerate(gpu_groups) for b in gpu_groups[i + 1:]):
            warnings.append("Models share monitored GPU IDs. Unload the other model before each test "
                            "(--pause-before-model), or memory cannot be attributed to individual models.")
        if not any(gpu_groups):
            warnings.append("No GPU monitoring requested; service memory will be unavailable, not zero.")
    report = {
        "schema_version": 3, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "source": str(args.input.resolve()) if args.input else "synthetic",
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "queries": len(records), "max_candidates": max(args.candidates),
            "synthetic_longest_words": None if args.input else longest_words,
        },
        "settings": {
            "warmup_rounds": args.warmup, "measured_rounds": args.repeats,
            "calls_per_configuration": args.repeats * len(records),
            "timeout_seconds": args.timeout, "allow_download": args.allow_download,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": args.backend, "backends": backends, "request_timeout_seconds": args.request_timeout,
            "memory_interval_seconds": args.memory_interval,
            "pause_before_model": args.pause_before_model,
        },
        "measurement": {
            "time": "Whole query candidate scoring via PaiMem score(); synchronized; includes preprocessing and CPU output.",
            "memory": "CUDA PyTorch allocated/reserved bytes only; not total GPU/process usage. "
                      "Inference peaks include weights and warmed allocator cache.",
            "isolation": "Sequential fresh subprocess per configuration; no other models loaded by this script.",
            "load": "Includes model construction and any downloads, excludes imports/device initialization.",
            "limitations": "External GPU workloads may interfere. Small samples give unstable p95. "
                          "Synthetic input does not measure retrieval quality. CPU/MPS CUDA memory fields are null.",
        },
        "matrix": matrix, "warnings": warnings, "complete": False, "results": [],
    }
    if has_http or any(gpu_groups):
        report["measurement"] = {
            "time": "Whole query latency: all sequential batches, local inference or HTTP requests. HTTP includes serialization, "
                    "network, queueing, JSON decode and per-batch validation. Warmup excluded. docs/s uses the sum of "
                    "query latencies for both backends; wall_documents_per_second also includes loop/progress overhead.",
            "memory": "Physical whole-GPU nvidia-smi memory.used sampled before warmup and around timed calls. "
                      "Multi-GPU values sum simultaneous card usage. Sampling may miss peaks; other processes included.",
            "isolation": "One client subprocess per configuration; servers remain independently managed.",
            "load": "Local model construction measured; HTTP server startup not measured.",
            "limitations": "No score-quality or token equivalence guarantee. Small samples give unstable p95. "
                          "Loopback URLs must refer to this host, not forwarded remote services.",
        }
    print(f"{len(matrix)} configurations; {len(records)} queries; "
          f"{args.repeats * len(records)} timed calls/configuration; "
          f"input={report['input']['source']}; backend={args.backend}; "
          f"offline={not args.allow_download if args.backend == 'cross-encoder' else 'n/a'}", flush=True)
    for warning in warnings:
        print(f"Warning: {warning}", flush=True)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    with tempfile.TemporaryDirectory(prefix="reranker-benchmark-") as temporary:
        directory = Path(temporary)
        input_path = directory / "inputs.json"
        input_path.write_text(encoded, encoding="utf-8")
        previous_server = None
        for index, config in enumerate(matrix):
            if args.pause_before_model and config["server_index"] != previous_server:
                instruction = (f"Start ONLY model {config['model']} at {config['base_url']}"
                               if config["backend"] == "vllm" else
                               f"Local model {config['model']} will be loaded by this script")
                input(f"{instruction}; unload other models on its GPUs, then press Enter: ")
                previous_server = config["server_index"]
            print(f"[{index + 1}/{len(matrix)}] {config}", flush=True)
            case_dir = directory / str(index)
            case_dir.mkdir()
            spec = dict(config, input_path=str(input_path), warmup=args.warmup,
                        repeats=args.repeats, allow_download=args.allow_download,
                        request_timeout=args.request_timeout, memory_interval=args.memory_interval,
                        api_key_env=args.api_key_env)
            result = execute_case(spec, case_dir, args.timeout)
            row = dict(config, **result)
            add_comparison(row, report["results"])
            report["results"].append(row)
            report["complete"] = len(report["results"]) == len(matrix)
            report["all_succeeded"] = all(item["status"] == "ok" for item in report["results"])
            if paths:
                save_reports(paths, report)
            print_row(row)
    if paths:
        print("Reports: " + ", ".join(str(path.resolve()) for path in paths))
    return 0 if report["all_succeeded"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
