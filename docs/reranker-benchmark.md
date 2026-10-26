# Reranker performance evaluation

[English](reranker-benchmark.md) | [简体中文](reranker-benchmark.zh-CN.md) | [README](../README.md)

For MemReranker GPU deployments, vLLM supports independent management of model serving and request batching. This guide describes a reproducible comparison of local CrossEncoder and vLLM `/score` deployments. To measure backend acceleration, keep the model, hardware, inputs, and scoring configuration consistent.

## Reproducible comparison

Prepare one `rerank-input.jsonl` file with a `query` and at least 200 `documents` strings per line. Set each backend's actual model identity. First use `--dry-run` to validate arguments.

```bash
export LOCAL_RERANKER_MODEL='/absolute/models/compatible-cross-encoder'
export MEMORY_RERANKER_MODEL='served-reranker-name'
export MEMORY_RERANKER_BASE_URL='http://127.0.0.1:8089'
uv run python scripts/benchmark_rerankers.py \
  --backends cross-encoder vllm \
  --models "$LOCAL_RERANKER_MODEL" "$MEMORY_RERANKER_MODEL" \
  --base-urls - "$MEMORY_RERANKER_BASE_URL" \
  --device cuda:0 --batch-sizes 16 --max-lengths 512 \
  --input rerank-input.jsonl --queries 3 --candidates 200 \
  --warmup 2 --repeats 10 --pause-before-model \
  --api-key-env MEMORY_RERANKER_API_KEY \
  --output artifacts/reranker-comparison
```

Omit `--api-key-env` if authentication is disabled. When both backends use the same GPU, follow the prompts to stop the other backend and avoid memory or compute interference. The script does not start or stop services itself.

Outputs are `.json` and `.csv`, containing configuration-level timings and available memory metrics. The script measures sequential full-pool calls. `candidate count / mean time` is the processing rate for that workload, not the limit of concurrent serving. HTTP timing includes networking, queueing, and serialization. Local timing includes tokenization, batching, and score transfer. Both exclude model loading and warmup.

Record the GPU, driver, software versions, model revisions, dtype, templates, candidate count, token lengths, and batch size. In this benchmark, `--max-lengths` controls only the local path; configure the remote service separately and ensure inputs fit. If models or scoring templates differ, describe the result as a deployment-configuration comparison, not a vLLM speedup. A BGE-versus-MemReranker comparison cannot isolate backend acceleration.
