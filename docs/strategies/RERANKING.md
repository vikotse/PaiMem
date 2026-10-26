# Reranker backends and context-aware scoring

[English](RERANKING.md) | [简体中文](RERANKING.zh-CN.md) | [README](../../README.md)

PaiMem supports an in-process Sentence Transformers CrossEncoder and a vLLM pooling `/score` service. The [default profile](../../configs/default.toml) enables MemReranker-4B through vLLM with context-aware reranking. The baseline profile disables reranking.

Edit the existing sections in `configs/config.local.toml`; environment overrides and file selection are described in [configuration](../configuration.md). Backend and model changes require a restart. Changing only the reranker does not require rebuilding the memory database.

## vLLM deployment

The shared `[reranker.vllm]` section configures the server and PaiMem client. These entries match the default profile; replace the API-key placeholder or explicitly set `api_key = ""` to disable authentication:

```toml
[reranker]
enabled = true
backend = "vllm"
candidates = 400
max_length = 1024
small2big_enabled = true
score_threshold = ""

[reranker.vllm]
deployment = "local"
model = "MemReranker-4B"
base_url = "http://127.0.0.1:8089"
api_key = "REPLACE_WITH_RERANKER_API_KEY"
timeout = 120
batch_size = 128
```

Complete the other required service and embedding settings before running the scripts. Local vLLM deployment requires Linux and a compatible NVIDIA GPU:

```bash
bash scripts/setup_env.sh
# Terminal 1: keep vLLM running and wait for readiness.
bash scripts/start_vllm.sh
```

```bash
# Terminal 2: start PaiMem after vLLM is ready.
bash start_svr.sh --check
bash start_svr.sh
```

To connect to an existing service, set `deployment="remote"` and fill its address, served model name, and API key. The service may run on the same machine or another machine. In this mode, setup skips local vLLM installation and reranker downloads; run `setup_env.sh` and `start_svr.sh`, without `start_vllm.sh`.

The vLLM startup script uses the Qwen3 reranker chat template and classification-head configuration required by MemReranker:

```text
--runner pooling
--hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
```

See [vLLM deployment](../vllm.md) for environment, model-download, and readiness details, and the [MemReranker model card](https://huggingface.co/IAAR-Shanghai/MemReranker-4B) for upstream model documentation.

## Local CrossEncoder

Set `backend="cross-encoder"` to load the reranker inside PaiMem. The backend value is `cross-encoder`; `local` is the configuration section name, not a backend value.

```toml
[reranker]
enabled = true
backend = "cross-encoder"
score_threshold = ""

[reranker.local]
model_path = "models/bge-reranker-v2-m3"
repo_id = "BAAI/bge-reranker-v2-m3"
download = true
revision = "main"
device = "cuda:0"
batch_size = 16
```

Use `device="cpu"` for CPU inference. Run `setup_env.sh` to prepare dependencies and the selected model, then `start_svr.sh` to load it. Set `download=false` to reuse an existing model directory. vLLM connection settings are not used by this backend.

## Scoring and context

| Field | Default profile | Behavior |
|---|---:|---|
| `reranker.candidates` | `400` | Candidate count, raised to at least the internal `top_k` and bounded by available memories |
| `reranker.max_length` | `1024` | Maximum scoring input length |
| `reranker.small2big_enabled` | `true` | Include adjacent same-session chunks when scoring |
| `reranker.score_threshold` | `""` | Empty disables threshold filtering; otherwise use a numeric string |
| `reranker.vllm.batch_size` | `128` | Documents per HTTP `/score` request |
| `reranker.vllm.timeout` | `120` | Per-request timeout in seconds |

With small-to-big enabled, scoring text includes the preceding and following positions in the retrieval sequence when they belong to the same session. This scoring window is independent of final output expansion. Reranker scores order the candidates and optionally filter seeds before MemPursuit and context packing.

For vLLM, batches are sent sequentially. With 200 candidates and `batch_size=128`, the client sends two requests containing 128 and 72 documents. This setting does not specify vLLM's internal GPU batch size. Request failures propagate without automatic retries, candidate omission, or a local-backend fallback.

The client sends `max_length` as `truncate_prompt_tokens`. It must not exceed `reranker.vllm.max_model_len`; the limit includes query, document, and template tokens. Truncation may remove useful context. The server must support this request parameter. GPU placement, precision, and server memory are configured in `[reranker.vllm]`; the local CrossEncoder device setting does not apply to the HTTP backend.

The client preserves server scores without an additional sigmoid or logit transformation. Recalibrate thresholds when changing models or backends. To disable filtering, set `reranker.score_threshold=""`. If `MEMORY_RERANK_SCORE_THRESHOLD` is exported, set it to `""` or unset it to restore the TOML value; unsetting alone does not clear a nonempty TOML threshold.

`base_url` must be the service root without `/v1` or `/score`, and `model` must match the served model name. PaiMem validates configuration at startup without probing the reranker endpoint; connectivity, authentication, and served-model availability are verified by scoring requests.

## Runtime and Python API

`POST /config` supports `reranker_enabled`, `rerank_candidates`, and `rerank_small2big_enabled`. Re-enabling requires a reranker configured at startup. Runtime changes affect the current process and do not modify TOML. See [configuration](../configuration.md) for management-endpoint access.

For direct Python API use, call `configure_store()` before the first store is created:

```python
from paimem.api import configure_store

configure_store(
    reranker_backend="vllm",
    reranker_model="MemReranker-4B",
    reranker_base_url="http://127.0.0.1:8089",
    reranker_api_key="",  # Use the server key when authentication is enabled.
    reranker_batch_size=128,
    reranker_timeout=120,
    reranker_max_length=1024,
    rerank_score_threshold=None,
)
```

Explicit arguments take precedence over environment variables. Direct API defaults differ from the launcher profile: the backend defaults to `cross-encoder` and scoring length to 512; no main reranker is constructed without a model. Use explicit settings when embedding PaiMem in another application.

Role-name extraction reuses the main reranker unless a different `role_name.reranker_model` is specified. A separate role-name reranker is loaded locally.

## Performance evaluation

Use [reranker benchmarking](../reranker-benchmark.md) to compare local and vLLM inference with `scripts/benchmark_rerankers.py`. Use the same inputs and record model, template, truncation, batching, and hardware settings. Throughput and memory usage depend on the deployment configuration.
