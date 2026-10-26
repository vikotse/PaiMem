# Configuration and strategies

[English](configuration.md) | [简体中文](configuration.zh-CN.md) | [README](../README.md)

All three scripts read the same TOML. Copy [default.toml](../configs/default.toml) (or [baseline.toml](../configs/baseline.toml)) to `configs/config.local.toml`, then edit the local copy. Model credentials, backend setup and algorithm parameters are grouped together by feature.

## File selection and overrides

`--config PATH` takes precedence over `PAIMEM_CONFIG`, then the existing `configs/config.local.toml`, then `configs/default.toml`. The selected file is complete: it is not merged with another profile. Paths are relative to the repository root when using the shell entry points.

Existing environment exports override fields in the selected TOML; comments beside every field give its export name. An explicit empty export also wins (and fails if that field is required). Use `unset NAME` to restore the file value. For example:

```bash
export EMBEDDING_BATCH_SIZE=20
bash start_svr.sh --config configs/config.local.toml --check
```

Only the selected backend is mapped: `embedding.local` versus `embedding.remote`, and `reranker.local` versus `reranker.vllm`. Select backends through `embedding.backend` and `reranker.backend`. Local and remote examples have their own batch sizes and model identities. `reranker.enabled=false` disables the main reranker even if model variables were exported previously.

`REPLACE_WITH_*` and `*.example` values are placeholders. Active required fields must be filled; disabled strategies may keep their examples. An empty `reranker.vllm.api_key` explicitly means no server API-key authentication. Empty `reranker.score_threshold` disables score filtering. Local `query_prefix="auto"` preserves model defaults; `""` explicitly removes the prefix. `download=false` means use an existing model directory; `revision="main"` follows the model's main revision, while a commit SHA pins it.

`--check` validates required credential fields, types, selected numeric ranges and strategy dependencies, and checks existing database compatibility read-only. It does not load models, call remote services, or verify credentials against providers. Remaining numerical constraints are checked by the API during startup. `setup_env.sh --only vllm` validates only vLLM settings. Keys are not printed in check output.

Each LLM strategy (`sira`, `role_name`, `conflict`, `long_question`) has its model, base URL, key, timeout and algorithm parameters in its own section. Shared `ANSWER_*`/`OPENAI_*` fallbacks apply only when its connection field is explicitly empty; nonempty placeholders must be replaced. Use concrete values in the strategy section to avoid ambiguity.

The service launcher loads TOML automatically. Other Python tools that only consume environment variables need an explicit import of the configuration. In Bash, from the repository root:

```bash
config_exports=$(.venv/bin/python -m paimem.configuration --config configs/config.local.toml --shell)
eval "$config_exports"
unset config_exports
```

`--shell` emits safely quoted exports **including credentials**; capture it as above, do not print or log it. TOML itself is data and is never executed as shell code.

## Configuration sections

| Section | Settings kept together |
|---|---|
| `[service]`, `[environment]` | API key, database, listener; Python and package/model download endpoints |
| `[embedding]`, `.remote`, `.local` | Backend, model/API or download path, batch size, RPM/TPM |
| `[reranker]`, `.local`, `.vllm` | Switch, retrieval candidates, context, local model or vLLM client and server |
| `[retrieval]`, `[context]`, `[time]` | Hybrid retrieval, chunking, context reconstruction, time handling |
| `[mempursuit]`, `[rank_fusion]`, `[near]` | Each strategy's switch and numerical parameters |
| `[sira]`, `[role_name]`, `[conflict]`, `[long_question]` | Each strategy's switch, LLM model/URL/key and algorithm parameters |

## Embedding: remote or local

### Remote embedding — use a third-party embedding model API without downloading the model.

In your copied file, edit the existing sections (do not append duplicate TOML tables):

```toml
[embedding]
backend = "remote"

[embedding.remote]
model = "text-embedding-v4" # Example; use the name accepted by your provider
base_url = "https://your-provider.example/v1" # Replace this endpoint
api_key = "REPLACE_WITH_EMBEDDING_API_KEY"
batch_size = 10
rpm = 1500    # Example quota; set your actual limit, or 0 to disable
tpm = 800000 # Example estimated-token quota; 0 disables
query_prefix = "" # Explicitly no prefix
```

Preparation skips all embedding downloads. No local embedding model directory or embedding vLLM process is needed. Do not append `/embeddings` to the API base. RPM/TPM use a rolling 60-second window; TPM is conservatively estimated using UTF-8 byte length. The quotas above are examples; use your provider's actual limits. See [rate-limit behavior](#embedding-batching-and-remote-rate-limits).

### Local embedding — prepared before PaiMem starts

Set `embedding.backend` to `"local"`; the separate local example is already populated:

```toml
[embedding.local]
model_path = "models/bge-small-en-v1.5"
repo_id = "BAAI/bge-small-en-v1.5"
download = true
revision = "main" # Replace with a model commit SHA for fixed results
device = "cuda:0" # Or cpu, mps
batch_size = 64
query_prefix = "auto" # Preserve the model's default; "" explicitly disables
```

The preparation script downloads to `model_path`; set `download=false` to reuse an existing directory. PaiMem loads the model at startup. Remote credentials and remote batch settings are ignored. When changing the embedding model, backend, or context encoding, use a new `service.db_path` and ingest again.

## Reranker backends

For MemReranker/Qwen3, [vLLM serving](vllm.md) is recommended; a local backend is also available below.

### Local CrossEncoder — loaded by PaiMem

Set `reranker.backend="cross-encoder"`. `[reranker.local]` already provides `BAAI/bge-reranker-v2-m3`, its download directory, `device="cuda:0"` in the default profile (`cpu` in baseline), and `batch_size=16`. Change these fields as needed. `download=false` reuses the existing model directory. No vLLM process or vLLM credentials are required. CUDA inference requires compatible drivers and PyTorch; scripts do not install drivers.

Reranking candidates, input token budget, score threshold and context switch remain beside the backend selector in `[reranker]`. Switching models may require threshold recalibration. The [baseline template](../configs/baseline.toml) disables reranking, MemPursuit and rank fusion and sets packing to 1.

## Environment preparation and alternative profiles

No repeated exports or environment activation are required. PaiMem dependencies are installed from `uv.lock` into `.venv`; vLLM uses its separate configured environment. Preparation can be rerun. Startup performs no package installation. Model preparation must finish before starting the service.

All three scripts support `--config configs/experiment.local.toml`. See [vLLM deployment](vllm.md) for separate hosts.

Optional mirrors are configured in `[environment]`. The standard [uv installer](https://docs.astral.sh/uv/getting-started/installation/) is used when uv is absent. To start from the baseline, copy `configs/baseline.toml` to a **new** `*.local.toml`, fill its active credentials, and pass that same path to preparation and startup.

## Embedding batching and remote rate limits

Both profiles group settings by feature, with bilingual parameter comments. Exported variables take precedence over TOML. Model connections and algorithm parameters are in the same feature section.

| Variable | default profile | Meaning |
|---|---:|---|
| `EMBEDDING_BATCH_SIZE` | Local: 64; remote: 10 | Positive integer; maximum inputs per batch. Respect the provider's per-request limit. |
| `EMBEDDING_RPM` | 1500 | Nonnegative integer; remote HTTP attempts per rolling 60 seconds. Zero disables this limit. |
| `EMBEDDING_TPM` | 800000 | Nonnegative integer; estimated remote input tokens per rolling 60 seconds. Zero disables this limit. |

Values above come from `configs/default.toml`; adjust RPM/TPM to your provider's quotas. The `baseline` profile and direct API startup without explicit quotas use 0, disabling the corresponding limits.

```bash
# Illustrative quotas: replace with your provider's limits.
export EMBEDDING_BATCH_SIZE=10
export EMBEDDING_RPM=1500
export EMBEDDING_TPM=800000
bash start_svr.sh --config configs/config.local.toml
```

Remote calls wait until both enabled quotas allow a request. Every retry reserves quota again, including attempts that fail. Batches are split to satisfy both batch size and TPM; an individual input above TPM fails before any input in that call is sent, without truncation. Query prefixes count toward the estimate. Local embedding only uses batch size.

TPM uses UTF-8 byte length as a conservative token estimate, not the provider's tokenizer or billing count. Leave quota headroom and adjust to your provider. A valid `Retry-After` response header (seconds or HTTP date) is honored in addition to exponential backoff and local quota checks, with at most three attempts per batch.

The limiter is thread-safe and shared by document and query calls on the same embedder instance. It does not coordinate multiple processes, instances, or other clients sharing the account; divide quotas accordingly. Limits are read at embedder creation and require a service restart to change. `--check` validates active embedding batch size and quota limits.


## Two profiles

| Parameter | baseline | default |
|---|---:|---:|
| Dense + BM25 / RRF | On | On |
| BM25 weight / RRF k | 0.5 / 60 | 0.5 / 60 |
| Reranker | Off | On, 400 candidates |
| Context-aware reranking | Off | On |
| MemPursuit | Off | On, candidate parameter 150 |
| MemPursuit context vectors | Off | On |
| Five-channel rank fusion | Off | Off; configured weights: Dense 0.35 + reranker 0.65 |
| Fusion prefixes: MP / remaining | — | 100 / 300 |
| Neighbor window / seed count | 1 / 20 | 1 / 20 |
| pack_span | 1 | 2 |
| SIRA / NEAR / role / conflict / long question | Off | Off |

`default` combines PaiMem's strategies; models and devices are configured in their TOML sections (with optional exported overrides). Results depend on the models, data, and runtime environment. MemPursuit's actual pool size also depends on top_k and the number of eligible upstream candidates; see the strategy documentation.

Edit boolean settings to toggle strategies. You can copy a profile to `configs/experiment.local.toml`; this suffix is ignored by Git. Example overrides:

```bash
export MEMORY_MEMPURSUIT_ENABLED=false
export MEMORY_RANK_FUSION_ENABLED=false # Requires MP; disable it together with MP
bash start_svr.sh --config configs/config.local.toml
```

The baseline profile prevents the main reranker from loading, even if model variables remain in the terminal environment. Rank fusion cannot be enabled while `[reranker] enabled=false`.

## Optional LLM strategies

Fill the model, base_url and api_key alongside enabled in each strategy section. Optional environment overrides still work:

```bash
export MEMORY_SIRA_MODEL='your-model'
export MEMORY_SIRA_API_BASE='https://your-provider.example/v1'
export MEMORY_SIRA_API_KEY='your-key'
```

Role extraction uses `MEMORY_ROLE_NAME_MODEL/API_BASE/API_KEY`; conflict checks use `MEMORY_CONFLICT_MODEL/API_BASE/API_KEY`. For explicitly empty connection fields, shared `ANSWER_MODEL/ANSWER_API_BASE/ANSWER_API_KEY` variables are supported, followed by fallback to `OPENAI_MODEL/OPENAI_BASE_URL/OPENAI_API_KEY`. Nonempty placeholders must be replaced. Role extraction also needs a reranker. It reuses the main reranker by default; a separate `MEMORY_ROLE_NAME_RERANKER_MODEL` is loaded locally.

## Long-question processing

`[long_question]` is disabled by default. When enabled and the query is longer than `threshold_chars`, the LLM receives only the first `head_chars` and last `tail_chars` characters to extract a retrieval query. The complete original question is chunked using the normal ingestion preparation path and joins historical memories as request-local evidence. It is not written to SQLite. Failed extraction/preparation falls back to the original query. Temporary evidence does not enter the shared NEAR cache.

All nine parameters are in one section:

```toml
[long_question]
enabled = false
model = "gpt-4o-mini"
base_url = "https://your-llm-provider.example/v1"
api_key = "REPLACE_WITH_LONG_QUESTION_API_KEY"
extra_body = '{}'
threshold_chars = 8000
head_chars = 2000
tail_chars = 2000
timeout = 30
```

`extra_body` is a JSON **object string**, not a TOML table or JSON array. `{}` or an empty string sends no extra parameters. For providers that support it, examples are `'{"enable_thinking":false}'` or `'{"thinking":{"type":"disabled"}}'`; use the format accepted by your endpoint. Counts are characters, not tokens; head + tail must not exceed the threshold. `--check` validates these bounds, positive timeout and JSON shape when enabled.

Model connection and extraction parameters take effect at startup. Runtime `long_question_enabled` can be changed through `/config` only if an extractor was configured at startup; otherwise enable the strategy in TOML and restart. Diagnostic retrieval traces include application/fallback status and the temporary evidence count.

## Runtime configuration and management endpoints

GET `/config` returns current values. POST `/config` accepts only fields declared in `ConfigPatch`, listed in `/docs`. Runtime changes are not written back to TOML; restarting reloads startup configuration. Startup settings and runtime switches are not interchangeable.

GET/POST `/config` and `/clear` require the service API key, including when `ALLOW_UNAUTHENTICATED=true`. Supply `[service].api_key` through `X-Api-Key` or `Authorization: Bearer`.

```bash
curl --fail "$MEMORY_BASE_URL/config" -H "X-Api-Key: $MEMORY_API_KEY"
curl --fail "$MEMORY_BASE_URL/config" -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"rank_fusion_enabled":false,"mempursuit_enabled":false}'
```

## Algorithm documentation

- [MemPursuit](strategies/MEMPURSUIT.md): residual-driven, fixed-budget selection.
- [Rank fusion](strategies/RANK_FUSION.md): weighted candidate rank scores.
- [SIRA](strategies/SIRA.md): query-side expansion and DF filtering.
- [NEAR](strategies/NEAR.md): neighborhood-residual scoring.
- [Reranker](strategies/RERANKING.md): local and vLLM interfaces.

These documents describe implementation details and adaptation boundaries in PaiMem. Regression tests are in `tests/`. Evaluate the default profile with your chosen models and datasets to assess its quality.

## Diagnostic output directory

`[service].diagnostics_dir` (environment variable `MEMORY_DIAGNOSTICS_DIR`, default `artifacts/diagnostics`) limits writes through `/search` with `debug_output_path`. Diagnostic requests require a loopback client and a valid service key. The target must be an absolute `.json` path inside this directory, with an existing parent and no symbolic links or `..` components. Existing regular files are atomically replaced to support evaluation retries. The directory setting is loaded at startup.
