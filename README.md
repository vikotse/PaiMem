# PaiMem

[English](README.md) | [简体中文](README.zh-CN.md)

**Long-Term Memory with Hybrid Retrieval and Context-Aware Evidence Selection**

PaiMem stores original conversation chunks and retrieves traceable evidence through semantic and lexical retrieval, context-aware reranking, evidence selection, and context reconstruction. It provides SQLite persistence, a user-isolated HTTP API, and tools for retrieval and end-to-end evaluation.

## Key Insights

PaiMem formulates long-term memory reading as **query-conditioned selection of contextual evidence**, guided by three principles:

- **Preserve source information; defer evidence composition**: Retain conversation details and provenance, determining their evidential role at query time to limit information loss from early compression.
- **Decouple retrieval and assessment granularity**: Locate information through fine-grained chunks, while assessing relevance and selecting evidence using context-enriched representations.
- **Account for both relevance and complementarity**: Assess candidate relevance through reranking, then select complementary evidence using MemPursuit’s query-reconstruction residual as a proxy for uncovered information.

```text
/add: Messages → chunking / embedding → SQLite
/search: Query → Dense + BM25 → RRF → context-aware reranking → MemPursuit
      → optional rank fusion → same-session context expansion / packing → evidence
```

SIRA query expansion, NEAR residual retrieval, role filtering, conflict-evidence checks, and long-question processing are optional. Search returns evidence; answer generation and judging are part of the separate evaluation workflow. See [configuration and strategies](docs/configuration.md).

## 1. Quick start

Use Bash on macOS or Linux. PaiMem defaults to Python 3.11; vLLM requires a compatible Linux NVIDIA GPU host. All three scripts read **one TOML file**, containing deployment settings and strategy parameters together.

```bash
cd /path/to/PaiMem
cp -n configs/default.toml configs/config.local.toml
# Edit configs/config.local.toml before preparing the environment.
```

### Required settings for the default setup

The default setup uses **remote embedding + MemReranker served by vLLM on this machine**. On a compatible Linux NVIDIA GPU host, update the following **four fields** in `configs/config.local.toml` to get started:

| Field | What to enter |
|---|---|
| `service.api_key` | Choose your own memory-service secret; use it in API requests. |
| `embedding.remote.base_url` | Your embedding provider's actual API base URL, without `/embeddings`. |
| `embedding.remote.api_key` | Your embedding provider's API key. |
| `reranker.vllm.api_key` | Replace the placeholder with a secret of your choice, or explicitly set `api_key = ""` to disable API-key authentication. Do not leave the placeholder unchanged. |

**Both vLLM startup and PaiMem's reranker client read `reranker.vllm.api_key`**, so configure it once; no separate server/client keys are needed.

If your provider uses a different model name, update `embedding.remote.model` (the template uses `text-embedding-v4`); adjust RPM/TPM to your provider's quotas. Other strategy parameters and disabled-strategy examples can stay as supplied. Remote embedding does not download a model; preparation downloads the selected reranker and its template.

> **Switching embedding models?** Also set `service.db_path` to a new path, such as `artifacts/memory-bge-m3.db`, and re-ingest your original data. Existing vectors are not converted automatically. The new database starts empty; the original database is preserved. Changing only API keys, batch size, or RPM/TPM does not require a new database.
>
> The compatibility check also tracks the embedding backend, remote `base_url`, local model path, query prefix, and ingestion context (`context.messages`). Changes to these fields can block reuse of a populated database. It cannot detect changed weights behind an unchanged model name or path.

```bash
# Install/check uv, install dependencies, and download the selected reranker and template.
bash scripts/setup_env.sh
# Terminal 1: start vLLM and leave it running; wait until it is ready.
bash scripts/start_vllm.sh
```

```bash
# Terminal 2, from the repository root:
bash start_svr.sh --check
bash start_svr.sh
```

**To connect to an existing vLLM reranking service**, set `reranker.vllm.deployment` to `"remote"` and enter its `base_url`, served model name (`model`), and `api_key` (use `""` if authentication is disabled). Here, `"remote"` means using an already running service, whether on this machine or another machine. Run `setup_env.sh`, then `start_svr.sh`. You do not need to download the reranker model, install vLLM, or run `start_vllm.sh`.

No manual environment activation is needed. PaiMem uses `.venv`; vLLM uses a separate environment. Finish model preparation and wait for vLLM readiness before starting PaiMem. `--check` validates configuration and checks existing database compatibility read-only, without loading models or calling remote services. Normal startup performs the same check before model loading.

See [configuration](docs/configuration.md) for local embedding, CrossEncoder, overrides and strategy settings; see [vLLM deployment](docs/vllm.md) for server parameters, separate hosts and readiness checks.

## 2. API and request examples

In another terminal, set the service URL and use the same value as `service.api_key` for `MEMORY_API_KEY`. Requests and evaluations below use these variables:

```bash
export MEMORY_BASE_URL='http://127.0.0.1:8000'
export MEMORY_API_KEY='YOUR_MEMORY_SERVICE_KEY'
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health and memory counts; no authentication |
| POST | `/add` | Synchronous ingestion; searchable immediately after success |
| POST | `/search` | Retrieve by user_id; top_k is 1–100 |
| GET | `/config` | Inspect effective settings without exposing keys |
| POST | `/config` | Update supported runtime switches and parameters |
| POST | `/clear` | Clear the entire dedicated memory database; authentication required |
| GET | `/docs` | Generated OpenAPI documentation |

Protected endpoints accept `X-Api-Key`, `Authorization: Bearer <key>`, or `Authorization: Token <key>`. The `/config` and `/clear` management endpoints require the service key, including when `ALLOW_UNAUTHENTICATED=true`. See `/docs` and the configuration guide for parameters.

```bash
curl --fail "$MEMORY_BASE_URL/health"

curl --fail "$MEMORY_BASE_URL/add" \
  -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"request_id":"demo:1","user_id":"demo","session_id":"session:1","messages":[{"role":"user","timestamp":1704067200000,"content":"My favorite drink is green tea."}]}'

curl --fail "$MEMORY_BASE_URL/search" \
  -H "X-Api-Key: $MEMORY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"What is my favorite drink?","user_id":"demo","top_k":10}'

# Clear the entire memory database, including all users' data.
curl --fail -X POST "$MEMORY_BASE_URL/clear" \
  -H "X-Api-Key: $MEMORY_API_KEY"
```

Add returns `success/request_id/user_id/session_id`. Repeating a request_id with identical content is idempotent; different content returns HTTP 409. The optional `timestamp` uses Unix milliseconds.

Search returns `{"data":[{"id":"...","content":"...","score":0.8,"created_at":"..."}]}` and accepts `options` for multiple-choice questions. Data is isolated by user_id. Preserve the returned order when using MemPursuit or rank fusion; do not sort again by score. Set `include_recall_queue: true` to include diagnostic queue sources. File-based diagnostics require an authenticated loopback request and are restricted to `[service].diagnostics_dir`; see [diagnostic directory configuration](docs/configuration.md#diagnostic-output-directory).

## 3. Reproduce retrieval and end-to-end results

See [evaluation](docs/evaluation.md) for the complete workflow and dataset mapping: obtain a fixed AML pipeline revision → download public data → convert to normalized JSONL → start the chosen service configuration → evaluate.

Retrieval example, without answer generation or judging; enabled service strategies may still call models:

```bash
uv run python scripts/local_benchmark.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/recall/locomo.jsonl \
  --base-url "$MEMORY_BASE_URL" --run-label locomo-run-001 --top-k 100
```

End-to-end example; first export `ANSWER_API_BASE/ANSWER_MODEL/ANSWER_API_KEY` and the corresponding `JUDGE_*` variables:

```bash
PYTHONHASHSEED=0 uv run python scripts/run_aml_e2e.py \
  --input data/normalized/locomo_refined.fixed.jsonl \
  --output artifacts/aml-e2e/run-001/locomo_refined \
  --pipeline agent-memory-leaderboard/data/locomo-refined/pipeline.py \
  --base-url "$MEMORY_BASE_URL" --top-k 100 --concurrency 2 \
  --save-search-results
```

The same configuration and output directory support resuming. Use a new output directory when changing models, data, or strategies. Do not clear memory during a resumed run. Record the code version, models, data, and configuration alongside results, as described in the evaluation guide.

## 4. Efficiency

We evaluate PaiMem’s ingestion and retrieval latency on **LoCoMo-refined** using the default parameters with the embedding backend switched to **local**, avoiding network requests to a remote embedding provider. The reranker is served through **vLLM**, with `gpu_memory_utilization` configured to keep the vLLM GPU memory budget at approximately **20 GB** on both GPUs.

The evaluation includes **1 record, 21 Add requests, and 138 questions**.

| GPU | Average Add latency (s/request) | Average Search latency (s/request) |
|---|---:|---:|
| NVIDIA A800 (80GB) | 0.1581 | 3.5244 |
| NVIDIA L4 (24GB) | 0.0380 | 12.3927 |

These results report mean request latency for this configuration and sample. Search latency measures memory retrieval and excludes downstream answer generation and judging.

## 5. Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/vikotse">
        <img src="docs/assets/contributors/vikotse.svg" width="80" height="80" alt="vikotse"/><br/>
        vikotse
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Vicky-shi">
        <img src="docs/assets/contributors/Vicky-shi.svg" width="80" height="80" alt="Vicky-shi"/><br/>
        Vicky-shi
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/nicole0220">
        <img src="docs/assets/contributors/nicole0220.svg" width="80" height="80" alt="nicole0220"/><br/>
        nicole0220
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hiudawn">
        <img src="docs/assets/contributors/hiudawn.svg" width="80" height="80" alt="hiudawn"/><br/>
        hiudawn
      </a>
    </td>
  </tr>
</table>

## 6. References

| Work | Use in PaiMem |
|---|---|
| [InvMem / vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | Foundation for the RAG and service implementation |
| [MemPursuit — Recall What is Missing: Composing Evidence from Long-Term Agent Memory](https://github.com/FengQi-HITSZ/MemPursuit) | Adapted fixed-budget NumPy evidence selection |
| [SIRA](https://github.com/facebookresearch/sira) · [Paper](https://arxiv.org/abs/2605.06647) | Query expansion, DF filtering, and lexical augmentation |
| [NEAR: Neighborhood-Aware Residuals for Memory Retrieval](https://arxiv.org/abs/2610.13314) | Optional neighborhood-residual scoring |
| [MemReranker](https://huggingface.co/IAAR-Shanghai/MemReranker-4B) · [Paper](https://arxiv.org/abs/2605.06132) | Optional memory reranking model |
| [Agent Memory Leaderboard](https://github.com/AML-memory/agent-memory-leaderboard) | Evaluation protocol and external answer/judge pipelines |

See [third-party notices](THIRD_PARTY_NOTICES.md) for sources and licenses, and [strategy documentation](docs/configuration.md#algorithm-documentation) for their implementation in PaiMem.

## 7. License

PaiMem is licensed under the [MIT License](LICENSE). See [Third-Party Notices](THIRD_PARTY_NOTICES.md) for third-party licenses.

## 8. Citation

```bibtex
@software{paimem2026,
  author = {Xie, Weijian and Shi, Lihui and Zhang, Xinyue and Gao, Xiaodong},
  title = {PaiMem: Long-Term Memory with Hybrid Retrieval and Context-Aware Evidence Selection},
  year = {2026},
  url = {https://github.com/vikotse/PaiMem}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff). When citing experimental results, also record the code commit, model revision, configuration, data split, and evaluator version.
