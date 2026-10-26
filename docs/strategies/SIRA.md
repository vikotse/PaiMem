# SIRA query-side expansion

[English](SIRA.md) | [简体中文](SIRA.zh-CN.md) | [README](../../README.md)

SIRA generates query-expansion phrases with an LLM and filters them using document frequencies in the current memory store to augment BM25 matching. PaiMem integrates query-side expansion and filtering. The default profile disables SIRA; no re-ingestion or embedding rebuild is required. See [third-party notices](../../THIRD_PARTY_NOTICES.md) for source revisions and licensing.

## Configuration

Edit the existing `[sira]` section in `configs/config.local.toml`. This example enables SIRA; supply a working model, endpoint, and key:

```toml
[sira]
enabled = true
model = "qwen-plus"
base_url = "https://your-llm-provider.example/v1"
api_key = "REPLACE_WITH_SIRA_API_KEY"
expansion_weight = 0.5
max_df_ratio = 0.05
max_n = 4
timeout = 30.0
max_tokens = 1024
```

```bash
bash start_svr.sh --check
bash start_svr.sh
```

| `[sira]` field | Default profile | Description |
|---|---:|---|
| `enabled` | `false` | Enable expansion; mutable at runtime |
| `expansion_weight` | `0.5` | Expanded BM25 score weight; 0 avoids LLM calls |
| `max_df_ratio` | `0.05` | DF upper-bound ratio; 0 checks existence only |
| `max_n` | `4` | Maximum DF sliding-window length, 1–4 |
| `timeout` | `30.0` | Model request timeout in seconds; no retries |
| `max_tokens` | `1024` | Maximum output tokens per expansion |

With the launcher, exported environment variables override TOML. For an enabled strategy, only explicitly empty connection fields use `ANSWER_*` and `OPENAI_*` fallbacks; nonempty placeholders must be replaced. Prefer a complete connection configuration in `[sira]`. See [configuration](../configuration.md).

### Runtime switching

```bash
curl -X POST http://127.0.0.1:8000/config -H "X-Api-Key: $MEMORY_API_KEY" \
  -H 'Content-Type: application/json' -d '{"sira_enabled":false}'
```

`POST /config` supports only `sira_enabled`; other SIRA settings require a restart. Re-enabling requires a model client configured at startup. Start with a valid enabled configuration, then pause or resume as needed. When `enabled=false` at startup, the launcher skips the section’s other TOML fields; filling connection fields there alone does not ensure runtime enabling is available.

`GET /config` returns effective settings without keys. Runtime switches affect the current process and do not modify TOML. See [configuration](../configuration.md) for management-endpoint access.

### Direct Python API use

When using `paimem.api` directly without the TOML launcher, client initialization selects the first nonempty environment variable in each row:

| Field | Precedence |
|---|---|
| model | `MEMORY_SIRA_MODEL` → `ANSWER_MODEL` → `LLM_MODEL` → `OPENAI_MODEL` |
| key | `MEMORY_SIRA_API_KEY` → `ANSWER_API_KEY` → `API_KEY` → `OPENAI_API_KEY` |
| base | `MEMORY_SIRA_API_BASE` → `ANSWER_API_BASE` → `API_BASE` → `OPENAI_BASE_URL` |

Complete connection settings create a client that can be enabled at runtime. Client creation sends no request. This fallback order describes direct API initialization only.

## Retrieval behavior

```text
Original query → dense ─────────────────────────────────────┐
Original query → BM25 + weight × BM25(filtered phrases) ────┤
                        → RRF → rerank → MemPursuit → optional rank fusion → context packing
```

- Expansion reads only the original question. The LLM uses the configured model, temperature=0.4, seed=42, and enable_thinking=false, retaining at most 10 phrases.
- DF is computed over searchable memories for the current user and role filter, without cross-user sharing.
- A phrase survives if at least one sub-ngram of length 1..max_n exists and meets the DF threshold. When the integer threshold `int(N * ratio)` is zero, no DF upper bound is imposed.
- Scoring uses the project tokenizer and unigram BM25. No document expansion, stemming, Rust n-gram scoring, or new reranker is introduced; this is a query-side adaptation.
- Embedding/rerank/MemPursuit query inputs and option handling are unchanged. Add does not call the expansion model.
- Each Search uses an immutable SIRA snapshot. There is no global or cross-user expansion cache.
- Conflict branches reuse one expansion of the original question rather than making an LLM call for every rewritten query. Each branch still uses its own query for dense retrieval, original BM25, and reranking.
- No expansion call is made in dense mode, when lexical_weight=0, when SIRA is off, when expansion weight is 0, or for an empty corpus.
- Network, timeout, or parsing failures, empty expansion, or complete filtering fall back to the original lexical score without retrying.

With diagnostics enabled, `sira` contains enabled, active, status, expansion_ms, proposed_count, and kept_count. Errors record only the exception type, not API responses or keys. Status can be disabled, not_run (lexical branch not entered), zero_weight, empty_memories, unconfigured, error, no_phrases, all_filtered, or ok.

## References and evaluation

See [third-party notices](../../THIRD_PARTY_NOTICES.md) for pinned sources and [evaluation](../evaluation.md) for reproduction steps.
