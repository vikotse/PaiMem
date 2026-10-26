# MemPursuit in PaiMem

[English](MEMPURSUIT.md) | [简体中文](MEMPURSUIT.zh-CN.md) | [README](../../README.md)

MemPursuit selects complementary evidence using query-reconstruction residuals. PaiMem uses its fixed-budget NumPy algorithm and supports candidate representations built from same-session context. The default profile enables both selection and context expansion; the baseline profile disables both.

The selector requires no generative LLM calls, training, or re-ingestion. With `mempursuit.small2big_enabled=false`, it reuses stored document vectors. When enabled, expanded candidate texts are embedded at query time, adding embedding computation or requests. Numerical optimization runs on the CPU.

## Source and algorithm

Based on [FengQi-HITSZ/MemPursuit](https://github.com/FengQi-HITSZ/MemPursuit), pinned to [`5138dd0e6f6dbf843692526a44b696a4b9f4bcfb`](https://github.com/FengQi-HITSZ/MemPursuit/tree/5138dd0e6f6dbf843692526a44b696a4b9f4bcfb). `mempursuit.py` adapts the default NumPy, fixed-budget paths in standalone `mempursuit/core.py` and `mempursuit/simplex.py`. Apache-2.0 is retained in `third_party/licenses/MemPursuit-Apache-2.0.txt`.

1. Convert query and candidate vectors to float64 and L2-normalize them.
2. Select the unselected candidate with the largest inner product against the current residual. Break ties in input order.
3. Refit all selected candidates with nonnegative weights summing to 1, minimizing `0.5 * a.T @ (G + ridge * I) @ a - b.T @ a`. The solver uses the upstream active-set KKT method, warm starts, and an `lstsq` fallback for singular matrices.
4. Update the residual until the fixed budget is reached. Preserve selection order and retain selected chunks even if their weights later become zero.

This implementation uses fixed-budget selection and does not include adaptive stopping, Numba acceleration, or Mem0 storage integration. The solver uses the project’s NumPy dependency. Zero or nonfinite vectors, incompatible dimensions, and numerical nonconvergence raise explicit errors.

## Configuration

Edit the existing `[mempursuit]` section in `configs/config.local.toml`. These values match the [default profile](../../configs/default.toml):

```toml
[mempursuit]
enabled = true
candidates = 150
ridge = 1e-10
small2big_enabled = true
```

| Field | Default profile | Description | Runtime field |
|---|---:|---|---|
| `enabled` | `true` | Enable evidence selection | `mempursuit_enabled` |
| `candidates` | `150` | Positive optimization-pool size | Startup only |
| `ridge` | `1e-10` | Finite, nonnegative regularization | Startup only |
| `small2big_enabled` | `true` | Encode candidates with adjacent same-session chunks | `mempursuit_small2big_enabled` |

Context expansion considers one position before and after each candidate in the retrieval sequence, including only chunks from the same session. This window is separate from `context.result_window`, which controls output expansion. Query-time candidate vectors change; stored vectors are unchanged.

Environment variables can override TOML; mappings appear in the configuration comments. After editing the configuration, run:

```bash
bash start_svr.sh --check
bash start_svr.sh
```

At runtime, submit the following to `POST /config` without restarting or re-ingesting:

```json
{"mempursuit_enabled": true, "mempursuit_small2big_enabled": true}
```

Runtime updates affect the current process and do not modify TOML; restarting restores startup settings. `GET /config` returns effective settings. Each search uses a configuration snapshot shared by its conflict-retrieval branches. See [configuration](../configuration.md) for management-endpoint access.

For direct Python API use, `MemPursuitConfig` defaults to `enabled=false`, `candidates=200`, and `ridge=1e-10`, distinct from the launcher profile above. Explicit `configure_store()` arguments take precedence over environment variables.

## Pipeline placement and candidate budget

```text
Dense / NEAR + BM25 → RRF → optional reranker → rerank threshold
  → MemPursuit selected prefix + eligible tail → optional rank fusion → window expansion / packing
```

The optimization pool is the first `max(top_k, mempursuit_candidates)` eligible candidates in upstream order after threshold filtering, bounded by the available candidates and, when enabled, the reranker's candidate range. Here top_k is the internal retrieval budget; conflict branches may over-retrieve when allocating quotas.

The fixed budget is `min(top_k, pool_count)`. The algorithm computes only the required selection prefix and refits weights for the selected set at each step. Diagnostic coefficients therefore describe the solution at the current budget.

To support PaiMem's packing and deduplication, all remaining eligible candidates are appended in their original order, including candidates outside the optimization pool. This tail is for postprocessing backfill and does not enlarge the Gram matrix or QP. This differs from the upstream adapter, which does not append unselected candidates. Threshold-rejected chunks cannot become seeds again, but may appear as neighboring context of eligible seeds under the window rules.

Window expansion can reintroduce similar neighbors. The selector optimizes seeds; the final expanded set is not the paper's unmodified selection output. Disabling the switch bypasses normalization, Gram construction, and QP, preserving upstream order, scores, and model calls.

## Combining strategies

- **NEAR:** can be enabled independently alongside MemPursuit. NEAR affects the pool and upstream order; MemPursuit uses the query vector and either stored document vectors or re-embedded context, depending on `small2big_enabled`; it does not use NEAR residual vectors.
- **Reranker:** optional. Threshold filtering precedes selection; raw relevance scores do not enter the QP directly.
- **Conflict strategy:** each branch selects with its own query vector; the results are merged according to configured quotas and accompanied by conflict explanations.
- **Options:** no extra query vector is generated. `enhanced` uses the existing first original-query vector; other modes use the existing query vector with options appended. Option views can still affect the upstream pool.

## Scores and diagnostics

`MemoryHit.score` retains the original dense/fusion/reranker score and may not decrease in the new result order. Do not sort by this score again, as that would erase MemPursuit's selection order.

When diagnostics are enabled, top-level `mempursuit` records configuration, execution status, pool size, budget, selected count, tail count, stop reason, and query view. Each candidate's `channels.mempursuit` contains:

- `rank`: selection order; null if unselected.
- `score`: upstream-adapter-style order score `(selected_count - rank + 1) / selected_count`, not a relevance probability.
- `coefficient`: final fitted coefficient for the current selected prefix.
- `selection_score`: residual inner product when selected, with residual and error reduction for that step.
- `pool_rank`: position in the optimization pool; null outside it.
- `backfill_candidate`: whether it is an eligible unselected tail candidate. Actual output membership is recorded in `returned_rank` / `returned_ranks`.

Original dense, fusion, NEAR, and reranker diagnostic scores are retained.

## References and evaluation

See [third-party notices](../../THIRD_PARTY_NOTICES.md) for pinned sources and [evaluation](../evaluation.md) for reproduction steps.
