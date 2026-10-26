# Multi-channel rank fusion

[English](RANK_FUSION.md) | [简体中文](RANK_FUSION.zh-CN.md) | [README](../../README.md)

The default profile disables this strategy. After MemPursuit selection and before neighbor expansion and packing, this strategy merges and reorders specified prefixes of two candidate groups. It sums weighted rank scores from original-text vectors, BM25, RRF, context-aware reranking, and MP selection order. No re-ingestion or additional model calls are needed.

| `[rank_fusion]` field | Default profile | Description |
|---|---:|---|
| `enabled` | `false` | Enable fusion |
| `dense_weight` | `0.35` | Original-vector rank weight |
| `rerank_weight` | `0.65` | Reranker rank weight |
| `bm25_weight` | `0` | Rank weight for positive-BM25 candidates |
| `rrf_weight` | `0` | Full initial-RRF rank weight |
| `mp_weight` | `0` | Full MP selection-order weight |
| `mp_limit` | `100` | MP-selected prefix participating in fusion |
| `remaining_limit` | `300` | Unselected eligible prefix in reranker order |

Both limits accept `-1` for all candidates and `0` for none, and are capped by availability. If both are zero, the queue is unchanged and diagnostics report `zero_limits`. These limits do not change MP's optimization pool or selection budget and are not final output quotas.

The five weights are normalized together. BM25, RRF, and MP default to zero, leaving 35% vector rank and 65% reranker rank. Only positive BM25 candidates have ranks; missing ranks score zero. RRF uses its complete ranking before the reranking candidate cutoff. MP uses selection order before prefix limits; unselected chunks score zero. If a retrieval mode did not execute BM25/RRF, those channels remain absent with zero scores: vector ranks are not substituted, and missing-channel weights are not redistributed.

Weights must be finite, nonnegative, and have a positive sum. Normalization makes 35/65 equivalent to 0.35/0.65. Rank r in a channel receives `1 - (r-1)/max(N-1,1)`, where N is the channel's full ranking length, not the selected prefix length. Vector rank uses original-query document similarity, not NEAR or option-blended ranks. Reranker ranks are saved before threshold filtering, but threshold-rejected candidates cannot reenter the queue.

Participating candidates move to the front in descending fused-score order. Ties preserve MP selection order followed by the remaining reranker order. Nonparticipants retain relative order at the tail for backfill after expansion and deduplication. Output expansion, seed counts, packing, and top_k remain independently configurable.

## Configuration

Edit the existing `[rank_fusion]` section in `configs/config.local.toml`. This example enables fusion; other fields match the [default profile](../../configs/default.toml):

```toml
[rank_fusion]
enabled = true
dense_weight = 0.35
rerank_weight = 0.65
bm25_weight = 0
rrf_weight = 0
mp_weight = 0
mp_limit = 100
remaining_limit = 300
```

Also enable `mempursuit.enabled=true` and `reranker.enabled=true`. Environment variables can override TOML; mappings appear in the configuration comments. The launcher rejects fusion configurations missing these dependencies.

```bash
bash start_svr.sh --check
bash start_svr.sh
```

### Runtime updates

Runtime field names use the `rank_fusion_` prefix. Updates affect the current process and do not modify TOML; restarting restores startup settings.

POST this JSON to `/config`. See [configuration](../configuration.md) for management-endpoint access:

```json
{
  "rank_fusion_enabled": true,
  "rank_fusion_dense_weight": 0.35,
  "rank_fusion_rerank_weight": 0.65,
  "rank_fusion_bm25_weight": 0,
  "rank_fusion_rrf_weight": 0,
  "rank_fusion_mp_weight": 0,
  "rank_fusion_mp_limit": 100,
  "rank_fusion_remaining_limit": 300
}
```

For 60% reranker + 20% RRF + 20% MP, submit `{"rank_fusion_dense_weight":0,"rank_fusion_rerank_weight":0.6,"rank_fusion_bm25_weight":0,"rank_fusion_rrf_weight":0.2,"rank_fusion_mp_weight":0.2}`. The sum of all five weights must remain positive.

Or use the CLI:

```bash
.venv/bin/python -m paimem.cli config --base-url http://127.0.0.1:8000 --set rank_fusion_enabled=true
.venv/bin/python -m paimem.cli config --base-url http://127.0.0.1:8000 --set rank_fusion_dense_weight=0.35 --set rank_fusion_rerank_weight=0.65
.venv/bin/python -m paimem.cli config --base-url http://127.0.0.1:8000 --set rank_fusion_mp_limit=50 --set rank_fusion_remaining_limit=150
```

GET `/config` returns the settings. Invalid updates are rejected atomically. A Search and its conflict branches share an immutable fusion-configuration snapshot. Disable fusion with `{"rank_fusion_enabled":false}`.

## Interaction with other stages

The [default profile](../../configs/default.toml) enables hybrid retrieval, context-aware reranking, and context-aware MemPursuit, while leaving rank fusion disabled. It uses 400 reranking candidates, `reranker.max_length=1024`, and `mempursuit.candidates=150`. Enabling fusion changes neither these settings nor the candidate pool; `mp_limit` and `remaining_limit` only control the prefixes participating in fusion.

Set both limits to `-1` to include all eligible candidates. Final output still depends on `top_k`, window expansion, and packing. If MemPursuit or the reranker is disabled at runtime, fusion is skipped with diagnostics `mempursuit_disabled` or `reranker_disabled`, respectively.

For direct Python API use, `configure_store(rank_fusion_enabled=True, ...)` accepts explicit settings that take precedence over environment variables.

## Diagnostics and evaluation

`rank_fusion` diagnostics record configuration, active status, reason, participating MP and remaining counts, total fused count, and backfill count. Each candidate's `channels.rank_fusion` records participation, five channel rank scores, and the fused score. Public `MemoryHit.score` retains its original model score; preserve returned order instead of sorting by that field.

See [evaluation](../evaluation.md) for the evaluation workflow. Hold candidate pools, models, data, and other retrieval settings fixed when comparing fusion configurations.
