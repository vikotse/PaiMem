# NEAR neighborhood-residual retrieval

[English](NEAR.md) | [简体中文](NEAR.zh-CN.md) | [README](../../README.md)

This implementation follows equations 1–4 and Appendix C of [NEAR, arXiv:2610.13314v1](https://arxiv.org/html/2610.13314v1). NEAR uses neighborhood reference directions to construct residual representations that distinguish similar memories. The enhanced scores feed the Dense retrieval channel. The default profile disables this strategy.

## Configuration

Edit the existing sections in `configs/config.local.toml`. This example enables NEAR and retains the other values from the [default profile](../../configs/default.toml):

```toml
[retrieval]
mode = "hybrid"

[near]
enabled = true
layers = 1
weight = 1.0
normalize = false
neighbors = 50
anchors = 16
routing = 4
beta = 10.0
cache_max_users = 4
cache_max_mb = 256.0
```

```bash
bash start_svr.sh --check
bash start_svr.sh
```

No re-ingestion or embedding rebuild is required. NEAR parameters are startup-only; restart the service after changes. `GET /config` reports effective settings; `POST /config` does not update NEAR parameters. See [configuration](../configuration.md) for environment overrides.

| `[near]` field | Default profile | Description |
|---|---:|---|
| `enabled` | `false` | Enhance Dense scores |
| `layers` | `1` | Positive integer residual-layer count |
| `weight` | `1.0` | Finite, nonnegative residual weight α; 0 bypasses NEAR |
| `normalize` | `false` | Normalize residuals used for matching |
| `neighbors` | `50` | Semantic neighbors per memory, excluding itself |
| `anchors` | `16` | Maximum query-reference table entries |
| `routing` | `4` | Reference entries used per query |
| `beta` | `10.0` | Finite, positive routing-softmax inverse temperature |
| `cache_max_users` | `4` | Maximum users in the derived cache |
| `cache_max_mb` | `256.0` | Cache-array budget in MiB, excluding temporary construction memory |

For direct Python API use, `configure_store(near_enabled=True, ...)` accepts explicit parameters that override environment variables.

## Algorithm and boundaries

Each layer finds neighbors in the current normalized memory state and uses their mean unit direction as a reference. Removing its projection produces the residual. The next layer uses normalized residuals, while `normalize` controls the matching view. Spherical K-means clusters input states and stores anchors and the mean unit reference direction within each cluster. Queries form their own residual using a softmax-weighted reference from the most similar anchors.

The score is `original inner product/τ0 + α/L × Σ(layer residual inner product/τlayer)`. Following Appendix C, τ is `sqrt(mean_i Var_j(z_i · z_j))`, including diagonal terms, with a floor of 1e-6. Second moments and covariance avoid constructing a full similarity matrix. Enhanced scores replace the Dense channel before option-view blending, BM25/RRF, optional reranking, thresholds, and context packing.

Implementation differences from the paper:

- Exact NumPy scoring; no HNSW, quantization, or concatenated index. Scores are equivalent to the paper's concatenated inner product.
- Spherical K-means uses farthest-first initialization with seed=42, at most 50 iterations, and removes empty clusters. These initialization and convergence settings are implementation choices.
- Residual norms at or below 1e-7 are treated as zero to avoid amplifying numerical noise. Empty stores return no results; single-memory stores use base matching.
- Neighbors come from the current user's actual searchable memories. Role filtering restricts this set before construction. Overlapping chunks from the same source can be neighbors.
- No text rewriting, document re-embedding, generative LLM calls, or database-schema changes.

User cache keys hash all IDs and vectors in the snapshot. Successful Add invalidates that user's cache; Clear removes all cached entries. External writes or changes in role filtering trigger rebuilding through changed snapshots. Queries hold immutable snapshots, preventing conflict branches from mixing old and new memories. LRU eviction considers both user count and total array bytes. Banks exceeding the cache budget can still be scored but are not retained. Construction is serialized under the cache lock.

The first Search, and the first Search after a memory update, pays the construction cost. Neighbor construction costs O(L·N²·d); similarities are generated in blocks without retaining an N² matrix. The cache limit is not a bound on peak process memory. This implementation rebuilds the derived representation in full; evaluate rebuild latency and peak memory for large stores or frequent writes.

Diagnostics retain original `channels.dense` scores and add `channels.near`. `dense_blended` and `fusion` reflect downstream scoring. The `near` metadata records parameters, cache hits, bank-array bytes, construction time, and per-layer scales. Public `/search` response fields are unchanged.

## References and evaluation

See [third-party notices](../../THIRD_PARTY_NOTICES.md) for pinned sources and [evaluation](../evaluation.md) for reproduction steps.
