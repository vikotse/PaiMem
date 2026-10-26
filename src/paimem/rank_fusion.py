"""Optional post-MemPursuit rank fusion, before neighbor expansion."""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RankFusionConfig:
    enabled: bool = False
    dense_weight: float = 0.35
    rerank_weight: float = 0.65
    bm25_weight: float = 0.0
    rrf_weight: float = 0.0
    mp_weight: float = 0.0
    mp_limit: int = 100
    remaining_limit: int = 300

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("rank_fusion_enabled must be a boolean")
        for name in ("dense_weight", "rerank_weight", "bm25_weight", "rrf_weight", "mp_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"rank_fusion_{name} must be finite and nonnegative")
        total = sum((self.dense_weight, self.bm25_weight, self.rrf_weight, self.rerank_weight, self.mp_weight))
        if not math.isfinite(total) or total <= 0:
            raise ValueError("rank fusion weights must have a finite positive sum")
        for name in ("mp_limit", "remaining_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError(f"rank_fusion_{name} must be -1 or a nonnegative integer")

    def settings(self):
        return {f"rank_fusion_{key}": value for key, value in asdict(self).items()}


def fuse(selected, remaining, dense_order, rerank_order, size, config, *, bm25_order=(), rrf_order=()):
    """Sort capped prefixes; retain all other eligible candidates for backfill.

    Rank normalization uses complete upstream rankings, including rejected seeds.
    Stable ties retain MP-first assembly order.
    """
    selected, remaining = np.asarray(selected, dtype=np.int64), np.asarray(remaining, dtype=np.int64)
    assembly = np.concatenate((selected, remaining))
    def take(order, limit):
        return order if limit == -1 else order[:limit]
    mp = take(selected, config.mp_limit)
    tail = take(remaining, config.remaining_limit)
    candidates = np.concatenate((mp, tail))
    parts = {}
    for name, order in (("dense", dense_order), ("rerank", rerank_order),
                        ("bm25", bm25_order), ("rrf", rrf_order), ("mp", selected)):
        order = np.asarray(order, dtype=np.int64)
        values = np.zeros(size, dtype=np.float64)
        values[order] = 1.0 - np.arange(len(order)) / max(len(order) - 1, 1)
        parts[name] = values
    total = sum((config.dense_weight, config.bm25_weight, config.rrf_weight,
                 config.rerank_weight, config.mp_weight))
    # Use a fixed addition order to preserve stable floating-point ties.
    scores = np.zeros(size, dtype=np.float64)
    for name in ("rerank", "rrf", "mp", "dense", "bm25"):
        weight = getattr(config, name + "_weight")
        if weight:
            scores += weight / total * parts[name]
    ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
    fallback = assembly[~np.isin(assembly, candidates)]
    return np.concatenate((ordered, fallback)), {
        "mp_count": len(mp), "remaining_count": len(tail), "weighted_count": len(candidates),
        "backfill_count": len(fallback), "weighted_positions": set(map(int, candidates)),
        "scores": scores, **parts,
    }
