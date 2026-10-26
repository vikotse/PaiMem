"""Pure reference implementation for PaiMem rank-fusion regression tests."""
from dataclasses import dataclass, field
import hashlib, json, math
import numpy as np
from paimem.mempursuit import select
from paimem.retrieval import bm25_scores
CHANNELS = ("dense", "bm25", "rrf", "ce", "mp")

def neighbors(rows, i, before, after):
    # Same behavior as the native store for a symmetric window, including order.
    for d in range(1, max(before, after) + 1):
        for j in ([i-d] if d <= before else []) + ([i+d] if d <= after else []):
            if 0 <= j < len(rows) and rows[j]["session_id"] == rows[i]["session_id"]:
                yield j

def pack(order, rows, limit, before, after, seed_k):
    groups, covered = [], set()
    for rank, value in enumerate(order):
        if len(groups) >= limit:
            break
        i = int(value)
        if i in covered:
            continue
        group = [i]
        covered.add(i)
        if rank < seed_k:
            for j in neighbors(rows, i, before, after):
                if j not in covered:
                    covered.add(j)
                    group.append(j)
        groups.append(group)
    return groups

def weights_valid(weights):
    if len(weights) != 5 or any(isinstance(w, bool) or not isinstance(w, (int, float))
                              or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Five finite nonnegative channel weights required")
    if not 0 < sum(weights) < float("inf"):
        raise ValueError("Positive finite total weight required")

def fusion_scores(orders, weights, size):
    """Exact lhshi-wrf rank scaling: full channel length, absent rank = 0."""
    weights_valid(weights)
    result = np.zeros(size, dtype=np.float64)
    # CE, RRF, MP addition order matches the reference implementation exactly.
    for index in (3, 2, 4, 0, 1):
        name, weight = CHANNELS[index], weights[index]
        if not weight:
            continue
        order = np.asarray(orders[name], dtype=np.int64)
        if len(set(order.tolist())) != len(order) or np.any(order < 0) or np.any(order >= size):
            raise ValueError("Channel ranks must be unique valid row positions")
        result[order] += weight / sum(weights) * (1.0 - np.arange(len(order)) / max(len(order) - 1, 1))
    return result

def reorder(order, orders, weights, size):
    scores = fusion_scores(orders, weights, size)
    order = np.asarray(order, dtype=np.int64)
    return order[np.argsort(-scores[order], kind="stable")]

def make_policy(site="baseline", weights=None, pre=None):
    weights = list(weights) if weights is not None else None
    pre = list(pre) if pre is not None else None
    if weights is not None:
        weights_valid(weights)
    if site == "pre_mp" and weights[4]:
        raise ValueError("MP rank is unavailable before MP")
    if pre is not None:
        weights_valid(pre)
        if pre[4]:
            raise ValueError("MP rank is unavailable before MP")
    value = {"site": site, "weights": weights, "pre_weights": pre}
    value["name"] = "baseline" if site == "baseline" else site + "_" + hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    return value

@dataclass
class RetrievalState:
    rows: list
    docs: list[str]
    query_vector: np.ndarray
    context_vectors: np.ndarray
    orders: dict
    selection_cache: dict = field(default_factory=dict)

def run_policy(state, policy):
    """Pure retrieval: no question labels, gold evidence or answer arguments."""
    orders = dict(state.orders)
    ce = np.asarray(orders["ce"], dtype=np.int64)
    pre = policy["weights"] if policy["site"] == "pre_mp" else policy.get("pre_weights")
    downstream = reorder(ce, orders, pre, len(state.rows)) if pre else ce
    pool = downstream[:150]
    key = tuple(map(int, pool))
    if key not in state.selection_cache:
        result = select(state.query_vector, state.context_vectors[pool], min(100, len(pool)), ridge=1e-10)
        state.selection_cache[key] = pool[result.selected_indices]
    chosen = state.selection_cache[key]
    orders["mp"] = chosen
    tail = downstream[~np.isin(downstream, chosen)]
    assembly = np.concatenate((chosen, tail))
    site = policy["site"]
    if site == "selected":
        assembly = np.concatenate((reorder(chosen, orders, policy["weights"], len(state.rows)), tail))
    elif site in {"assembly", "combined"}:
        assembly = reorder(assembly, orders, policy["weights"], len(state.rows))
    groups = pack(assembly, state.rows, 100, 1, 1, 20)
    if site == "final":
        scores = fusion_scores(orders, policy["weights"], len(state.rows))
        groups = sorted(groups, key=lambda group: -scores[group[0]])
    flat = [i for g in groups for i in g]
    if len(flat) != len(set(flat)) or len(groups) > 100:
        raise AssertionError("Invalid packed output")
    return groups, {"mp_pool": pool.tolist(), "selected": chosen.tolist(),
                    "assembly_order": assembly.tolist(), "final_groups": groups}

def build_orders(rows, docs, raw_vectors, query_vector, score_fn):
    dense = np.asarray(raw_vectors @ query_vector, dtype=np.float32)
    dense_order = np.argsort(-dense, kind="stable")
    lexical = bm25_scores([r["content"] for r in rows], score_fn.query_text)
    lex_order = np.flatnonzero(lexical > 0)
    lex_order = lex_order[np.argsort(-lexical[lex_order], kind="stable")]
    fused = np.zeros(len(rows), dtype=np.float32)
    for rank, pos in enumerate(dense_order):
        fused[pos] += 1 / (60 + rank + 1)
    for rank, pos in enumerate(lex_order):
        fused[pos] += .5 / (60 + rank + 1)
    rrf = np.lexsort((-dense, -fused))
    pool = rrf[:400]
    rs = np.asarray(score_fn(pool), dtype=np.float32)
    if rs.shape != (len(pool),) or not np.isfinite(rs).all():
        raise ValueError("Invalid reranker model outputs")
    ce = pool[np.lexsort((-fused[pool], -rs))]
    return dict(dense=dense_order, bm25=lex_order, rrf=rrf, ce=ce), fused, rs
