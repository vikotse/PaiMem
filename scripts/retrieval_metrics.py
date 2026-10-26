"""Deterministic retrieval metrics on explicit source IDs (no LLM calls)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def unique_order(values):
    return list(dict.fromkeys(values))


def answer_turn_gold(record):
    """Map original LongMemEval has_answer flags to normalized message ordinals."""
    qid = record["question_id"]
    if qid.endswith("_abs"):
        raise ValueError("Abstention questions have no positive retrieval target")
    gold = []
    index = 0
    for sid, session in zip(record["haystack_session_ids"], record["haystack_sessions"], strict=True):
        for turn in session:
            if not str(turn.get("content", turn.get("text", ""))).strip():
                continue
            if turn.get("has_answer") is True:
                if sid not in record["answer_session_ids"]:
                    raise ValueError("Positive answer turn is outside annotated answer sessions")
                gold.append(f"{qid}:message-{index}")
            index += 1
    if not gold:
        raise ValueError("No positive answer-turn annotations")
    return gold


def answer_turn_metrics(chunk_ids, gold, ks=(10, 20)):
    turns = [chunk.rsplit(":chunk-", 1)[0] for chunk in chunk_ids]
    ranking = unique_order(turns)
    if len(ranking) < max(ks):
        raise ValueError("Saved chunk prefix does not cover the requested turn cutoff")
    metrics = ranking_metrics(ranking, gold, ks)
    for k in (10, 20, 100):
        if len(chunk_ids) < k:
            raise ValueError("Saved ranking does not cover chunk budget")
        metrics[f"parent_coverage_in_{k}_chunks"] = len(set(turns[:k]) & set(gold)) / len(set(gold))
    return metrics, ranking


def ranking_metrics(ranking, gold, ks=(10, 20, 100)):
    gold = set(gold)
    if not gold:
        raise ValueError("Queries without positive qrels must be audited, not scored")
    ranking = list(ranking)
    if len(set(ranking)) != len(ranking):
        raise ValueError("Ranking must contain unique document IDs")
    result = {}
    for k in ks:
        if k <= 0:
            raise ValueError("k must be positive")
        relevance = [int(doc in gold) for doc in ranking[:k]]
        hits = sum(relevance)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        result.update({f"recall@{k}": hits / len(gold),
                       f"capped_recall@{k}": hits / min(k, len(gold)),
                       f"ndcg@{k}": dcg / ideal,
                       f"complete@{k}": float(hits == len(gold))})
    return result


def parent_metrics(chunk_order, parent_ids, gold, ks=(10, 20, 100)):
    """Collapse chunks by first appearance; keep chunk-budget coverage separate."""
    parents = [parent_ids[int(i)] for i in chunk_order]
    metrics = ranking_metrics(unique_order(parents), gold, ks)
    for k in ks:
        metrics[f"parent_coverage_in_{k}_chunks"] = len(set(parents[:k]) & set(gold)) / len(set(gold))
    return metrics


def returned_chunk_metrics(returned_ids, chunk_parents, gold, top_k=100):
    """Evaluate actual returned slots, without deduplication/refilling the list.

    Evidence recall counts each source message once. Standard binary chunk NDCG
    inherits relevance from the source message, including all its split chunks.
    This is chunk-level relevance, not an annotation of the exact answer span.
    """
    returned_ids = list(returned_ids)
    gold = set(gold)
    if len(returned_ids) != min(top_k, len(chunk_parents)):
        raise ValueError("Expected the complete fixed-budget return list")
    if len(set(returned_ids)) != len(returned_ids) or not set(returned_ids) <= set(chunk_parents):
        raise ValueError("Returned chunk IDs must be unique and inside the candidate bank")
    if not gold or not gold <= set(chunk_parents.values()):
        raise ValueError("Positive evidence IDs must exist in the bank")
    gold_chunks = {chunk for chunk, parent in chunk_parents.items() if parent in gold}
    cutoffs = tuple(k for k in (10, 20, 100) if k <= top_k)
    if top_k not in cutoffs:
        cutoffs += (top_k,)
    raw = ranking_metrics(returned_ids, gold_chunks, cutoffs)
    metrics = {}
    for k in cutoffs:
        found = {chunk_parents[c] for c in returned_ids[:k]} & gold
        metrics.update({f"evidence_recall@{k}": len(found) / len(gold),
                        f"complete_evidence@{k}": float(found == gold),
                        f"chunk_recall@{k}": raw[f"recall@{k}"],
                        f"capped_chunk_recall@{k}": raw[f"capped_recall@{k}"],
                        f"chunk_ndcg@{k}": raw[f"ndcg@{k}"]})
    return metrics


def overlap_groups(banks):
    """Connected components of banks sharing at least one source session ID."""
    parents = list(range(len(banks)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, bank in enumerate(banks):
        for doc in bank:
            if doc in seen:
                parents[find(i)] = find(seen[doc])
            else:
                seen[doc] = i
    return [find(i) for i in range(len(banks))]


def paired_summary(rows, base, near, group_key="group", repeats=2000):
    if not rows:
        return {"questions": 0, "metrics": {}}
    result = {"questions": len(rows), "groups": len({r[group_key] for r in rows}), "metrics": {}}
    for metric in rows[0]["arms"][base]["metrics"]:
        deltas = np.array([r["arms"][near]["metrics"][metric] - r["arms"][base]["metrics"][metric] for r in rows])
        groups = defaultdict(list)
        for row, delta in zip(rows, deltas):
            groups[row[group_key]].append(delta)
        sums = np.array([sum(v) for v in groups.values()])
        counts = np.array([len(v) for v in groups.values()])
        ci = None
        if len(groups) >= 2:
            sample = np.random.default_rng(20261015).integers(len(groups), size=(repeats, len(groups)))
            draws = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
            ci = np.percentile(draws, [2.5, 97.5]).tolist()
        result["metrics"][metric] = {
            "base": float(np.mean([r["arms"][base]["metrics"][metric] for r in rows])),
            "near": float(np.mean([r["arms"][near]["metrics"][metric] for r in rows])),
            "delta": float(deltas.mean()), "improved": int((deltas > 1e-10).sum()),
            "regressed": int((deltas < -1e-10).sum()), "unchanged": int((np.abs(deltas) <= 1e-10).sum()),
            "cluster_bootstrap_95_ci": ci,
        }
    return result
