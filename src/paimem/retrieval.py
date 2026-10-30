"""Shared chunk retrieval, before result expansion and packing."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .chunking import TOKEN_PATTERN
from .embeddings import Embedder, timed_embedding_lock, timed_operation, timing_stage
from .rank_fusion import RankFusionConfig, fuse
from .mempursuit import MemPursuitConfig, select as select_memories
from .reranking import Reranker


def lexical_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if any(character.isalnum() or character == "_" for character in token)
    ]


def bm25_scores(documents: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized = [lexical_tokens(document) for document in documents]
    query_terms = set(lexical_tokens(query))
    scores = np.zeros(len(documents), dtype=np.float32)
    if not documents or not query_terms:
        return scores

    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    average_length = average_length or 1.0
    document_frequency = Counter(
        term for tokens in tokenized for term in query_terms.intersection(tokens)
    )
    document_count = len(documents)
    for index, tokens in enumerate(tokenized):
        frequencies = Counter(tokens)
        length_normalization = 1.0 - b + b * len(tokens) / average_length
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            scores[index] += inverse_document_frequency * (
                frequency * (k1 + 1.0) / (frequency + k1 * length_normalization)
            )
    return scores


@dataclass
class ChunkRanking:
    positions: np.ndarray
    scores: np.ndarray
    reranked_count: int
    diagnostics: dict | None = None


@timed_operation('retrieval.total')
def retrieve_chunks(
    rows: Sequence, names: Mapping[str, str], query: str, top_k: int, *,
    embedder: Embedder, embedding_lock, render_content: Callable,
    retrieval_mode: str = "hybrid", lexical_weight: float = 0.5,
    rrf_k: int = 60, option_view_weight: float = 1.0,
    reranker: Reranker | None = None, rerank_candidates: int = 200,
    rerank_score_threshold: float | None = None,
    options: list[str] | None = None, diagnostics: bool = False, near_context=None,
    mempursuit_config: MemPursuitConfig | None = None, sira_context=None,
    rank_fusion_config: RankFusionConfig | None = None,
    rerank_small2big_enabled: bool = False,
    mempursuit_small2big_enabled: bool = False,
) -> ChunkRanking:
    """Recall, optionally rerank/filter, then select complementary evidence.

    Positions refer to the supplied rows. MemPursuit changes their order while
    retaining upstream scores and eligible tails for subsequent expansion/packing.
    """
    if top_k <= 0 or rerank_candidates <= 0:
        raise ValueError("Retrieval budgets must be positive")
    for value in (rerank_small2big_enabled, mempursuit_small2big_enabled):
        if not isinstance(value, bool):
            raise ValueError("small2big stage switches must be booleans")
    stage_detail = {
        "rerank_small2big_enabled": rerank_small2big_enabled,
        "mempursuit_small2big_enabled": mempursuit_small2big_enabled,
        "rerank_active": False, "mempursuit_active": False,
        "before": 1, "after": 1,
    }
    rank_fusion_config = rank_fusion_config or RankFusionConfig()
    fusion_detail = {**rank_fusion_config.settings(), "active": False, "reason": "disabled"}
    fusion_data = None
    if rank_fusion_config.enabled:
        fusion_detail["reason"] = ("reranker_disabled" if reranker is None else
                                   "mempursuit_disabled" if not mempursuit_config or not mempursuit_config.enabled else
                                   "empty_candidates")
    if not rows:
        fusion_detail["reason"] = "empty_memories"
        detail = {"schema_version": 1, "stored_count": 0, "top_k": top_k,
                  "rerank_candidate_count": 0, "candidates": [], "final_results": [],
                  "near": {"active": False}, "small2big_stages": stage_detail,
                  "rank_fusion": fusion_detail,
                  "mempursuit": {**_pursuit_settings(mempursuit_config),
                                 "reason": "empty_memories"}} if diagnostics else None
        return ChunkRanking(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0, detail)
    retrieval_query = query
    # Request-local text views: candidates keep their identities and original
    # recall vectors. Neighbors are drawn from the supplied same-session rows.
    context_texts = {}
    def context_text(position):
        position = int(position)
        if position not in context_texts:
            context_texts[position] = "\n".join(
                render_content(rows[i], names)
                for i in range(max(0, position - 1), min(len(rows), position + 2))
                if rows[i]["session_id"] == rows[position]["session_id"]
            )
        return context_texts[position]
    query_views = [query]
    if options:
        retrieval_query += "\nOptions:\n" + "\n".join(options)
        if retrieval_mode == "enhanced":
            # retrieval_mode 为 "enhanced" 则采用 query + options 进行检索
            query_views.extend(f"{query}\nCandidate answer: {option}" for option in options)
        else:
            # 否则仅采用 query 进行检索
            query_views = [retrieval_query]
    # retrieval query embeddings
    with timed_embedding_lock(embedding_lock, 'query_embedding', len(query_views), embedder=embedder):
        query_vectors = np.asarray(embedder.encode_queries(query_views), dtype=np.float32)
    if query_vectors.ndim != 2 or query_vectors.shape[0] != len(query_views):
        raise ValueError("Embedder returned an invalid query vector matrix")
    dimension = int(rows[0]["dimension"])
    if query_vectors.shape[1] != dimension:
        raise ValueError("Query embedding dimension does not match stored memories")

    # Reuse stored document vectors; only the query is embedded per retrieval.
    matrix = np.vstack(
        [np.frombuffer(row["embedding"], dtype=np.float32, count=dimension) for row in rows]
    )
    view_scores = matrix @ query_vectors.T
    raw_view_scores = view_scores
    fusion_dense_order = np.argsort(-raw_view_scores[:, 0], kind="stable") if rank_fusion_config.enabled else None
    if near_context is not None:
        view_scores = near_context[0].score(query_vectors)
    # 计算 embedding 检索相似度
    dense_scores = view_scores[:, 0]
    # retrieval_mode 默认为 "hybrid"
    if retrieval_mode == "enhanced" and len(query_views) > 1:
        option_scores = np.max(view_scores[:, 1:], axis=1)
        dense_scores = dense_scores + option_view_weight * np.maximum(
            option_scores - dense_scores,
            0.0,
        )
    result_scores = dense_scores
    lexical = None
    lexical_positions = np.empty(0, dtype=np.int64)
    # embedding 检索结果排序
    positions = np.argsort(-dense_scores, kind="stable")
    if retrieval_mode in {"hybrid", "enhanced"} and lexical_weight > 0:
        # 计算 bm25 相似度
        documents = [f"[{names[row['role']]}] {row['content']}"
                     if row["role"] in names else str(row["content"]) for row in rows]
        lexical = bm25_scores(documents, retrieval_query)
        if sira_context is not None:
            lexical = sira_context.score(documents, lexical)
        fused = np.zeros(len(rows), dtype=np.float32)
        # 计算 rrf 混排得分
        # 1. embedding 检索结果排名分
        for rank, position in enumerate(positions):
            fused[int(position)] += 1.0 / (rrf_k + rank + 1)
        lexical_positions = np.flatnonzero(lexical > 0)
        lexical_positions = lexical_positions[
            np.argsort(-lexical[lexical_positions], kind="stable")
        ]
        # 2. 加上 bm25 检索结果排名分
        for rank, position in enumerate(lexical_positions):
            fused[int(position)] += lexical_weight / (rrf_k + rank + 1)
        # 计算 rrf 排名：先按 fused 从大到小排序；当 fused 相同时，再按 dense_scores 从大到小排序，最后返回排序后的下标
        positions = np.lexsort((-dense_scores, -fused))
        # 混排分
        result_scores = fused

    # Snapshot the full pre-rerank ordering before the optional candidate cap.
    fused_positions = positions.copy() if diagnostics or rank_fusion_config.enabled else None
    fusion_rrf_order = fused_positions if lexical is not None else np.empty(0, dtype=np.int64)
    fused_scores = result_scores.copy() if diagnostics else None
    rerank_positions = None
    rerank_scores = None
    reranked_count = 0
    # 进行重排序
    if reranker is not None:
        candidate_count = min(len(rows), max(top_k, rerank_candidates))
        reranked_count = candidate_count
        candidate_positions = np.asarray(positions[:candidate_count], dtype=np.int64)
        candidate_documents = [
            context_text(position) if rerank_small2big_enabled else render_content(rows[int(position)], names)
            for position in candidate_positions
        ]
        stage_detail["rerank_active"] = rerank_small2big_enabled
        with timing_stage('reranker.score', inputs=candidate_count):
            rerank_scores = np.asarray(
                reranker.score(query, candidate_documents),
                dtype=np.float32,
            ).reshape(-1)
            if rerank_scores.shape != (candidate_count,) or not np.all(np.isfinite(rerank_scores)):
                raise ValueError("Reranker returned invalid scores")
        rerank_order = np.lexsort(
            (-result_scores[candidate_positions], -rerank_scores)
        )
        positions = candidate_positions[rerank_order]
        if diagnostics or rank_fusion_config.enabled:
            rerank_positions = positions.copy()
        result_scores = np.full(len(rows), -np.inf, dtype=np.float32)
        result_scores[candidate_positions] = rerank_scores
        # Filter seeds before expansion/packing; their neighbors retain context.
        # Keep the full rerank snapshot above for diagnostics, including rejected seeds.
        if rerank_score_threshold is not None:
            positions = positions[result_scores[positions] >= rerank_score_threshold]

    pursuit_detail = _pursuit_settings(mempursuit_config) if diagnostics else None
    pursuit_channels = {}
    if mempursuit_config is not None and mempursuit_config.enabled:
        if len(positions):
            pool = positions[:max(top_k, mempursuit_config.candidates)]
            budget = min(top_k, len(pool))
            # Reuse the first existing query view: raw query in enhanced mode,
            # or the existing options-appended query in other modes.
            pursuit_matrix = matrix[pool]
            if mempursuit_small2big_enabled:
                with timed_embedding_lock(embedding_lock, 'mempursuit_embedding', len(pool), embedder=embedder):
                    pursuit_matrix = np.asarray(embedder.encode_documents(
                        [context_text(position) for position in pool]), dtype=np.float32)
                if pursuit_matrix.shape != (len(pool), dimension) or not np.all(np.isfinite(pursuit_matrix)):
                    raise ValueError("Embedder returned invalid small2big document vectors")
                stage_detail["mempursuit_active"] = True
            pursuit = select_memories(query_vectors[0], pursuit_matrix, budget,
                                      ridge=mempursuit_config.ridge)
            selected = pool[np.asarray(pursuit.selected_indices, dtype=np.int64)]
            tail = positions[~np.isin(positions, selected)]
            positions = np.concatenate((selected, tail))
            if rank_fusion_config.enabled and reranker is not None:
                positions, fusion_data = fuse(selected, tail, fusion_dense_order, rerank_positions,
                                              len(rows), rank_fusion_config,
                                              bm25_order=lexical_positions, rrf_order=fusion_rrf_order)
                fusion_detail.update(active=bool(fusion_data["weighted_count"]),
                                     reason="applied" if fusion_data["weighted_count"] else "zero_limits",
                                     **{key: fusion_data[key] for key in
                                        ("mp_count", "remaining_count", "weighted_count", "backfill_count")})
            if diagnostics:
                pursuit_detail.update(active=True, pool_count=len(pool), budget=budget,
                                      selected_count=len(selected), backfill_count=len(tail),
                                      stop_reason=pursuit.stop_reason, query_view=query_views[0])
                pool_ranks = {int(position): rank for rank, position in enumerate(pool, 1)}
                # Tails include eligible candidates outside the optimization pool.
                # They are used only when existing packing consumes selected seeds.
                for position in positions:
                    pursuit_channels[int(position)] = {
                        "rank": None, "score": None, "coefficient": None,
                        "selection_score": None, "backfill_candidate": True,
                        "pool_rank": pool_ranks.get(int(position)),
                    }
                for rank, (position, step, coefficient) in enumerate(
                    zip(selected, pursuit.steps, pursuit.weights), 1
                ):
                    pursuit_channels[int(position)].update(
                        rank=rank, score=(len(selected) - rank + 1) / len(selected),
                        coefficient=float(coefficient), selection_score=float(step.selection_score),
                        residual_norm=float(step.residual_norm), squared_error=float(step.squared_error),
                        squared_error_reduction=float(step.squared_error_reduction),
                        relative_improvement=float(step.relative_improvement),
                        backfill_candidate=False,
                    )
        elif diagnostics:
            pursuit_detail["reason"] = "empty_candidates"

    if not diagnostics:
        return ChunkRanking(positions, result_scores, reranked_count)

    def ranks(order):
        values = [None] * len(rows)
        for rank, position in enumerate(order, 1):
            values[int(position)] = rank
        return values

    query_ranks = ranks(np.argsort(-view_scores[:, 0], kind="stable"))
    raw_query_ranks = ranks(np.argsort(-raw_view_scores[:, 0], kind="stable"))
    blended_ranks = ranks(np.argsort(-dense_scores, kind="stable"))
    option_ranks = [ranks(np.argsort(-view_scores[:, view], kind="stable"))
                    for view in range(1, view_scores.shape[1])]
    lexical_ranks = ranks(lexical_positions) if lexical is not None else None
    fused_ranks = ranks(fused_positions)
    reranked_ranks = ranks(rerank_positions) if rerank_positions is not None else None
    candidates = []
    for position in fused_positions:
        i = int(position)
        channels = {
            "dense": {"score": float(raw_view_scores[i, 0]), "rank": raw_query_ranks[i]},
            "dense_blended": {"score": float(dense_scores[i]), "rank": blended_ranks[i]},
            "fusion": {"score": float(fused_scores[i]), "rank": fused_ranks[i]},
        }
        if near_context is not None:
            channels["near"] = {"score": float(view_scores[i, 0]), "rank": query_ranks[i]}
        if lexical is not None:
            channels["bm25"] = {"score": float(lexical[i]), "rank": lexical_ranks[i]}
        for view, ranking in enumerate(option_ranks, 1):
            channels[f"option_dense_{view}"] = {"score": float(view_scores[i, view]), "rank": ranking[i]}
        if reranked_ranks is not None:
            channels["reranker"] = {"score": float(result_scores[i]) if reranked_ranks[i] else None,
                                    "rank": reranked_ranks[i]}
        if mempursuit_config is not None and mempursuit_config.enabled:
            channels["mempursuit"] = pursuit_channels.get(i, {
                "rank": None, "score": None, "coefficient": None, "selection_score": None,
                "backfill_candidate": False, "pool_rank": None,
            })
        if fusion_data is not None:
            channels["rank_fusion"] = {
                "participated": i in fusion_data["weighted_positions"],
                "dense_rank_score": float(fusion_data["dense"][i]),
                "rerank_rank_score": float(fusion_data["rerank"][i]),
                "bm25_rank_score": float(fusion_data["bm25"][i]),
                "rrf_rank_score": float(fusion_data["rrf"][i]),
                "mp_rank_score": float(fusion_data["mp"][i]),
                "score": float(fusion_data["scores"][i]) if i in fusion_data["weighted_positions"] else None,
            }
        candidates.append({"id": str(rows[i]["id"]), "content": render_content(rows[i], names),
                           "channels": channels, "returned_rank": None})
    detail = {"schema_version": 1, "stored_count": len(rows), "top_k": top_k,
              "retrieval_mode": retrieval_mode,
              "fusion_method": "rrf" if lexical is not None else "dense",
              "rrf_k": rrf_k if lexical is not None else None,
              "lexical_weight": lexical_weight if lexical is not None else None,
              "option_views": {f"option_dense_{view}": query_views[view]
                               for view in range(1, len(query_views))},
              "rerank_candidate_count": len(rerank_positions) if rerank_positions is not None else 0,
              "rerank_score_threshold": rerank_score_threshold,
              "candidates": candidates,
              "mempursuit": pursuit_detail,
              "rank_fusion": fusion_detail,
              "small2big_stages": stage_detail,
              "final_results": []}
    if sira_context is not None:
        detail["sira"] = dict(sira_context.detail)
    detail["near"] = {"active": near_context is not None}
    if near_context is not None:
        bank, cache_hit = near_context
        detail["near"].update(cache_hit=cache_hit, build_seconds=bank.build_seconds,
                              bank_bytes=bank.nbytes, scales=bank.scales,
                              effective_layers=len(bank.tables))
    return ChunkRanking(positions, result_scores, reranked_count, detail)


def _pursuit_settings(config: MemPursuitConfig | None) -> dict:
    config = config or MemPursuitConfig()
    return {**config.settings(), "enabled": config.enabled, "active": False,
            "pool_count": 0, "budget": 0, "selected_count": 0, "backfill_count": 0}
