from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from .schema import Query


def normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\s+", " ", value)
    return value


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


@dataclass
class AggregateMetrics:
    count: int = 0
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    answer_in_context: float = 0.0
    exact_match: float = 0.0
    f1: float = 0.0
    observed: dict[str, int] = field(default_factory=lambda: {
        "recall": 0,
        "reciprocal_rank": 0,
        "answer_in_context": 0,
        "exact_match": 0,
        "f1": 0,
    })

    def add(self, query: Query, retrieved_doc_ids: list[str], context: str, prediction: str | None) -> dict[str, float | None]:
        self.count += 1
        relevant = set(query.relevant_doc_ids)
        hit_positions = [index + 1 for index, doc_id in enumerate(retrieved_doc_ids) if doc_id in relevant]
        recall = len(set(retrieved_doc_ids) & relevant) / len(relevant) if relevant else None
        mrr = 1.0 / hit_positions[0] if hit_positions else (0.0 if relevant else None)
        answer_in_context = float(normalize_answer(query.answer) in normalize_answer(context)) if query.answer else None
        exact = float(normalize_answer(prediction or "") == normalize_answer(query.answer)) if prediction is not None and query.answer else None
        f1 = token_f1(prediction or "", query.answer) if prediction is not None and query.answer else None
        for field, value in [("recall", recall), ("reciprocal_rank", mrr), ("answer_in_context", answer_in_context), ("exact_match", exact), ("f1", f1)]:
            if value is not None:
                setattr(self, field, getattr(self, field) + value)
                self.observed[field] += 1
        return {"retrieval_recall": recall, "reciprocal_rank": mrr, "answer_in_context": answer_in_context, "exact_match": exact, "token_f1": f1}

    def summary(self) -> dict[str, float | int | None]:
        def average(field: str, total: float) -> float | None:
            observed = self.observed[field]
            return total / observed if observed else None

        return {
            "count": self.count,
            "retrieval_recall_at_k": average("recall", self.recall),
            "mrr": average("reciprocal_rank", self.reciprocal_rank),
            "answer_in_context_rate": average("answer_in_context", self.answer_in_context),
            "exact_match": average("exact_match", self.exact_match),
            "token_f1": average("f1", self.f1),
        }
