from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .generation import OpenAIAnswerGenerator, format_context
from .index import FaissChunkIndex
from .schema import Chunk


@dataclass
class RagResult:
    question: str
    contexts: list[tuple[Chunk, float]]
    context_text: str
    answer: str | None


class PaiMemRAG:
    def __init__(self, index: FaissChunkIndex, embedder: Embedder, generator: OpenAIAnswerGenerator | None = None) -> None:
        self.index = index
        self.embedder = embedder
        self.generator = generator

    def ask(self, question: str, top_k: int) -> RagResult:
        contexts = self.index.search(question, self.embedder, top_k)
        context_text = format_context(contexts)
        answer = self.generator.answer(question, context_text) if self.generator else None
        return RagResult(question=question, contexts=contexts, context_text=context_text, answer=answer)
