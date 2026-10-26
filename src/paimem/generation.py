from __future__ import annotations

import os

from .schema import Chunk


SYSTEM_PROMPT = """Answer the question only using the supplied context. If the context does not contain enough evidence, say that the answer cannot be determined from the context. Keep the answer concise."""


def format_context(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for rank, (chunk, score) in enumerate(results, start=1):
        heading = chunk.title or chunk.document_id
        blocks.append(f"[{rank}] {heading} (score={score:.4f})\n{chunk.text}")
    return "\n\n".join(blocks)


class OpenAIAnswerGenerator:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required when --generator openai is used")
        self.client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self.model = model

    def answer(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or ""
