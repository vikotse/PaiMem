from __future__ import annotations

import re

from .schema import Chunk, Document


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize English words and individual CJK characters for stable chunk sizing."""
    return TOKEN_PATTERN.findall(text)


def chunk_document(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    tokens = tokenize(document.text)
    if not tokens:
        return []

    result: list[Chunk] = []
    start = 0
    ordinal = 0
    step = chunk_size - overlap
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        text = " ".join(tokens[start:end])
        result.append(
            Chunk(
                id=f"{document.id}::chunk::{ordinal}",
                document_id=document.id,
                text=text,
                ordinal=ordinal,
                title=document.title,
                metadata={**document.metadata, "token_start": start, "token_end": end},
            )
        )
        if end == len(tokens):
            break
        ordinal += 1
        start += step
    return result


def chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap))
    if not chunks:
        raise ValueError("No non-empty chunks were produced from the documents")
    return chunks
