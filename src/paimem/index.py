from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .schema import Chunk


class FaissChunkIndex:
    def __init__(self, index: object, chunks: list[Chunk], metadata: dict[str, object]) -> None:
        self.index = index
        self.chunks = chunks
        self.metadata = metadata

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, embedding_model: str) -> "FaissChunkIndex":
        import faiss

        vectors = embedder.encode_documents([chunk.text for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedder returned an invalid document vector matrix")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index=index, chunks=chunks, metadata={"embedding_model": embedding_model, "dimension": int(vectors.shape[1])})

    def search(self, question: str, embedder: Embedder, top_k: int) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = embedder.encode_queries([question])
        scores, positions = self.index.search(np.ascontiguousarray(query_vector, dtype=np.float32), min(top_k, len(self.chunks)))
        return [(self.chunks[int(position)], float(score)) for score, position in zip(scores[0], positions[0]) if position >= 0]

    def save(self, output_dir: str | Path) -> None:
        import faiss

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in self.chunks:
                stream.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, input_dir: str | Path) -> "FaissChunkIndex":
        import faiss

        directory = Path(input_dir)
        index = faiss.read_index(str(directory / "index.faiss"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(json.loads(line)) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if index.ntotal != len(chunks):
            raise ValueError("Index and chunk metadata counts do not match")
        return cls(index=index, chunks=chunks, metadata=metadata)
