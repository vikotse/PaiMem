from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Document, Query


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl_documents(path: str | Path) -> list[Document]:
    documents = []
    for row in _read_jsonl(path):
        documents.append(Document(id=str(row["id"]), text=str(row["text"]), title=str(row.get("title", "")), metadata=dict(row.get("metadata", {}))))
    return documents


def load_jsonl_queries(path: str | Path) -> list[Query]:
    queries = []
    for row in _read_jsonl(path):
        queries.append(
            Query(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row.get("answer", "")),
                relevant_doc_ids=tuple(map(str, row.get("relevant_doc_ids", []))),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return queries


def convert_hotpotqa(input_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Convert official HotpotQA JSON (context + supporting_facts) into project JSONL files."""
    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    documents: dict[str, Document] = {}
    queries: list[Query] = []
    for record in records:
        relevant_titles = {title for title, _ in record.get("supporting_facts", [])}
        for title, sentences in record["context"]:
            doc_id = f"hotpot::{title}"
            documents.setdefault(doc_id, Document(id=doc_id, title=title, text=" ".join(sentences), metadata={"dataset": "hotpotqa"}))
        queries.append(
            Query(
                id=str(record["_id"]),
                question=str(record["question"]),
                answer=str(record["answer"]),
                relevant_doc_ids=tuple(f"hotpot::{title}" for title in relevant_titles),
                metadata={"dataset": "hotpotqa", "level": record.get("level", "")},
            )
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    docs_path = destination / "documents.jsonl"
    questions_path = destination / "questions.jsonl"
    with docs_path.open("w", encoding="utf-8") as stream:
        for document in documents.values():
            stream.write(json.dumps({"id": document.id, "title": document.title, "text": document.text, "metadata": document.metadata}, ensure_ascii=False) + "\n")
    with questions_path.open("w", encoding="utf-8") as stream:
        for query in queries:
            stream.write(json.dumps({"id": query.id, "question": query.question, "answer": query.answer, "relevant_doc_ids": query.relevant_doc_ids, "metadata": query.metadata}, ensure_ascii=False) + "\n")
    return docs_path, questions_path
