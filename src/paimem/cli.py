from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from .chunking import chunk_documents
from .datasets import convert_hotpotqa, load_jsonl_documents, load_jsonl_queries
from .embeddings import build_embedder, embedding_options
from .evaluation import AggregateMetrics
from .generation import OpenAIAnswerGenerator
from .index import FaissChunkIndex
from .pipeline import PaiMemRAG


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Runtime-tunable retrieval switches exposed by the service's /config endpoint.
RUNTIME_SETTINGS: dict[str, type] = {
    "retrieval_mode": str,
    "rrf_k": int,
    "lexical_weight": float,
    "result_window": int,
    "result_window_seed_k": int,
    "pack_span": int,
    "rerank_candidates": int,
    "reranker_enabled": bool,
    "temporal_enrichment": bool,
    "inject_add_timestamp": bool,
    "rank_fusion_enabled": bool,
    "rank_fusion_dense_weight": float,
    "rank_fusion_rerank_weight": float,
    "rank_fusion_bm25_weight": float,
    "rank_fusion_rrf_weight": float,
    "rank_fusion_mp_weight": float,
    "rank_fusion_mp_limit": int,
    "rank_fusion_remaining_limit": int,
    "mempursuit_enabled": bool,
    "rerank_small2big_enabled": bool,
    "mempursuit_small2big_enabled": bool,
    "sira_enabled": bool,
    "long_question_enabled": bool,
}


def add_embedding_args(parser: argparse.ArgumentParser, default_model: str | None = None) -> None:
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", default_model))
    parser.add_argument("--device", default=os.environ.get("EMBEDDING_DEVICE") or None, help="SentenceTransformer device, e.g. cuda or cpu")


def build_generator(kind: str, chat_model: str):
    if kind == "none":
        return None
    return OpenAIAnswerGenerator(model=chat_model)


def cmd_index(args: argparse.Namespace) -> None:
    documents = load_jsonl_documents(args.documents)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    embedder = build_embedder(args.embedding_model, args.device, os.environ.get("EMBEDDING_QUERY_PREFIX"), **embedding_options())
    index = FaissChunkIndex.build(chunks, embedder, embedder.model_name)
    index.metadata.update({"embedding_identity": embedder.identity, "embedding_backend": os.environ.get("EMBEDDING_BACKEND", "local").strip().lower()})
    index.metadata.update({"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "document_count": len(documents)})
    index.save(args.output)
    print(json.dumps({"index": str(args.output), "documents": len(documents), "chunks": len(chunks), **index.metadata}, ensure_ascii=False, indent=2))


def load_rag(index_dir: str, embedding_model: str | None, device: str | None, generator: str, chat_model: str) -> PaiMemRAG:
    index = FaissChunkIndex.load(index_dir)
    model = embedding_model or str(index.metadata["embedding_model"])
    options = embedding_options()
    if "EMBEDDING_BACKEND" not in os.environ:
        options["backend"] = index.metadata.get("embedding_backend", "local")
    embedder = build_embedder(model, device, os.environ.get("EMBEDDING_QUERY_PREFIX"), **options)
    identity = index.metadata.get("embedding_identity")
    if identity is not None and identity != embedder.identity:
        raise ValueError("Configured embedding model does not match the index; rebuild the index")
    if identity is None and (options["backend"].strip().lower() != "local" or model != index.metadata["embedding_model"]):
        raise ValueError("Legacy index requires its original local embedding model; rebuild the index")
    return PaiMemRAG(index=index, embedder=embedder, generator=build_generator(generator, chat_model))


def serialise_result(result) -> dict[str, object]:
    return {
        "question": result.question,
        "answer": result.answer,
        "context": result.context_text,
        "retrieved": [
            {"chunk_id": chunk.id, "document_id": chunk.document_id, "title": chunk.title, "score": score, "text": chunk.text}
            for chunk, score in result.contexts
        ],
    }


def cmd_ask(args: argparse.Namespace) -> None:
    rag = load_rag(args.index, args.embedding_model, args.device, args.generator, args.chat_model)
    print(json.dumps(serialise_result(rag.ask(args.question, args.top_k)), ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    rag = load_rag(args.index, args.embedding_model, args.device, args.generator, args.chat_model)
    queries = load_jsonl_queries(args.questions)
    metrics = AggregateMetrics()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for query in queries:
            result = rag.ask(query.question, args.top_k)
            row = serialise_result(result)
            row.update({"query_id": query.id, "gold_answer": query.answer, "relevant_doc_ids": query.relevant_doc_ids})
            row["metrics"] = metrics.add(query, [chunk.document_id for chunk, _ in result.contexts], result.context_text, result.answer)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics.summary(), ensure_ascii=False, indent=2))


def cmd_prepare_hotpot(args: argparse.Namespace) -> None:
    docs_path, questions_path = convert_hotpotqa(args.input, args.output_dir)
    print(json.dumps({"documents": str(docs_path), "questions": str(questions_path)}, ensure_ascii=False, indent=2))


def parse_setting(raw: str) -> tuple[str, object]:
    """Parse a KEY=VALUE switch; value typing follows the /config whitelist."""
    key, separator, value = raw.partition("=")
    if not separator or key not in RUNTIME_SETTINGS:
        raise ValueError(
            f"Invalid setting {raw!r}; expected KEY=VALUE with KEY one of: {', '.join(sorted(RUNTIME_SETTINGS))}"
        )
    text = value.strip()
    expected = RUNTIME_SETTINGS[key]
    if expected is bool:
        if text.lower() not in {"true", "false"}:
            raise ValueError(f"{key} expects true or false, got {value!r}")
        return key, text.lower() == "true"
    if expected is int:
        return key, int(text)
    if expected is float:
        return key, float(text)
    return key, text


def request_config(base_url: str, api_key: str | None, payload: dict[str, object] | None = None) -> dict[str, object]:
    url = base_url.rstrip("/") + "/config"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST" if payload is not None else "GET"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def cmd_config(args: argparse.Namespace) -> None:
    payload: dict[str, object] | None = None
    try:
        if args.set:
            payload = {}
            for raw in args.set:
                key, value = parse_setting(raw)
                payload[key] = value
        print(json.dumps(request_config(args.base_url, args.api_key, payload), ensure_ascii=False, indent=2))
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"error: HTTP {error.code} from {args.base_url}/config: {detail}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="PaiMem benchmark runner")
    commands = root.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Chunk documents, embed them, and build a FAISS index")
    index.add_argument("--documents", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--chunk-size", type=int, default=320)
    index.add_argument("--chunk-overlap", type=int, default=60)
    add_embedding_args(index)
    index.set_defaults(func=cmd_index)

    ask = commands.add_parser("ask", help="Retrieve context and optionally generate an answer")
    ask.add_argument("--index", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--generator", choices=["none", "openai"], default="none")
    ask.add_argument("--chat-model", default="gpt-4o-mini")
    add_embedding_args(ask)
    ask.set_defaults(func=cmd_ask)

    evaluate = commands.add_parser("evaluate", help="Run retrieval or end-to-end evaluation")
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--generator", choices=["none", "openai"], default="none")
    evaluate.add_argument("--chat-model", default="gpt-4o-mini")
    add_embedding_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    hotpot = commands.add_parser("prepare-hotpot", help="Convert official HotpotQA JSON to project JSONL")
    hotpot.add_argument("--input", required=True)
    hotpot.add_argument("--output-dir", required=True)
    hotpot.set_defaults(func=cmd_prepare_hotpot)

    config = commands.add_parser("config", help="View or change retrieval switches on a running service")
    config.add_argument("--base-url", default="http://127.0.0.1:8000")
    config.add_argument("--api-key", default=None)
    config.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Switch to apply, e.g. --set pack_span=2 --set lexical_weight=0.3; omit to view current values",
    )
    config.set_defaults(func=cmd_config)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
