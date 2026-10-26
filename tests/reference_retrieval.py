"""Pure provenance and ranking helpers for PaiMem retrieval regression tests."""
from datetime import datetime, timezone
import numpy as np
from paimem.memory_store import SQLiteMemoryStore, split_memory_text
from scripts.retrieval_metrics import unique_order

def source_turns(dataset, raw):
    """Preserve exact source IDs; alignment is asserted against normalized inputs."""
    turns = []
    if dataset == "locomo_refined":
        conversation = raw["conversation"]
        keys = sorted((k for k in conversation if k.startswith("session_") and k[8:].isdigit()), key=lambda k: int(k[8:]))
        for key in keys:
            for turn in conversation[key]:
                content = str(turn.get("text", turn.get("content", ""))).strip()
                if content:
                    turns.append({"text": content, "parent": str(turn.get("dia_id", turn.get("id"))), "session": key})
    else:
        for sid, session in zip(raw["haystack_session_ids"], raw["haystack_sessions"], strict=True):
            for index, turn in enumerate(session):
                content = str(turn.get("content", turn.get("text", ""))).strip()
                if content:
                    turns.append({"text": content, "parent": str(sid), "session": str(sid), "turn": index})
    return turns

def make_bank(dataset, record, raw):
    turns = source_turns(dataset, raw)
    messages = record["messages"]
    if len(turns) != len(messages):
        raise ValueError(f"Source alignment length mismatch: {dataset}/{record['id']}")
    docs, parents, ids = [], [], []
    for index, (turn, message) in enumerate(zip(turns, messages, strict=True)):
        if turn["text"] != str(message["content"]).strip():
            raise ValueError(f"Source text alignment mismatch: {record['id']}/{index}")
        prefix = ""
        if message.get("timestamp") is not None:
            date = datetime.fromtimestamp(message["timestamp"] / 1000, tz=timezone.utc)
            prefix = f"[{date.strftime('%B')} {date.day}, {date.year}] "
        for ci, text in enumerate(split_memory_text(str(message["content"]), 320, 40)):
            docs.append(prefix + text)
            parents.append(turn["parent"])
            ids.append(f"{record['id']}:message-{index}:chunk-{ci}")
    available = set(parents)
    questions = []
    for q in record["questions"]:
        reason = None
        if dataset == "locomo_refined":
            qi = int(q["id"].removeprefix("q-"))
            source = raw["qa"][qi]
            if source["question"] != q["question"]:
                raise ValueError("Question ID alignment mismatch")
            gold = [str(x) for x in source.get("evidence", [])]
            category = str(source.get("category", "unknown"))
        else:
            if raw["question"] != q["question"]:
                raise ValueError("Question text alignment mismatch")
            gold = [str(x) for x in raw["answer_session_ids"]]
            category = raw["question_type"]
            if raw["question_id"].endswith("_abs"):
                reason = "abstention_question"
        gold = unique_order(gold)
        missing = sorted(set(gold) - available)
        if reason is None and not gold:
            reason = "no_positive_qrels"
        if reason is None and missing:
            reason = "gold_source_id_missing_from_bank"
        questions.append({"id": f"{record['id']}:{q['id']}", "question": q["question"],
                          "query": q["question"] + ("\nOptions:\n" + "\n".join(q["options"]) if q.get("options") else ""),
                          "gold": gold, "missing_gold": missing, "excluded_reason": reason, "category": category})
    return docs, parents, ids, questions

def fused_order(dense, lexical):
    """Mirror PaiMem's float32 RRF and tie breaks, including positive BM25 only."""
    order = np.argsort(-dense, kind="stable")
    fused = np.zeros(len(dense), dtype=np.float32)
    for rank, pos in enumerate(order):
        fused[int(pos)] += 1.0 / (60 + rank + 1)
    positions = np.flatnonzero(lexical > 0)
    positions = positions[np.argsort(-lexical[positions], kind="stable")]
    for rank, pos in enumerate(positions):
        fused[int(pos)] += .5 / (60 + rank + 1)
    return np.lexsort((-dense, -fused))

def reconstruct_top100(prefix_ids, bank_ids, session_ids):
    """Reuse the production expansion method; no database/model initialization.

    After expanding 20 seeds, filling from the first 100 unique original IDs is
    sufficient to reach 100 slots. Additional original candidates cannot enter.
    """
    if len(bank_ids) != len(session_ids) or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("Invalid bank provenance")
    limit = min(100, len(bank_ids))
    if len(prefix_ids) != limit or len(set(prefix_ids)) != len(prefix_ids):
        raise ValueError("Saved prefix must contain the first 100 unique chunks")
    positions = {identity: i for i, identity in enumerate(bank_ids)}
    if not set(prefix_ids) <= set(positions):
        raise ValueError("Saved ranking contains out-of-bank IDs")
    store = SQLiteMemoryStore.__new__(SQLiteMemoryStore)
    store.result_window = 1
    store.result_window_seed_k = 20
    ordered = np.array([positions[c] for c in prefix_ids], dtype=np.int64)
    rows = [{"session_id": session} for session in session_ids]
    returned = store._expand_result_positions(ordered, rows, limit)
    if len(returned) != limit or len(set(returned)) != limit:
        raise AssertionError("Expansion did not produce the requested unique slots")
    return [bank_ids[int(i)] for i in returned]
