from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from calendar import monthrange
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
import numpy as np

from .maintenance import reset_memory_database

from .conflict_strategy import ConflictStrategy, JudgmentQueries, validate_ratios
from .chunking import TOKEN_PATTERN
from .embeddings import Embedder
from .sira import SiraConfig, SiraRequest
from .long_question import LongQuestionConfig
from .rank_fusion import RankFusionConfig
from .mempursuit import MemPursuitConfig
from .near import NearCache, NearConfig
from .reranking import Reranker
from .retrieval import bm25_scores, lexical_tokens, retrieve_chunks
from .role_names import RoleNameExtractor, RoleNames, merge_role_names, query_role, role_names_conflict


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryMessage:
    role: str
    content: str
    timestamp: int | None = None


RecallQueue = Literal["semantic", "positive", "negative", "scope", "conflict_summary"]


@dataclass(frozen=True)
class MemoryHit:
    id: str
    content: str
    score: float
    created_at: str
    recall_queue: RecallQueue = "semantic"


class RequestConflictError(ValueError):
    pass


def _shift_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _shift_years(value: datetime, years: int) -> datetime:
    year = value.year + years
    day = min(value.day, monthrange(year, value.month)[1])
    return value.replace(year=year, day=day)


def _format_date(value: datetime) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def _format_month(value: datetime) -> str:
    return f"{value:%B} {value.year}"


def _format_week(value: datetime) -> str:
    start = value - timedelta(days=value.weekday())
    end = start + timedelta(days=6)
    return f"{_format_date(start)} through {_format_date(end)}"


def _relative_time_annotations(content: str, message_time: datetime) -> list[str]:
    rules = (
        (r"\blast night\b", _format_date(message_time - timedelta(days=1))),
        (r"\byesterday\b", _format_date(message_time - timedelta(days=1))),
        (r"\btomorrow\b", _format_date(message_time + timedelta(days=1))),
        (r"\b(?:today|tonight)\b", _format_date(message_time)),
        (r"\blast week\b", _format_week(message_time - timedelta(weeks=1))),
        (r"\bthis week\b", _format_week(message_time)),
        (r"\bnext week\b", _format_week(message_time + timedelta(weeks=1))),
        (r"\blast month\b", _format_month(_shift_months(message_time, -1))),
        (r"\bthis month\b", _format_month(message_time)),
        (r"\bnext month\b", _format_month(_shift_months(message_time, 1))),
        (r"\blast year\b", str(message_time.year - 1)),
        (r"\bthis year\b", str(message_time.year)),
        (r"\bnext year\b", str(message_time.year + 1)),
    )
    annotations: list[str] = []
    seen: set[str] = set()
    for pattern, resolved in rules:
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            phrase = match.group(0).casefold()
            if phrase not in seen:
                annotations.append(f"{match.group(0)} = {resolved}")
                seen.add(phrase)

    quantities = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
    ago_pattern = re.compile(
        r"\b(?P<count>a|an|one|two|three|\d+)\s+"
        r"(?P<unit>day|week|month|year)s?\s+ago\b",
        flags=re.IGNORECASE,
    )
    for match in ago_pattern.finditer(content):
        phrase = match.group(0).casefold()
        if phrase in seen:
            continue
        raw_count = match.group("count").casefold()
        count = quantities.get(raw_count, int(raw_count) if raw_count.isdigit() else 1)
        unit = match.group("unit").casefold()
        if unit == "day":
            resolved_time = message_time - timedelta(days=count)
        elif unit == "week":
            resolved_time = message_time - timedelta(weeks=count)
        elif unit == "month":
            resolved_time = _shift_months(message_time, -count)
        else:
            resolved_time = _shift_years(message_time, -count)
        annotations.append(f"{match.group(0)} = {_format_date(resolved_time)}")
        seen.add(phrase)
    return annotations


def _resolve_relative_time(content: str, message_time: datetime) -> str:
    replacements = dict(
        annotation.split(" = ", 1)
        for annotation in _relative_time_annotations(content, message_time)
    )
    if not replacements:
        return content
    replacements = {phrase.casefold(): value for phrase, value in replacements.items()}
    pattern = r"\b(?:" + "|".join(
        re.escape(phrase) for phrase in sorted(replacements, key=len, reverse=True)
    ) + r")\b"
    return re.sub(pattern, lambda match: replacements[match.group(0).casefold()], content, flags=re.IGNORECASE)


def split_memory_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_tokens")
    matches = list(TOKEN_PATTERN.finditer(text))
    if not matches:
        return []
    chunks: list[str] = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < len(matches):
        end = min(start + chunk_tokens, len(matches))
        chunk = text[matches[start].start() : matches[end - 1].end()].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(matches):
            break
        start += step
    return chunks


class SQLiteMemoryStore:
    """Persistent vector store with per-user memory isolation."""

    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        chunk_tokens: int = 320,
        overlap_tokens: int = 40,
        retrieval_mode: str = "dense",
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        context_messages: int = 0,
        option_view_weight: float = 0.2,
        reranker: Reranker | None = None,
        rerank_candidates: int = 200,
        temporal_enrichment: bool = False,
        result_window: int = 0,
        result_window_seed_k: int = 30,
        pack_span: int = 1,
        resolve_relative_time_on_add: bool = False,
        role_name_extractor: RoleNameExtractor | None = None,
        rerank_score_threshold: float | None = None,
        conflict_strategy: ConflictStrategy | None = None,
        conflict_positive_ratio: float = 0.2,
        conflict_negative_ratio: float = 0.2,
        conflict_scope_ratio: float = 0.2,
        near_config: NearConfig | None = None,
        inject_add_timestamp: bool = False,
        mempursuit_config: MemPursuitConfig | None = None,
        rank_fusion_config: RankFusionConfig | None = None,
        sira_config: SiraConfig | None = None,
        sira_expander=None,
        long_question_config: LongQuestionConfig | None = None,
        long_question_extractor=None,
        rerank_small2big_enabled: bool = False,
        mempursuit_small2big_enabled: bool = False,
    ) -> None:
        self.mempursuit_config = mempursuit_config if mempursuit_config is not None else MemPursuitConfig()
        self.rank_fusion_config = rank_fusion_config or RankFusionConfig()
        self.sira_config = sira_config or SiraConfig()
        self.sira_expander = sira_expander
        self.long_question_config = long_question_config or LongQuestionConfig()
        self.long_question_extractor = long_question_extractor
        if self.long_question_config.enabled and long_question_extractor is None:
            raise ValueError("long_question_enabled requires a configured extractor")
        for key, value in (("rerank_small2big_enabled", rerank_small2big_enabled),
                           ("mempursuit_small2big_enabled", mempursuit_small2big_enabled)):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            setattr(self, key, value)
        if self.sira_config.enabled and self.sira_expander is None:
            raise ValueError("sira_enabled requires a configured expander")
        self.near_config = near_config or NearConfig()
        self._near_cache = NearCache(self.near_config)
        validate_ratios(conflict_positive_ratio, conflict_negative_ratio, conflict_scope_ratio)
        self.conflict_strategy = conflict_strategy
        self.conflict_positive_ratio = conflict_positive_ratio
        self.conflict_negative_ratio = conflict_negative_ratio
        self.conflict_scope_ratio = conflict_scope_ratio
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        embedding_identity = str(
            getattr(
                embedder,
                "identity",
                getattr(embedder, "model_name", f"{embedder.__class__.__module__}.{embedder.__class__.__qualname__}"),
            )
        )
        self.embedding_identity = (
            embedding_identity
            if context_messages == 0
            else f"{embedding_identity}|context=v1:{context_messages}"
        )
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        if retrieval_mode not in {"dense", "hybrid", "enhanced"}:
            raise ValueError("retrieval_mode must be 'dense', 'hybrid', or 'enhanced'")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if lexical_weight < 0:
            raise ValueError("lexical_weight must be non-negative")
        if context_messages < 0:
            raise ValueError("context_messages must be non-negative")
        if not 0.0 <= option_view_weight <= 1.0:
            raise ValueError("option_view_weight must be between 0 and 1")
        if rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        if rerank_score_threshold is not None and (
            isinstance(rerank_score_threshold, bool)
            or not isinstance(rerank_score_threshold, (int, float))
            or not math.isfinite(rerank_score_threshold)
        ):
            raise ValueError("rerank_score_threshold must be a finite number or None")
        if result_window < 0:
            raise ValueError("result_window must be non-negative")
        if result_window_seed_k <= 0:
            raise ValueError("result_window_seed_k must be positive")
        if pack_span <= 0:
            raise ValueError("pack_span must be positive")
        if not isinstance(inject_add_timestamp, bool):
            raise ValueError("inject_add_timestamp must be a boolean")
        self.retrieval_mode = retrieval_mode
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.context_messages = context_messages
        self.option_view_weight = option_view_weight
        self.reranker = reranker
        self.reranker_enabled = reranker is not None
        self.rerank_candidates = rerank_candidates
        self.rerank_score_threshold = rerank_score_threshold
        self.inject_add_timestamp = inject_add_timestamp
        self.temporal_enrichment = temporal_enrichment
        self.resolve_relative_time_on_add = resolve_relative_time_on_add
        self.role_name_extractor = role_name_extractor
        self.result_window = result_window
        self.result_window_seed_k = result_window_seed_k
        self.pack_span = pack_span
        self._embedding_lock = threading.RLock()
        self._mutation_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 60000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_timestamp INTEGER,
                    created_at TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES add_requests(request_id)
                );

                CREATE INDEX IF NOT EXISTS memories_user_id_idx
                ON memories(user_id);

                CREATE INDEX IF NOT EXISTS memories_user_role_idx
                ON memories(user_id, role);

                CREATE TABLE IF NOT EXISTS user_role_names (
                    user_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    assistant_name TEXT,
                    conflicted INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS role_name_extraction_state (
                    user_id TEXT PRIMARY KEY,
                    consecutive_no_new_names INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS role_name_conflicts (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    previous_names TEXT NOT NULL,
                    incoming_names TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES add_requests(request_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                """
            )
            # Serialize the empty-store check and identity update with other writes.
            connection.execute("BEGIN IMMEDIATE")
            existing_identity = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'embedding_identity'"
            ).fetchone()
            if existing_identity is None:
                connection.execute(
                    "INSERT INTO store_metadata(key, value) VALUES ('embedding_identity', ?)",
                    (self.embedding_identity,),
                )
            elif existing_identity[0] != self.embedding_identity:
                has_data = connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM memories) OR EXISTS(SELECT 1 FROM add_requests)"
                ).fetchone()[0]
                if has_data:
                    raise ValueError(
                        "Configured embedding model does not match the model used by this memory database"
                    )
                connection.execute(
                    "UPDATE store_metadata SET value = ? WHERE key = 'embedding_identity'",
                    (self.embedding_identity,),
                )
            connection.commit()

    @staticmethod
    def _payload_hash(user_id: str, session_id: str, messages: list[MemoryMessage]) -> str:
        value = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [message.__dict__ for message in messages],
        }
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp(value: int | None) -> str:
        if value is None:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    def _embedding_text(
        self,
        messages: list[MemoryMessage],
        message_index: int,
        content: str,
        role_names: dict[str, str] | None = None,
    ) -> str:
        names = role_names or {}
        current = messages[message_index]
        if self.context_messages == 0:
            name = names.get(current.role)
            return f"[{name}] {content}" if name else content
        lines = [f"{names.get(current.role, current.role)}: {content}"]
        excerpt_tokens = max(16, min(96, self.chunk_tokens // 3))
        for distance in range(1, self.context_messages + 1):
            previous_index = message_index - distance
            if previous_index >= 0:
                previous = messages[previous_index]
                body = self._context_excerpt(previous.content, excerpt_tokens, from_end=True)
                if body:
                    lines.append(f"Previous {names.get(previous.role, previous.role)}: {body}")
            next_index = message_index + distance
            if next_index < len(messages):
                following = messages[next_index]
                body = self._context_excerpt(following.content, excerpt_tokens, from_end=False)
                if body:
                    lines.append(f"Next {names.get(following.role, following.role)}: {body}")
        return "\n".join(lines)

    @staticmethod
    def _context_excerpt(text: str, token_limit: int, from_end: bool) -> str:
        matches = list(TOKEN_PATTERN.finditer(text))
        if not matches:
            return ""
        if len(matches) <= token_limit:
            return text.strip()
        if from_end:
            return text[matches[-token_limit].start() : matches[-1].end()].strip()
        return text[matches[0].start() : matches[token_limit - 1].end()].strip()

    def add(self, request_id: str, user_id: str, session_id: str, messages: list[MemoryMessage]) -> int:
        # Serialize memory and role-state updates with clear.
        with self._mutation_lock:
            return self._add_unlocked(request_id, user_id, session_id, messages)

    @staticmethod
    def _read_role_names(connection: sqlite3.Connection, user_id: str) -> RoleNames:
        row = connection.execute(
            "SELECT user_name, assistant_name, conflicted FROM user_role_names WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return RoleNames()
        if row["conflicted"]:
            return RoleNames(conflicted=True)
        return RoleNames({role: row[f"{role}_name"] for role in ("user", "assistant")
                          if row[f"{role}_name"] is not None})

    @staticmethod
    def _read_role_name_misses(connection: sqlite3.Connection, user_id: str) -> int:
        row = connection.execute(
            "SELECT consecutive_no_new_names FROM role_name_extraction_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def _save_role_names(self, connection: sqlite3.Connection, user_id: str,
                         incoming: RoleNames, request_id: str) -> None:
        # Read again inside the write transaction to merge concurrent processes safely.
        previous = self._read_role_names(connection, user_id)
        if role_names_conflict(previous, incoming):
            connection.execute(
                """INSERT INTO role_name_conflicts
                   (request_id, user_id, previous_names, incoming_names, evidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (request_id, user_id, json.dumps(previous.names, ensure_ascii=False),
                 json.dumps(incoming.names, ensure_ascii=False),
                 json.dumps(incoming.evidence, ensure_ascii=False), self._timestamp(None)),
            )
            logger.warning("Rejected conflicting role-name batch; evidence saved to role_name_conflicts")
        state = merge_role_names(previous, incoming)
        # Only accepted names for previously unnamed roles reset the streak.
        # Persist even empty results, atomically with the successful /add.
        has_new_names = bool(state.names.keys() - previous.names.keys())
        misses = 0 if has_new_names else min(2, self._read_role_name_misses(connection, user_id) + 1)
        connection.execute(
            """INSERT INTO role_name_extraction_state(user_id, consecutive_no_new_names)
               VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET
               consecutive_no_new_names=excluded.consecutive_no_new_names""",
            (user_id, misses),
        )
        if not state.names and not state.conflicted:
            return
        connection.execute(
            """INSERT INTO user_role_names(user_id, user_name, assistant_name, conflicted)
               VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET
               user_name=excluded.user_name, assistant_name=excluded.assistant_name,
               conflicted=excluded.conflicted""",
            (user_id, state.names.get("user"), state.names.get("assistant"), int(state.conflicted)),
        )

    def _add_unlocked(self, request_id: str, user_id: str, session_id: str, messages: list[MemoryMessage]) -> int:
        # 幂等命中直接返回，不重复保存记忆或运行角色抽取。
        payload_hash = self._payload_hash(user_id, session_id, messages)
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT user_id, session_id, payload_hash FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != user_id
                    or existing["session_id"] != session_id
                    or existing["payload_hash"] != payload_hash
                ):
                    raise RequestConflictError("request_id was already used with a different payload")
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM memories WHERE request_id = ?", (request_id,)
                    ).fetchone()[0]
                )
            previous_names = (self._read_role_names(connection, user_id)
                              if self.role_name_extractor is not None else RoleNames())
            name_misses = (self._read_role_name_misses(connection, user_id)
                           if self.role_name_extractor is not None else 0)

        # 沿用现有角色抽取规则，合并姓名供记忆 embedding 和后续检索使用。
        extracted_names = RoleNames()
        names_complete = not previous_names.conflicted and len(previous_names.names) == 2
        attempt_names = self.role_name_extractor is not None and not names_complete and name_misses < 2
        if attempt_names:
            try:
                # Use the original transcript, before chunking/date transformations.
                extracted_names = self.role_name_extractor.extract(messages)
            except Exception as error:
                # Provider exception text may include credentials or transcript content.
                logger.warning("Role-name extraction failed (%s); keeping existing mapping",
                               type(error).__name__)
        role_names = merge_role_names(previous_names, extracted_names).names

        pending, vectors = self._prepare_memories(request_id, messages, role_names)

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT user_id, session_id, payload_hash FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != user_id
                    or existing["session_id"] != session_id
                    or existing["payload_hash"] != payload_hash
                ):
                    raise RequestConflictError("request_id was already used with a different payload")
                count = connection.execute(
                    "SELECT COUNT(*) FROM memories WHERE request_id = ?", (request_id,)
                ).fetchone()[0]
                connection.commit()
                return int(count)

            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            connection.execute(
                "INSERT INTO add_requests(request_id, user_id, session_id, payload_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (request_id, user_id, session_id, payload_hash, now),
            )
            for item, vector in zip(pending, vectors):
                memory_id, role, content, source_timestamp, created_at, _ = item
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, request_id, user_id, session_id, role, content,
                        source_timestamp, created_at, embedding, dimension
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        request_id,
                        user_id,
                        session_id,
                        role,
                        content,
                        source_timestamp,
                        created_at,
                        np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
                        int(vector.shape[0]),
                    ),
                )
            # Commit memories and role state together.
            if attempt_names:
                self._save_role_names(connection, user_id, extracted_names, request_id)
            connection.commit()
            self._near_cache.invalidate(user_id)
            return len(pending)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _prepare_memories(self, request_id, messages, role_names):
        # Keep the request hash based on the original payload, including on retries.
        if self.resolve_relative_time_on_add:
            messages = [
                MemoryMessage(
                    message.role,
                    _resolve_relative_time(
                        message.content,
                        datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc),
                    ) if message.timestamp is not None else message.content,
                    message.timestamp,
                )
                for message in messages
            ]

        # 以请求、消息和分块序号生成稳定 mem_id。
        # pending 保存 ID、消息角色、存储正文、来源时间、创建时间和 embedding 输入。
        pending: list[tuple[str, str, str, int | None, str, str]] = []
        for message_index, message in enumerate(messages):
            chunks = split_memory_text(message.content, self.chunk_tokens, self.overlap_tokens)
            created_at = self._timestamp(message.timestamp)
            # Only source dates belong in stored/embedded text. Add time is metadata;
            # inject_add_timestamp controls its display when formatting search results.
            date_prefix = ""
            if message.timestamp is not None:
                message_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                month_names = (
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December",
                )
                date_prefix = (
                    f"[{month_names[message_date.month - 1]} {message_date.day}, {message_date.year}] "
                )
            for chunk_index, content in enumerate(chunks):
                content = date_prefix + content
                identity = f"{request_id}:{message_index}:{chunk_index}"
                memory_id = "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                pending.append(
                    (
                        memory_id,
                        message.role,
                        content,
                        message.timestamp,
                        created_at,
                        self._embedding_text(messages, message_index, content, role_names),
                    )
                )
        if not pending:
            raise ValueError("messages produced no searchable content")

        with self._embedding_lock:
            vectors = self.embedder.encode_documents([item[5] for item in pending])
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(pending):
            raise ValueError("Embedder returned an invalid document vector matrix")

        return pending, vectors

    def search(self, user_id: str, query: str, top_k: int, options: list[str] | None = None,
               diagnostics: bool = False):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        config = self.long_question_config
        original = query
        extra_rows = []
        detail = {**config.settings(), "original_chars": len(query), "candidate_count": 0,
                  "status": "disabled" if not config.enabled else "below_threshold"}
        if config.enabled and len(query) > config.threshold_chars:
            try:
                extracted = self.long_question_extractor.extract(query[:config.head_chars], query[-config.tail_chars:])
                if not isinstance(extracted, str) or not extracted.strip() or len(extracted) > config.head_chars + config.tail_chars:
                    raise ValueError("Invalid extracted query")
                with closing(self._connect()) as connection:
                    state = self._read_role_names(connection, user_id) if self.role_name_extractor else RoleNames()
                identity = "question_" + hashlib.sha256((user_id + "\0" + original).encode()).hexdigest()
                # The entire question, including head and tail, uses the add preparation path.
                pending, vectors = self._prepare_memories(identity, [MemoryMessage("user", original)], state.names)
                for item, vector in zip(pending, vectors):
                    memory_id, role, content, timestamp, created_at, _ = item
                    extra_rows.append(dict(id="question_" + memory_id, session_id=identity, role=role,
                                           content=content, source_timestamp=timestamp, created_at=created_at,
                                           embedding=vector.tobytes(), dimension=len(vector)))
                query = extracted.strip()
                detail.update(status="applied", query=query, candidate_count=len(extra_rows), evidence_chars=len(original))
            except Exception as error:
                extra_rows = []
                query = original
                detail.update(status="fallback", error_type=type(error).__name__)
                logger.warning("Long question preparation failed (%s); using original query", type(error).__name__)
        result = self._search_prepared(user_id, query, top_k, options, diagnostics, extra_rows=extra_rows)
        if diagnostics:
            hits, trace = result
            trace["long_question"] = detail
            trace["persistent_count"] = trace.get("stored_count", 0) - len(extra_rows)
            return hits, trace
        return result

    def _search_prepared(self, user_id: str, query: str, top_k: int, options: list[str] | None = None,
               diagnostics: bool = False, *, extra_rows=()) -> list[MemoryHit] | tuple[list[MemoryHit], dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        # Keep every route on the same immutable config if /config changes mid-search.
        mempursuit_config = self.mempursuit_config
        rank_fusion_config = self.rank_fusion_config
        sira_context = SiraRequest(query, self.sira_config, self.sira_expander)
        with closing(self._connect()) as connection:
            state = (self._read_role_names(connection, user_id)
                     if self.role_name_extractor is not None else RoleNames())
            selected_role = query_role(query, state)
            rows = connection.execute(
                """
                SELECT id, session_id, role, content, source_timestamp, created_at, embedding, dimension
                FROM memories WHERE user_id = ?
                """ + (" AND role = ?" if selected_role else "") + " ORDER BY rowid",
                (user_id, selected_role) if selected_role else (user_id,),
            ).fetchall()
        rows = list(rows) + list(extra_rows)
        role_detail = {"role_name_enabled": self.role_name_extractor is not None,
                       "role_names": state.names, "role_names_conflicted": state.conflicted,
                       "selected_role": selected_role}
        near_context = self._near_context(user_id, rows, ephemeral=bool(extra_rows))
        conflict_detail = {"enabled": self.conflict_strategy is not None, "status": "bypassed",
                           "strategy": "three_queue_v1",
                           "is_judgment": None, "classification_status": "not_run",
                           "has_conflict": None, "assessment_status": "not_run",
                           "skip_reason": "empty_memories" if not rows else
                                          "disabled" if self.conflict_strategy is None else
                                          "options" if options else None}
        if not rows:
            empty = {"schema_version": 1, "stored_count": 0, "top_k": top_k, "candidates": [], "final_results": [],
                     "conflict_strategy": conflict_detail,
                     "rank_fusion": {**rank_fusion_config.settings(), "active": False, "reason": "empty_memories"},
                     "sira": {**sira_context.detail, "status": "empty_memories"},
                     "mempursuit": {"enabled": mempursuit_config.enabled, "active": False,
                                    "bypass_reason": "empty_memories", "reason": "empty_memories",
                                    **mempursuit_config.settings()}, **role_detail}
            return ([], empty) if diagnostics else []

        if self.conflict_strategy is not None:
            queries, conflict_detail = self._classify_conflict_query(query, conflict_detail)
            if queries is not None:
                return self._search_conflict(
                    user_id, rows, state, role_detail, query, top_k, queries, conflict_detail,
                    options=options, diagnostics=diagnostics, near_context=near_context,
                    mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config,
                )
        return self._search_normal(
            rows, state, role_detail, query, top_k, conflict_detail,
            options=options, diagnostics=diagnostics, near_context=near_context,
            mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config,
        )

    def _classify_conflict_query(self, query: str, conflict_detail: dict) -> tuple[JudgmentQueries | None, dict]:
        """Classify for an enabled strategy, preserving skips and failure fallback."""
        if conflict_detail["skip_reason"] is not None:
            return None, conflict_detail
        queries = None
        try:
            queries = self.conflict_strategy.classify(query)
            conflict_detail["status"] = "judgment" if queries is not None else "not_judgment"
            conflict_detail.update(is_judgment=queries is not None, classification_status="completed")
        except Exception:
            logger.warning("Conflict question classification failed; using ordinary search")
            conflict_detail.update(status="classification_failed", classification_status="failed")
        return queries, conflict_detail

    def _search_normal(self, rows, state, role_detail, query: str, top_k: int, conflict_detail: dict, *,
                       options=None, diagnostics=False, near_context=None, mempursuit_config=None, sira_context=None, rank_fusion_config=None):
        """Return the ordinary recall queue with its search diagnostics."""
        ordinary = self._search_rows(rows, state, role_detail, query, top_k, options, diagnostics, near_context,
                                     mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config)
        if diagnostics:
            trace = ordinary[1]
            trace["conflict_strategy"] = conflict_detail
            trace["semantic_results"] = [{"id": hit.id, "rank": rank}
                                         for rank, hit in enumerate(ordinary[0], 1)]
            for candidate in trace["candidates"]:
                candidate["semantic_returned_rank"] = candidate["returned_rank"]
        return ordinary

    def _search_conflict(self, user_id: str, rows, state, role_detail, query: str, top_k: int,
                         queries: JudgmentQueries, conflict_detail: dict, *, options=None,
                         diagnostics=False, near_context=None, mempursuit_config=None, sira_context=None, rank_fusion_config=None):
        """Return the three evidence queues and optional conflict explanation."""
        # Only offline diagnostics run the original question for rank comparison.
        # Its hits never fill or replace any of the three evidence queues.
        if diagnostics:
            _, detail = self._search_normal(
                rows, state, role_detail, query, top_k, conflict_detail,
                options=options, diagnostics=True, near_context=near_context,
                mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config,
            )

        selected = {}
        queue_details = {}
        errors = {}
        routes = (("positive", queries.positive, self.conflict_positive_ratio, "[Positive-related evidence] "),
                  ("negative", queries.negative, self.conflict_negative_ratio, "[Negative-related evidence] "),
                  ("scope", queries.scope, self.conflict_scope_ratio, "[Scope/correction evidence] "))
        for name, rewritten, ratio, prefix in routes:
            quota = math.floor(top_k * ratio)
            conflict_detail[f"{name}_query"] = rewritten
            conflict_detail[f"{name}_quota"] = quota
            hits, trace = [], None
            try:
                if quota:
                    # Retain the same reranker candidate budget and packing rules.
                    result = self._search_rows(rows, state, role_detail, rewritten, top_k, None, diagnostics, near_context,
                                               mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config)
                    hits, trace = result if diagnostics else (result, None)
            except Exception as error:
                logger.warning("Conflict %s retrieval failed; leaving this queue empty (%s)",
                               name, type(error).__name__)
                errors[name] = type(error).__name__
            selected[name] = hits[:quota]
            conflict_detail[f"{name}_count"] = len(selected[name])
            if diagnostics:
                queue_details[name] = {"pipeline": trace,
                                       "selected_ids": [hit.id for hit in selected[name]]}

        note = None
        if errors:
            conflict_detail.update(status="retrieval_failed", retrieval_errors=errors)
        elif selected["positive"] and selected["negative"]:
            try:
                note = self.conflict_strategy.assess(query, selected["positive"], selected["negative"], selected["scope"])
                conflict_detail.update(status="conflict" if note else "no_conflict",
                                       has_conflict=note is not None, assessment_status="completed")
            except Exception as error:
                logger.warning("Conflict evidence assessment failed; returning evidence without a note (%s)",
                               type(error).__name__)
                conflict_detail.update(status="assessment_failed", assessment_status="failed",
                                       assessment_error_type=type(error).__name__)
                # Local validation errors contain no API credentials or raw response.
                if isinstance(error, ValueError):
                    conflict_detail["assessment_error"] = str(error)
        else:
            conflict_detail["status"] = "insufficient_evidence"
        # A generated explanation leads the results without consuming any of
        # the three evidence queues' budgets. It is never stored as a memory.
        conflict_detail["explanation"] = note
        conflict_detail["explanation_placement"] = "first_result"
        results = []
        if note:
            identity = json.dumps([user_id, query, note], ensure_ascii=False)
            results.append(MemoryHit(
                id="conflict:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                content=note, score=0.0, created_at="", recall_queue="conflict_summary",
            ))
        evidence_offset = len(results)
        results.extend(replace(hit, content=prefix + hit.content, recall_queue=name)
                       for name, _, _, prefix in routes for hit in selected[name])
        if not diagnostics:
            return results
        conflict_detail["queues"] = queue_details
        detail["returned_count"] = len(results)
        detail["final_results"] = [{**hit.__dict__, "rank": rank}
                                   for rank, hit in enumerate(results, 1)]
        # Map each route's packed members to their final occurrences while
        # retaining the original question's ranks as a separate diagnostic.
        member_ranks: dict[str, set[int]] = {}
        offset = evidence_offset
        for name, _, _, _ in routes:
            trace = queue_details[name]["pipeline"]
            hits = selected[name]
            if trace is not None:
                for candidate in trace["candidates"]:
                    local_rank = candidate["returned_rank"]
                    if local_rank is not None and local_rank <= len(hits):
                        member_ranks.setdefault(candidate["id"], set()).add(offset + local_rank)
            offset += len(hits)
        for candidate in detail["candidates"]:
            ranks = sorted(member_ranks.get(candidate["id"], ()))
            candidate["returned_rank"] = ranks[0] if ranks else None
            candidate["returned_ranks"] = ranks
        return results, detail

    def _near_context(self, user_id, rows, ephemeral=False):
        if not rows or not self.near_config.enabled or self.near_config.weight <= 0:
            return None
        dimension = int(rows[0]["dimension"])
        matrix = np.vstack([np.frombuffer(row["embedding"], dtype=np.float32, count=dimension)
                            for row in rows])
        cache = NearCache(self.near_config) if ephemeral else self._near_cache
        return cache.get(user_id, [row["id"] for row in rows], matrix)

    def _retrieve_chunks(self, rows, state, query, top_k, *, options=None,
                         diagnostics=False, near_context=None, mempursuit_config=None, sira_context=None, rank_fusion_config=None):
        if rank_fusion_config is None:
            rank_fusion_config = self.rank_fusion_config
        if mempursuit_config is None:
            mempursuit_config = self.mempursuit_config
        if sira_context is None:
            sira_context = SiraRequest(query, self.sira_config, self.sira_expander)
        ranking = retrieve_chunks(
            rows, state.names, query, top_k, embedder=self.embedder,
            embedding_lock=self._embedding_lock, render_content=self._result_content,
            retrieval_mode=self.retrieval_mode, lexical_weight=self.lexical_weight,
            rrf_k=self.rrf_k, option_view_weight=self.option_view_weight,
            reranker=self.reranker if self.reranker_enabled else None,
            rerank_candidates=self.rerank_candidates,
            rerank_score_threshold=self.rerank_score_threshold,
            rerank_small2big_enabled=self.rerank_small2big_enabled,
            mempursuit_small2big_enabled=self.mempursuit_small2big_enabled,
            options=options, diagnostics=diagnostics, near_context=near_context,
            mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config,
        )
        if diagnostics:
            ranking.diagnostics["near"].update(self.near_config.settings())
            pursuit_detail = ranking.diagnostics.setdefault("mempursuit", {"active": False})
            pursuit_detail.update(mempursuit_config.settings())
            pursuit_detail["enabled"] = mempursuit_config.enabled
        return ranking

    def _search_rows(self, rows, state, role_detail, query: str, top_k: int,
                     options: list[str] | None = None, diagnostics: bool = False, near_context=None, *,
                     mempursuit_config=None, sira_context=None, rank_fusion_config=None):
        ranking = self._retrieve_chunks(rows, state, query, top_k, options=options,
                                        diagnostics=diagnostics, near_context=near_context,
                                        mempursuit_config=mempursuit_config, sira_context=sira_context, rank_fusion_config=rank_fusion_config)
        positions, result_scores = ranking.positions, ranking.scores

        limit = min(top_k, len(rows))
        if self.pack_span > 1:
            clusters = self._pack_result_positions(positions, rows, limit)
        else:
            if self.result_window:
                positions = self._expand_result_positions(positions, rows, limit)
            else:
                positions = positions[:limit]
            clusters = [[int(position)] for position in positions]
        results: list[MemoryHit] = []
        for cluster in clusters:
            seed_row = rows[cluster[0]]
            timestamp = (
                self._timestamp(seed_row["source_timestamp"])
                if seed_row["source_timestamp"] is not None
                else seed_row["created_at"]
            )
            score = float(result_scores[cluster[0]])
            if not math.isfinite(score):
                score = 0.0
            contents: list[str] = []
            # Keep cluster[0] as the hit identity, but render in conversation order.
            for position in sorted(cluster):
                row = rows[position]
                contents.append(self._result_content(row, state.names))
            results.append(
                MemoryHit(
                    id=str(seed_row["id"]),
                    content="\n".join(contents),
                    score=score,
                    created_at=str(timestamp),
                )
            )
        if not diagnostics:
            return results

        detail = ranking.diagnostics
        returned = {position: rank for rank, cluster in enumerate(clusters, 1) for position in cluster}
        row_positions = {str(row["id"]): i for i, row in enumerate(rows)}
        for candidate in detail["candidates"]:
            candidate["returned_rank"] = returned.get(row_positions[candidate["id"]])
        detail.update(returned_count=len(results),
                      final_results=[{**hit.__dict__, "rank": rank}
                                     for rank, hit in enumerate(results, 1)], **role_detail)
        return results, detail

    def _pack_result_positions(
        self,
        positions: np.ndarray,
        rows: list[sqlite3.Row],
        items: int,
    ) -> list[list[int]]:
        """Pack the same bidirectional window used by result expansion.

        ``pack_span > 1`` enables packing; ``result_window`` controls its radius.
        Only the original first ``result_window_seed_k`` candidates expand.
        Covered rows are skipped and remaining candidates backfill the item budget.
        Each cluster keeps its seed first for result identity; rendering sorts rows.
        """
        clusters: list[list[int]] = []
        covered: set[int] = set()
        for rank, raw_position in enumerate(positions):
            if len(clusters) >= items:
                break
            position = int(raw_position)
            if position in covered:
                continue
            cluster = [position]
            covered.add(position)
            if rank < self.result_window_seed_k:
                for neighbor in self._result_neighbors(position, rows):
                    if neighbor not in covered:
                        cluster.append(neighbor)
                        covered.add(neighbor)
            clusters.append(cluster)
        return clusters

    def get_config(self) -> dict[str, object]:
        """Current runtime-tunable retrieval settings."""
        return {
            "retrieval_mode": self.retrieval_mode,
            "rrf_k": self.rrf_k,
            "lexical_weight": self.lexical_weight,
            "result_window": self.result_window,
            "result_window_seed_k": self.result_window_seed_k,
            "pack_span": self.pack_span,
            "rerank_candidates": self.rerank_candidates,
            "rerank_score_threshold": self.rerank_score_threshold,
            "reranker_loaded": self.reranker is not None,
            "reranker_enabled": self.reranker_enabled,
            "rerank_small2big_enabled": self.rerank_small2big_enabled,
            "mempursuit_small2big_enabled": self.mempursuit_small2big_enabled,
            "temporal_enrichment": self.temporal_enrichment,
            "inject_add_timestamp": self.inject_add_timestamp,
            "role_name_enabled": self.role_name_extractor is not None,
            "conflict_enabled": self.conflict_strategy is not None,
            "conflict_positive_ratio": self.conflict_positive_ratio,
            "conflict_negative_ratio": self.conflict_negative_ratio,
            "conflict_scope_ratio": self.conflict_scope_ratio,
            **self.near_config.settings(),
            **self.mempursuit_config.settings(),
            **self.rank_fusion_config.settings(),
            **self.sira_config.settings(),
            **self.long_question_config.settings(),
        }

    def update_config(self, **kwargs: object) -> None:
        """Runtime-tunable retrieval settings; write-path/model parameters excluded.

        Validates every supplied value before assigning any, so a rejected patch
        leaves the store unchanged.
        """
        allowed = {
            "retrieval_mode",
            "rrf_k",
            "lexical_weight",
            "result_window",
            "result_window_seed_k",
            "pack_span",
            "rerank_candidates",
            "reranker_enabled",
            "temporal_enrichment",
            "inject_add_timestamp",
            "mempursuit_enabled", "sira_enabled", "long_question_enabled",
            "rerank_small2big_enabled", "mempursuit_small2big_enabled",
        }
        allowed.update(self.rank_fusion_config.settings())
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(f"Unknown or non-runtime settings: {', '.join(unknown)}")

        retrieval_mode = kwargs.get("retrieval_mode", self.retrieval_mode)
        if retrieval_mode not in {"dense", "hybrid", "enhanced"}:
            raise ValueError("retrieval_mode must be 'dense', 'hybrid', or 'enhanced'")
        rrf_k = kwargs.get("rrf_k", self.rrf_k)
        if not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        lexical_weight = kwargs.get("lexical_weight", self.lexical_weight)
        if not isinstance(lexical_weight, (int, float)) or lexical_weight < 0:
            raise ValueError("lexical_weight must be non-negative")
        result_window = kwargs.get("result_window", self.result_window)
        if not isinstance(result_window, int) or result_window < 0:
            raise ValueError("result_window must be non-negative")
        result_window_seed_k = kwargs.get("result_window_seed_k", self.result_window_seed_k)
        if not isinstance(result_window_seed_k, int) or result_window_seed_k <= 0:
            raise ValueError("result_window_seed_k must be positive")
        pack_span = kwargs.get("pack_span", self.pack_span)
        if not isinstance(pack_span, int) or pack_span <= 0:
            raise ValueError("pack_span must be positive")
        rerank_candidates = kwargs.get("rerank_candidates", self.rerank_candidates)
        if not isinstance(rerank_candidates, int) or rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        reranker_enabled = kwargs.get("reranker_enabled", self.reranker_enabled)
        if not isinstance(reranker_enabled, bool):
            raise ValueError("reranker_enabled must be a boolean")
        if reranker_enabled and self.reranker is None:
            raise ValueError("reranker_enabled requires a reranker model loaded at startup")
        inject_add_timestamp = kwargs.get("inject_add_timestamp", self.inject_add_timestamp)
        temporal_enrichment = kwargs.get("temporal_enrichment", self.temporal_enrichment)
        if not isinstance(temporal_enrichment, bool):
            raise ValueError("temporal_enrichment must be a boolean")

        if not isinstance(inject_add_timestamp, bool):
            raise ValueError("inject_add_timestamp must be a boolean")
        mempursuit_enabled = kwargs.get("mempursuit_enabled", self.mempursuit_config.enabled)
        if not isinstance(mempursuit_enabled, bool):
            raise ValueError("mempursuit_enabled must be a boolean")
        mempursuit_config = replace(self.mempursuit_config, enabled=mempursuit_enabled)
        sira_enabled = kwargs.get("sira_enabled", self.sira_config.enabled)
        sira_config = replace(self.sira_config, enabled=sira_enabled)
        if sira_enabled and self.sira_expander is None:
            raise ValueError("sira_enabled requires an expander configured at startup")
        long_question_config = replace(self.long_question_config, enabled=kwargs.get("long_question_enabled", self.long_question_config.enabled))
        if long_question_config.enabled and self.long_question_extractor is None:
            raise ValueError("long_question_enabled requires an extractor configured at startup")
        stage_settings = {}
        for key in ("rerank_small2big_enabled", "mempursuit_small2big_enabled"):
            value = kwargs.get(key, getattr(self, key))
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            stage_settings[key] = value
        rank_fusion_config = replace(self.rank_fusion_config, **{
            key.removeprefix("rank_fusion_"): value for key, value in kwargs.items()
            if key.startswith("rank_fusion_")})
        self.long_question_config = long_question_config
        self.rank_fusion_config = rank_fusion_config
        self.sira_config = sira_config
        self.retrieval_mode = retrieval_mode
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.result_window = result_window
        self.result_window_seed_k = result_window_seed_k
        self.pack_span = pack_span
        self.rerank_candidates = rerank_candidates
        self.reranker_enabled = reranker_enabled
        self.inject_add_timestamp = inject_add_timestamp
        self.temporal_enrichment = temporal_enrichment
        self.mempursuit_config = mempursuit_config
        for key, value in stage_settings.items():
            setattr(self, key, value)

    def _result_neighbors(self, position: int, rows: list[sqlite3.Row]):
        """Yield same-session neighbors in the shared result-window range."""
        for distance in range(1, self.result_window + 1):
            for neighbor in (position - distance, position + distance):
                if (
                    0 <= neighbor < len(rows)
                    and rows[neighbor]["session_id"] == rows[position]["session_id"]
                ):
                    yield neighbor

    def _expand_result_positions(
        self,
        positions: np.ndarray,
        rows: list[sqlite3.Row],
        limit: int,
    ) -> np.ndarray:
        expanded: list[int] = []
        seen: set[int] = set()

        def add(position: int) -> None:
            if 0 <= position < len(rows) and position not in seen and len(expanded) < limit:
                seen.add(position)
                expanded.append(position)

        seed_count = min(self.result_window_seed_k, len(positions))
        for raw_position in positions[:seed_count]:
            position = int(raw_position)
            add(position)
            for neighbor in self._result_neighbors(position, rows):
                add(neighbor)

        for raw_position in positions:
            add(int(raw_position))
        return np.asarray(expanded, dtype=np.int64)

    def _result_content(self, row: sqlite3.Row, role_names: dict[str, str] | None = None) -> str:
        speaker = (role_names or {}).get(row["role"], row["role"])
        if row["source_timestamp"] is None:
            if self.inject_add_timestamp:
                return f"[{speaker} | {row['created_at']}] {row['content']}"
            return f"[{speaker}] {row['content']}"
        if self.temporal_enrichment:
            message_time = datetime.fromtimestamp(
                row["source_timestamp"] / 1000,
                tz=timezone.utc,
            )
            rendered = (
                f"[{speaker} | Message date: {_format_date(message_time)} "
                f"at {message_time:%H:%M} UTC] {row['content']}"
            )
            annotations = _relative_time_annotations(str(row["content"]), message_time)
            if annotations:
                rendered += f" [Resolved relative dates: {'; '.join(annotations)}]"
            return rendered
        timestamp = self._timestamp(row["source_timestamp"])
        return f"[{speaker} | {timestamp}] {row['content']}"

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            requests = int(connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0])
            memories = int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            users = int(connection.execute("SELECT COUNT(DISTINCT user_id) FROM memories").fetchone()[0])
        return {"requests": requests, "memories": memories, "users": users}

    def clear(self) -> dict[str, int]:
        """Clear stored memories and request records, retaining the database schema."""
        with self._mutation_lock:
            with closing(self._connect()) as connection:
                return reset_memory_database(connection, self.embedding_identity, self._near_cache.invalidate)
