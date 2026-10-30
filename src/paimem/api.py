from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import secrets
import stat
import time
from uuid import uuid4
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, model_validator

from .conflict_strategy import LLMConflictStrategy, validate_ratios
from .embeddings import build_embedder, embedding_options
from .memory_store import MemoryMessage, RecallQueue, RequestConflictError, SQLiteMemoryStore
from .sira import SiraConfig, LLMSiraExpander
from .long_question import LongQuestionConfig, LLMLongQuestionExtractor
from .rank_fusion import RankFusionConfig
from .mempursuit import MemPursuitConfig
from .near import NearConfig
from .reranking import build_reranker, reranker_options
from .role_names import EvidenceRoleNameExtractor


class MessagePayload(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: int | None = None


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[MessagePayload] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=100)
    debug_output_path: str | None = Field(default=None, min_length=1)
    include_recall_queue: bool = False


class SearchItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None
    recall_queue: RecallQueue | None = None


class SearchResponse(BaseModel):
    data: list[SearchItem]


def _diagnostics_root() -> Path:
    return Path(os.environ.get("MEMORY_DIAGNOSTICS_DIR", "artifacts/diagnostics")).resolve()


def _local_debug_path(request: Request, raw_path: str) -> Path:
    """Restrict authenticated local diagnostics to the configured output directory."""
    require_management_api_key(request.headers.get("authorization"), request.headers.get("x-api-key"))
    try:
        local = ipaddress.ip_address(request.client.host).is_loopback
    except (AttributeError, ValueError):
        local = False
    if not local:
        raise HTTPException(status_code=403, detail="Search diagnostics require a loopback client")
    path = Path(raw_path)
    if not path.is_absolute() or path.suffix != ".json" or ".." in path.parts:
        raise HTTPException(status_code=422, detail="debug_output_path must be an absolute .json path inside service.diagnostics_dir")
    # Resolve the operator-configured root, but never follow caller-supplied links.
    root = _diagnostics_root()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=422, detail="debug_output_path must be inside service.diagnostics_dir") from None
    with _debug_directory(path):
        pass
    return path


@contextmanager
def _debug_directory(path: Path):
    """Walk beneath a trusted root using directory FDs, rejecting symbolic links."""
    descriptors = []
    try:
        root = _diagnostics_root()
        relative = path.relative_to(root)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptors.append(os.open(root, flags))
        for part in relative.parts[:-1]:
            if part == "..":
                raise ValueError("Parent traversal")
            descriptors.append(os.open(part, flags, dir_fd=descriptors[-1]))
        try:
            metadata = os.stat(path.name, dir_fd=descriptors[-1], follow_symlinks=False)
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Non-regular destination")
        yield descriptors[-1]
    except (OSError, ValueError):
        raise HTTPException(status_code=422, detail="Diagnostics require an existing directory beneath service.diagnostics_dir; symbolic links and non-regular targets are not allowed") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _write_debug(path: Path, data: dict) -> None:
    # Atomic replacement supports retries; directory FDs prevent path redirection.
    with _debug_directory(path) as directory:
        temporary = ".paimem-" + secrets.token_hex(16) + ".tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    requests: int
    memories: int
    users: int


class ClearResponse(BaseModel):
    success: Literal[True] = True
    deleted_requests: int
    deleted_memories: int
    requests: int
    memories: int
    users: int


class ConfigPatch(BaseModel):
    retrieval_mode: Literal["dense", "hybrid", "enhanced"] | None = None
    rrf_k: int | None = Field(default=None, gt=0)
    lexical_weight: float | None = Field(default=None, ge=0)
    result_window: int | None = Field(default=None, ge=0)
    result_window_seed_k: int | None = Field(default=None, gt=0)
    pack_span: int | None = Field(default=None, gt=0)
    rerank_candidates: int | None = Field(default=None, gt=0)
    reranker_enabled: bool | None = None
    temporal_enrichment: bool | None = None
    inject_add_timestamp: bool | None = None
    rank_fusion_enabled: StrictBool | None = None
    rank_fusion_dense_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    rank_fusion_rerank_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    rank_fusion_bm25_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    rank_fusion_rrf_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    rank_fusion_mp_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)
    rank_fusion_mp_limit: int | None = Field(default=None, ge=-1, strict=True)
    rank_fusion_remaining_limit: int | None = Field(default=None, ge=-1, strict=True)
    mempursuit_enabled: StrictBool | None = None
    rerank_small2big_enabled: StrictBool | None = None
    mempursuit_small2big_enabled: StrictBool | None = None
    sira_enabled: StrictBool | None = None
    long_question_enabled: StrictBool | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_startup_mempursuit_settings(cls, values):
        if isinstance(values, dict):
            startup_only = sorted(({"mempursuit_candidates", "mempursuit_ridge"} | (set(SiraConfig().settings()) - {"sira_enabled"}) | (set(LongQuestionConfig().settings()) - {"long_question_enabled"})).intersection(values))
            if startup_only:
                raise ValueError(f"Startup-only settings: {', '.join(startup_only)}")
        return values


STORE_INIT_OVERRIDES: dict[str, object] = {}

_STORE_SETTING_KEYS = {
    "database",
    "embedding_model",
    "embedding_backend",
    "embedding_base_url",
    "embedding_api_key",
    "embedding_device",
    "embedding_query_prefix",
    "chunk_tokens",
    "chunk_overlap",
    "retrieval_mode",
    "rrf_k",
    "lexical_weight",
    "context_messages",
    "option_view_weight",
    "reranker_model",
    "reranker_backend",
    "reranker_base_url",
    "reranker_api_key",
    "reranker_timeout",
    "reranker_device",
    "reranker_batch_size",
    "reranker_max_length",
    "rerank_candidates",
    "rerank_small2big_enabled",
    "mempursuit_small2big_enabled",
    "rerank_score_threshold",
    "temporal_enrichment",
    "inject_add_timestamp",
    "resolve_relative_time_on_add",
    "result_window",
    "result_window_seed_k",
    "pack_span",
    "role_name_enabled",
    "conflict_enabled",
    "conflict_positive_ratio",
    "conflict_negative_ratio",
    "conflict_scope_ratio",
}
_STORE_SETTING_KEYS.update(NearConfig().settings())
_STORE_SETTING_KEYS.update(MemPursuitConfig().settings())
_STORE_SETTING_KEYS.update(RankFusionConfig().settings())
_STORE_SETTING_KEYS.update(SiraConfig().settings())
_STORE_SETTING_KEYS.update(LongQuestionConfig().settings())


def _stage_boolean(key: str) -> bool:
    raw = STORE_INIT_OVERRIDES.get(key, os.environ.get(f"MEMORY_{key.upper()}", "false"))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false", "1", "0"}:
        return raw.strip().lower() in {"true", "1"}
    raise ValueError(f"Invalid MEMORY_{key.upper()}; expected true/false or 1/0")


def _build_near_config() -> NearConfig:
    values = {}
    for name, default in NearConfig().__dict__.items():
        key = f"near_{name}"
        raw = STORE_INIT_OVERRIDES.get(key, os.environ.get(f"MEMORY_{key.upper()}", default))
        try:
            if isinstance(default, bool):
                if isinstance(raw, bool):
                    value = raw
                elif str(raw).strip().lower() in {"true", "false", "1", "0"}:
                    value = str(raw).strip().lower() in {"true", "1"}
                else:
                    raise ValueError()
            elif isinstance(default, int):
                if isinstance(raw, bool) or (not isinstance(raw, (str, int))):
                    raise ValueError()
                value = int(raw)
            else:
                if isinstance(raw, bool):
                    raise ValueError()
                value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid MEMORY_{key.upper()}") from exc
        values[name] = value
    return NearConfig(**values)


def _build_rank_fusion_config() -> RankFusionConfig:
    values = {}
    for name, default in RankFusionConfig().__dict__.items():
        key = f"rank_fusion_{name}"
        raw = STORE_INIT_OVERRIDES.get(key, os.environ.get(f"MEMORY_{key.upper()}", default))
        try:
            if isinstance(default, bool):
                if isinstance(raw, bool):
                    value = raw
                elif isinstance(raw, str) and raw.strip().lower() in {"true", "false", "1", "0"}:
                    value = raw.strip().lower() in {"true", "1"}
                else:
                    raise ValueError()
            elif isinstance(default, int):
                if isinstance(raw, bool) or not isinstance(raw, (str, int)):
                    raise ValueError()
                value = int(raw)
            else:
                if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                    raise ValueError()
                value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid MEMORY_{key.upper()}") from exc
        values[name] = value
    try:
        return RankFusionConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MEMORY_RANK_FUSION configuration: {exc}") from exc


def _build_mempursuit_config() -> MemPursuitConfig:
    values = {}
    for name, default in MemPursuitConfig().__dict__.items():
        key = f"mempursuit_{name}"
        raw = STORE_INIT_OVERRIDES.get(key, os.environ.get(f"MEMORY_{key.upper()}", default))
        try:
            if isinstance(default, bool):
                if isinstance(raw, bool):
                    value = raw
                elif isinstance(raw, str) and raw.strip().lower() in {"true", "false", "1", "0"}:
                    value = raw.strip().lower() in {"true", "1"}
                else:
                    raise ValueError()
            elif isinstance(default, int):
                if isinstance(raw, bool) or not isinstance(raw, (str, int)):
                    raise ValueError()
                value = int(raw)
            else:
                if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                    raise ValueError()
                value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid MEMORY_{key.upper()}") from exc
        values[name] = value
    try:
        return MemPursuitConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MEMORY_MEMPURSUIT configuration: {exc}") from exc


def _build_long_question_config() -> LongQuestionConfig:
    values = {}
    for name, default in LongQuestionConfig().__dict__.items():
        key = f"long_question_{name}"
        raw = STORE_INIT_OVERRIDES.get(key, os.environ.get(f"MEMORY_{key.upper()}", default))
        try:
            if name == "extra_body":
                value = raw  # LongQuestionConfig parses and validates the JSON object.
            elif isinstance(default, bool):
                if isinstance(raw, bool):
                    value = raw
                elif isinstance(raw, str) and raw.strip().lower() in {"true", "false", "1", "0"}:
                    value = raw.strip().lower() in {"true", "1"}
                else:
                    raise ValueError()
            elif isinstance(default, int):
                if isinstance(raw, bool) or not isinstance(raw, (str, int)):
                    raise ValueError()
                value = int(raw)
            else:
                if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                    raise ValueError()
                value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid MEMORY_{key.upper()}") from exc
        values[name] = value
    try:
        return LongQuestionConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MEMORY_LONG_QUESTION configuration: {exc}") from exc


def _build_long_question_extractor(config):
    model = (os.environ.get("MEMORY_LONG_QUESTION_MODEL") or os.environ.get("ANSWER_MODEL")
             or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL"))
    key = (os.environ.get("MEMORY_LONG_QUESTION_API_KEY") or os.environ.get("ANSWER_API_KEY")
           or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY"))
    base = (os.environ.get("MEMORY_LONG_QUESTION_API_BASE") or os.environ.get("ANSWER_API_BASE")
            or os.environ.get("API_BASE") or os.environ.get("OPENAI_BASE_URL"))
    if not all((model, key, base)):
        if config.enabled:
            raise ValueError("Long question requires an explicit model, API key and API base")
        return None
    from openai import OpenAI
    return LLMLongQuestionExtractor(OpenAI(api_key=key, base_url=base, timeout=config.timeout, max_retries=0), model, extra_body=config.extra_body)


def _build_sira_config() -> SiraConfig:
    values = {}
    for name, default in SiraConfig().__dict__.items():
        key = f"sira_{name}"
        raw = STORE_INIT_OVERRIDES.get(key, os.environ.get(f"MEMORY_{key.upper()}", default))
        try:
            if isinstance(default, bool):
                if isinstance(raw, bool):
                    value = raw
                elif isinstance(raw, str) and raw.strip().lower() in {"true", "false", "1", "0"}:
                    value = raw.strip().lower() in {"true", "1"}
                else:
                    raise ValueError()
            elif isinstance(default, int):
                if isinstance(raw, bool) or not isinstance(raw, (str, int)):
                    raise ValueError()
                value = int(raw)
            else:
                if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                    raise ValueError()
                value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid MEMORY_{key.upper()}") from exc
        values[name] = value
    try:
        return SiraConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MEMORY_SIRA configuration: {exc}") from exc


def _build_sira_expander(config):
    model = (os.environ.get("MEMORY_SIRA_MODEL") or os.environ.get("ANSWER_MODEL")
             or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL"))
    key = (os.environ.get("MEMORY_SIRA_API_KEY") or os.environ.get("ANSWER_API_KEY")
           or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY"))
    base = (os.environ.get("MEMORY_SIRA_API_BASE") or os.environ.get("ANSWER_API_BASE")
            or os.environ.get("API_BASE") or os.environ.get("OPENAI_BASE_URL"))
    if not all((model, key, base)):
        if config.enabled:
            raise ValueError("SIRA requires an explicit model, API key and API base")
        return None
    from openai import OpenAI
    return LLMSiraExpander(OpenAI(api_key=key, base_url=base, timeout=config.timeout, max_retries=0), model)


def configure_store(**kwargs: object) -> None:
    """Seed startup settings programmatically (CLI flags), ahead of env defaults.

    Must be called before the first store construction; unknown keys are rejected.
    """
    unknown = sorted(set(kwargs) - _STORE_SETTING_KEYS)
    if unknown:
        raise ValueError(f"Unknown store settings: {', '.join(unknown)}")
    STORE_INIT_OVERRIDES.update(kwargs)


def _build_role_name_extractor(reranker, reranker_model: str, device):
    enabled = STORE_INIT_OVERRIDES.get("role_name_enabled")
    if enabled is None:
        raw = os.environ.get("MEMORY_ROLE_NAME_ENABLED", "false").strip().lower()
        if raw not in {"true", "false", "1", "0"}:
            raise ValueError("MEMORY_ROLE_NAME_ENABLED must be true or false")
        enabled = raw in {"true", "1"}
    if not enabled:
        return None
    model = (os.environ.get("MEMORY_ROLE_NAME_MODEL") or os.environ.get("ANSWER_MODEL")
             or os.environ.get("OPENAI_MODEL"))
    key = (os.environ.get("MEMORY_ROLE_NAME_API_KEY") or os.environ.get("ANSWER_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    base = (os.environ.get("MEMORY_ROLE_NAME_API_BASE") or os.environ.get("ANSWER_API_BASE")
            or os.environ.get("OPENAI_BASE_URL"))
    role_reranker_model = os.environ.get("MEMORY_ROLE_NAME_RERANKER_MODEL", "").strip() or reranker_model
    if not model or not key or not role_reranker_model:
        raise ValueError("Role names require an LLM model/API key and MEMORY_ROLE_NAME_RERANKER_MODEL "
                         "or MEMORY_RERANKER_MODEL")
    timeout = float(os.environ.get("MEMORY_ROLE_NAME_TIMEOUT", "60"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("MEMORY_ROLE_NAME_TIMEOUT must be positive and finite")
    if role_reranker_model != reranker_model or reranker is None:
        reranker = build_reranker(role_reranker_model, device=device)
    from openai import OpenAI

    return EvidenceRoleNameExtractor(
        OpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=0), model, reranker,
        top_k=int(os.environ.get("MEMORY_ROLE_NAME_TOP_K", "8")),
        window=int(os.environ.get("MEMORY_ROLE_NAME_WINDOW", "1")),
        context_chars=int(os.environ.get("MEMORY_ROLE_NAME_CONTEXT_CHARS", "800")),
        max_tokens=int(os.environ.get("MEMORY_ROLE_NAME_MAX_TOKENS", "2500")),
        disable_thinking=os.environ.get("MEMORY_ROLE_NAME_DISABLE_THINKING", "false").lower() == "true",
    )


def _build_conflict_strategy():
    enabled = STORE_INIT_OVERRIDES.get("conflict_enabled")
    if enabled is None:
        raw = os.environ.get("MEMORY_CONFLICT_ENABLED", "false").strip().lower()
        if raw not in {"true", "false", "1", "0"}:
            raise ValueError("MEMORY_CONFLICT_ENABLED must be true or false")
        enabled = raw in {"true", "1"}
    if type(enabled) is not bool:
        raise ValueError("conflict_enabled must be a boolean")
    ratios = []
    for suffix in ("positive", "negative", "scope"):
        key = f"conflict_{suffix}_ratio"
        value = (STORE_INIT_OVERRIDES[key] if key in STORE_INIT_OVERRIDES
                 else float(os.environ.get(f"MEMORY_CONFLICT_{suffix.upper()}_RATIO", "0.2")))
        ratios.append(value)
    validate_ratios(*ratios)
    if not enabled:
        return None, *ratios
    model = (os.environ.get("MEMORY_CONFLICT_MODEL") or os.environ.get("ANSWER_MODEL")
             or os.environ.get("OPENAI_MODEL"))
    key = (os.environ.get("MEMORY_CONFLICT_API_KEY") or os.environ.get("ANSWER_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    base = (os.environ.get("MEMORY_CONFLICT_API_BASE") or os.environ.get("ANSWER_API_BASE")
            or os.environ.get("OPENAI_BASE_URL"))
    if not model or not key:
        raise ValueError("Conflict strategy requires an LLM model and API key")
    timeout = float(os.environ.get("MEMORY_CONFLICT_TIMEOUT", "60"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("MEMORY_CONFLICT_TIMEOUT must be positive and finite")
    max_tokens = int(os.environ.get("MEMORY_CONFLICT_MAX_TOKENS", "2000"))
    if max_tokens <= 0:
        raise ValueError("MEMORY_CONFLICT_MAX_TOKENS must be positive")
    from openai import OpenAI

    return (LLMConflictStrategy(
        OpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=0), model,
        max_tokens=max_tokens,
        disable_thinking=os.environ.get("MEMORY_CONFLICT_DISABLE_THINKING", "false").lower() == "true",
    ), *ratios)


@lru_cache(maxsize=1)
def get_store() -> SQLiteMemoryStore:
    mempursuit_config = _build_mempursuit_config()
    rank_fusion_config = _build_rank_fusion_config()
    long_question_config = _build_long_question_config()
    long_question_extractor = _build_long_question_extractor(long_question_config)
    sira_config = _build_sira_config()
    sira_expander = _build_sira_expander(sira_config)
    near_config = _build_near_config()
    conflict_strategy, positive_ratio, negative_ratio, scope_ratio = _build_conflict_strategy()
    model = STORE_INIT_OVERRIDES.get("embedding_model", os.environ.get("EMBEDDING_MODEL"))
    device = STORE_INIT_OVERRIDES.get("embedding_device", os.environ.get("EMBEDDING_DEVICE") or None)
    query_prefix = STORE_INIT_OVERRIDES.get("embedding_query_prefix", os.environ.get("EMBEDDING_QUERY_PREFIX"))
    database = STORE_INIT_OVERRIDES.get("database", os.environ.get("MEMORY_DB_PATH", "artifacts/memory.db"))
    chunk_tokens = STORE_INIT_OVERRIDES.get("chunk_tokens", int(os.environ.get("MEMORY_CHUNK_TOKENS", "320")))
    overlap_tokens = STORE_INIT_OVERRIDES.get("chunk_overlap", int(os.environ.get("MEMORY_CHUNK_OVERLAP", "40")))
    retrieval_mode = STORE_INIT_OVERRIDES.get(
        "retrieval_mode", os.environ.get("MEMORY_RETRIEVAL_MODE", "hybrid")
    )
    retrieval_mode = str(retrieval_mode).lower()
    rrf_k = STORE_INIT_OVERRIDES.get("rrf_k", int(os.environ.get("MEMORY_RRF_K", "60")))
    lexical_weight = STORE_INIT_OVERRIDES.get(
        "lexical_weight", float(os.environ.get("MEMORY_LEXICAL_WEIGHT", "0.5"))
    )
    context_messages = STORE_INIT_OVERRIDES.get(
        "context_messages", int(os.environ.get("MEMORY_CONTEXT_MESSAGES", "0"))
    )
    option_view_weight = STORE_INIT_OVERRIDES.get(
        "option_view_weight", float(os.environ.get("MEMORY_OPTION_VIEW_WEIGHT", "0"))
    )
    reranker_model = str(
        STORE_INIT_OVERRIDES.get("reranker_model", os.environ.get("MEMORY_RERANKER_MODEL", ""))
    ).strip()
    reranker_settings = reranker_options(STORE_INIT_OVERRIDES)
    if reranker_settings["backend"] == "vllm" and not reranker_model:
        raise ValueError("MEMORY_RERANKER_MODEL is required for the vllm backend")
    reranker_device = STORE_INIT_OVERRIDES.get("reranker_device") or os.environ.get("MEMORY_RERANKER_DEVICE") or device
    reranker_batch_size = STORE_INIT_OVERRIDES.get(
        "reranker_batch_size", os.environ.get("MEMORY_RERANKER_BATCH_SIZE", "32")
    )
    reranker_max_length = STORE_INIT_OVERRIDES.get(
        "reranker_max_length", os.environ.get("MEMORY_RERANKER_MAX_LENGTH", "512")
    )
    rerank_candidates = STORE_INIT_OVERRIDES.get(
        "rerank_candidates", int(os.environ.get("MEMORY_RERANK_CANDIDATES", "200"))
    )
    raw_threshold = STORE_INIT_OVERRIDES.get(
        "rerank_score_threshold", os.environ.get("MEMORY_RERANK_SCORE_THRESHOLD", "")
    )
    rerank_score_threshold = None
    if raw_threshold is not None and str(raw_threshold).strip():
        try:
            rerank_score_threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("MEMORY_RERANK_SCORE_THRESHOLD must be a finite number or empty") from exc
        if not math.isfinite(rerank_score_threshold):
            raise ValueError("MEMORY_RERANK_SCORE_THRESHOLD must be a finite number or empty")
    inject_add_timestamp = STORE_INIT_OVERRIDES.get(
        "inject_add_timestamp",
        os.environ.get("MEMORY_INJECT_ADD_TIMESTAMP", "false").lower() == "true",
    )
    temporal_enrichment_override = STORE_INIT_OVERRIDES.get("temporal_enrichment")
    temporal_enrichment = (
        bool(temporal_enrichment_override)
        if temporal_enrichment_override is not None
        else os.environ.get("MEMORY_TEMPORAL_ENRICHMENT", "false").lower() == "true"
    )
    resolve_relative_time_override = STORE_INIT_OVERRIDES.get("resolve_relative_time_on_add")
    resolve_relative_time_on_add = (
        bool(resolve_relative_time_override)
        if resolve_relative_time_override is not None
        else os.environ.get("MEMORY_RESOLVE_RELATIVE_TIME_ON_ADD", "false").lower() == "true"
    )
    result_window = STORE_INIT_OVERRIDES.get("result_window", int(os.environ.get("MEMORY_RESULT_WINDOW", "1")))
    result_window_seed_k = STORE_INIT_OVERRIDES.get(
        "result_window_seed_k", int(os.environ.get("MEMORY_RESULT_WINDOW_SEED_K", "20"))
    )
    pack_span = STORE_INIT_OVERRIDES.get("pack_span", int(os.environ.get("MEMORY_PACK_SPAN", "1")))
    reranker = (
        build_reranker(
            reranker_model,
            device=reranker_device,
            batch_size=int(reranker_batch_size),
            max_length=int(reranker_max_length),
            **reranker_settings,
        )
        if reranker_model
        else None
    )
    role_name_extractor = _build_role_name_extractor(reranker, reranker_model, reranker_device)
    return SQLiteMemoryStore(
        database,
        build_embedder(model, device, query_prefix, **embedding_options(STORE_INIT_OVERRIDES)),
        chunk_tokens=int(chunk_tokens),
        overlap_tokens=int(overlap_tokens),
        retrieval_mode=retrieval_mode,
        rrf_k=int(rrf_k),
        lexical_weight=float(lexical_weight),
        context_messages=int(context_messages),
        option_view_weight=float(option_view_weight),
        reranker=reranker,
        rerank_candidates=int(rerank_candidates),
        rerank_small2big_enabled=_stage_boolean("rerank_small2big_enabled"),
        mempursuit_small2big_enabled=_stage_boolean("mempursuit_small2big_enabled"),
        rerank_score_threshold=rerank_score_threshold,
        temporal_enrichment=bool(temporal_enrichment),
        inject_add_timestamp=inject_add_timestamp,
        resolve_relative_time_on_add=resolve_relative_time_on_add,
        result_window=int(result_window),
        result_window_seed_k=int(result_window_seed_k),
        pack_span=int(pack_span),
        role_name_extractor=role_name_extractor,
        conflict_strategy=conflict_strategy,
        conflict_positive_ratio=positive_ratio,
        conflict_negative_ratio=negative_ratio,
        conflict_scope_ratio=scope_ratio,
        near_config=near_config,
        mempursuit_config=mempursuit_config,
        rank_fusion_config=rank_fusion_config,
        sira_config=sira_config, sira_expander=sira_expander,
        long_question_config=long_question_config, long_question_extractor=long_question_extractor,
    )


def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.environ.get("MEMORY_API_KEY", "")
    if not expected:
        if os.environ.get("ALLOW_UNAUTHENTICATED", "false").lower() == "true":
            return
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MEMORY_API_KEY is not configured")
    provided = x_api_key
    if not provided and authorization:
        scheme, separator, credential = authorization.partition(" ")
        if separator and scheme.lower() in {"bearer", "token"}:
            provided = credential
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def require_management_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not os.environ.get("MEMORY_API_KEY"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MEMORY_API_KEY is not configured")
    require_api_key(authorization, x_api_key)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("SKIP_MODEL_STARTUP", "false").lower() != "true":
        get_store()
    yield


app = FastAPI(title="PaiMem Memory API", version="0.1.0", lifespan=lifespan)


class RequestTimingMiddleware:
    """Log one elapsed time through the final response body, without request data."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope['type'] != 'http' or scope.get('method') != 'POST'
                or scope.get('path') not in {'/add', '/search'}
                or os.environ.get('MEMORY_REQUEST_TIMING_ENABLED', '').lower() not in {'1', 'true'}):
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        trace_id = uuid4().hex
        status_code = 500
        logged = False

        def log_once():
            nonlocal logged
            if logged:
                return
            logged = True
            logging.getLogger('paimem.request_timing').warning(
                'paimem_request_timing %s', json.dumps({
                    'trace_id': trace_id, 'path': scope['path'],
                    'status_code': status_code,
                    'elapsed_ms': round((time.perf_counter() - started) * 1000, 3),
                }, separators=(',', ':')))

        async def timed_send(message):
            nonlocal status_code
            if message['type'] == 'http.response.start':
                status_code = message['status']
            await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                log_once()

        try:
            await self.app(scope, receive, timed_send)
        finally:
            # Unhandled failures may be converted to 500 by the outer server middleware.
            log_once()


app.add_middleware(RequestTimingMiddleware)


validation_logger = logging.getLogger("paimem.validation")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    safe_errors = [
        {
            "loc": list(error.get("loc", ())),
            "type": error.get("type"),
            "msg": error.get("msg"),
        }
        for error in exc.errors()
    ]
    validation_logger.warning(
        "request validation failed path=%s errors=%s",
        request.url.path,
        safe_errors,
    )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.get("/health", response_model=HealthResponse)
def health(store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> HealthResponse:
    return HealthResponse(**store.stats())


@app.post("/clear", response_model=ClearResponse, dependencies=[Depends(require_management_api_key)])
def clear_memory(store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> ClearResponse:
    return ClearResponse(**store.clear())


@app.post("/add", response_model=AddResponse, dependencies=[Depends(require_api_key)])
def add_memory(payload: AddRequest, store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> AddResponse:
    messages = [MemoryMessage(role=item.role, content=item.content, timestamp=item.timestamp) for item in payload.messages]
    try:
        store.add(payload.request_id, payload.user_id, payload.session_id, messages)
    except RequestConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return AddResponse(
        request_id=payload.request_id,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )


@app.post("/search", response_model=SearchResponse, response_model_exclude_unset=True,
          dependencies=[Depends(require_api_key)])
def search_memory(payload: SearchRequest, store: Annotated[SQLiteMemoryStore, Depends(get_store)],
                  request: Request) -> SearchResponse:
    path = _local_debug_path(request, payload.debug_output_path) if payload.debug_output_path else None
    if path is None:
        hits = store.search(payload.user_id, payload.query, payload.top_k, payload.options)
    else:
        hits, detail = store.search(payload.user_id, payload.query, payload.top_k, payload.options,
                                    diagnostics=True)
        _write_debug(path, detail)
    # Only offline callers opt into queue metadata. Construct the four public
    # fields explicitly so diagnostic metadata is returned only when requested.
    items = []
    for hit in hits:
        fields = dict(id=hit.id, content=hit.content, score=hit.score, created_at=hit.created_at)
        if payload.include_recall_queue or path is not None:
            fields["recall_queue"] = hit.recall_queue
        items.append(SearchItem(**fields))
    return SearchResponse(data=items)


@app.get("/config", dependencies=[Depends(require_management_api_key)])
def get_retrieval_config(store: Annotated[SQLiteMemoryStore, Depends(get_store)]) -> dict[str, object]:
    return store.get_config()


@app.post("/config", dependencies=[Depends(require_management_api_key)])
def patch_retrieval_config(
    payload: ConfigPatch,
    store: Annotated[SQLiteMemoryStore, Depends(get_store)],
) -> dict[str, object]:
    patch = {key: value for key, value in payload.model_dump().items() if value is not None}
    if patch:
        try:
            store.update_config(**patch)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return store.get_config()
