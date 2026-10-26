"""Check stored embedding compatibility using only the standard library."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Mapping


def configured_identity(env: Mapping[str, str]) -> str:
    """Mirror built-in embedder identities without constructing either backend."""
    backend = env.get('EMBEDDING_BACKEND', 'local')
    model = env.get('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
    prefix = env.get('EMBEDDING_QUERY_PREFIX')
    if backend == 'remote':
        identity = 'remote:' + json.dumps(
            [env['EMBEDDING_BASE_URL'].strip().rstrip('/'), model.strip(), prefix or ''],
            separators=(',', ':'))
    else:
        if prefix is None and 'bge' in model.lower() and '-en-' in model.lower():
            prefix = 'Represent this sentence for searching relevant passages: '
        identity = f'{model}|query_prefix={prefix or ""}'
    context_messages = int(env.get('MEMORY_CONTEXT_MESSAGES', '0'))
    if context_messages:
        identity += f'|context=v1:{context_messages}'
    return identity


def _identity_fields(identity: str) -> dict[str, object]:
    """Decode known formats only to name changed fields, never to print values."""
    context_messages = 0
    base, separator, count = identity.rpartition('|context=v1:')
    if separator and count.isdigit():
        identity, context_messages = base, int(count)
    fields: dict[str, object] = {'context.messages': context_messages}
    if identity.startswith('remote:'):
        values = json.loads(identity[len('remote:'):])
        if not isinstance(values, list) or len(values) != 3 or not all(isinstance(v, str) for v in values):
            raise ValueError('Unknown remote identity format')
        fields.update(zip(('embedding.remote.base_url', 'embedding.remote.model',
                           'embedding.remote.query_prefix'), values))
        fields['embedding.backend'] = 'remote'
    else:
        model, separator, prefix = identity.partition('|query_prefix=')
        if not separator:
            raise ValueError('Unknown local identity format')
        fields.update({'embedding.backend': 'local', 'embedding.local.model_path': model,
                       'embedding.local.query_prefix': prefix})
    return fields


def _changed_fields(stored: str, configured: str) -> str:
    try:
        old, new = _identity_fields(stored), _identity_fields(configured)
    except (ValueError, TypeError):
        return 'embedding identity (legacy or unrecognized format)'
    if old['embedding.backend'] != new['embedding.backend']:
        return 'embedding.backend'
    changes = [field for field in new if old.get(field) != new[field]]
    return ', '.join(changes) or 'embedding identity'


def check_database(env: Mapping[str, str]) -> str:
    """Read existing metadata without creating a database or updating any rows.

    The store's transactional check remains authoritative at initialization.
    This preflight catches configuration mistakes before model loading.
    """
    path = Path(env.get('MEMORY_DB_PATH', 'artifacts/memory.db')).resolve()
    # stat distinguishes a missing file from a permissions/read error.
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 'new'
    if size == 0 and path.is_file():
        return 'empty'

    expected = configured_identity(env)
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)) as connection:
            connection.execute('PRAGMA query_only = ON')
            connection.execute('BEGIN')
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            has_data = any(
                connection.execute(f'SELECT EXISTS(SELECT 1 FROM {table})').fetchone()[0]
                for table in ('memories', 'add_requests') if table in tables)
            if not has_data:
                return 'empty'
            stored = None
            if 'store_metadata' in tables:
                row = connection.execute(
                    "SELECT value FROM store_metadata WHERE key = 'embedding_identity'").fetchone()
                stored = row[0] if row else None
    except sqlite3.Error:
        raise ValueError(
            f'Cannot inspect memory database {str(path)!r} read-only. '
            'Check service.db_path, file permissions, and SQLite integrity. '
            '无法只读检查记忆数据库，请检查路径、权限及数据库完整性。') from None

    if stored == expected:
        return 'compatible'
    fields = _changed_fields(stored, expected) if isinstance(stored, str) else 'missing embedding_identity metadata'
    raise ValueError(
        f'Memory database {str(path)!r} is incompatible with the configured embedding settings. '
        f'Changed or unverifiable fields: {fields}. '
        'Set service.db_path (MEMORY_DB_PATH) to a NEW database and re-ingest the original data, '
        'or restore the original embedding settings. Existing vectors are not converted automatically; '
        'do not edit the stored identity to bypass this check. '
        '记忆数据库与当前 embedding 配置不兼容：请改用新的 service.db_path 并重新写入原始数据，'
        '或恢复原配置；已有向量不会自动转换。')
