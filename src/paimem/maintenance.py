import sqlite3
from typing import Callable

def reset_memory_database(connection: sqlite3.Connection, embedding_identity: str, invalidate_cache: Callable[[], None]) -> dict[str, int]:
    """Reset a dedicated memory database; callers must hold their mutation lock.

    Data deletion is atomic. Compaction runs after commit and may fail separately;
    in that case callers must report failure and may safely retry the reset.
    No other process/client may write during this maintenance operation.
    """
    foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    # Unknown legacy tables may have RESTRICT or cyclic foreign keys. Every
    # business table is emptied and checked before constraints are re-enabled.
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        deleted_requests = connection.execute("SELECT COUNT(*) FROM add_requests").fetchone()[0]
        deleted_memories = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        shadows = {row[1] for row in connection.execute("PRAGMA main.table_list") if row[2] == "shadow"}
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) if not row[0].startswith("sqlite_") and row[0] not in shadows]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            connection.execute(f"DELETE FROM {quoted}")
        # Reset sequence counters and ANALYZE samples that can retain old values.
        for (internal_table,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('sqlite_sequence', 'sqlite_stat1', 'sqlite_stat2', 'sqlite_stat3', 'sqlite_stat4')"
        ).fetchall():
            connection.execute(f'DELETE FROM "{internal_table}"')
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            if connection.execute(f"SELECT EXISTS(SELECT 1 FROM {quoted})").fetchone()[0]:
                raise RuntimeError(f"memory database was not fully cleared: {table}")
        connection.execute(
            "INSERT INTO store_metadata(key, value) VALUES ('embedding_identity', ?)",
            (embedding_identity,),
        )
        if connection.execute("PRAGMA foreign_key_check").fetchone():
            raise RuntimeError("memory database has invalid foreign keys after clear")
        connection.commit()
        # Even when compaction fails, committed deletions must invalidate caches.
        invalidate_cache()
        connection.execute("VACUUM")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint[0] != 0:
            raise RuntimeError("memory cleared but WAL checkpoint is busy; stop other clients and retry clear")
        return {
            "deleted_requests": deleted_requests,
            "deleted_memories": deleted_memories,
            "requests": 0,
            "memories": 0,
            "users": 0,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute(f"PRAGMA foreign_keys = {int(foreign_keys)}")
