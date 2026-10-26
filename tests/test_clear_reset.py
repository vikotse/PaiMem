"""Full PaiMem SQLite store reset, using temporary files."""
import importlib
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paimem.memory_store import reset_memory_database


class Embedder:
    identity = "reset-test"

    def encode_documents(self, texts):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

    encode_queries = encode_documents


class ClearResetTests(unittest.TestCase):
    def stores(self):
        for package in ("paimem",):
            with tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(package + ".memory_store")
                path = Path(directory) / "memory.db"
                options = {}
                store = module.SQLiteMemoryStore(path, Embedder(), **options)
                store.add("same-id", "user", "session", [module.MemoryMessage("user", "old experiment")])
                yield package, module, store, path

    def test_all_tables_reset_compaction_and_request_id_reuse(self):
        for package, module, store, path in self.stores():
            with self.subTest(package=package):
                store.search("user", "old", 3)
                with closing(sqlite3.connect(path)) as connection:
                    connection.executescript('''
                        CREATE TABLE temporal_turns(id INTEGER PRIMARY KEY, content TEXT);
                        CREATE TABLE temporal_resolutions(turn_id INTEGER REFERENCES temporal_turns(id), payload TEXT);
                        INSERT INTO temporal_turns VALUES(1, 'old private conversation');
                        INSERT INTO temporal_resolutions VALUES(1, 'old resolution');
                        CREATE TABLE "extra""table"(id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB,
                            request_id TEXT REFERENCES add_requests(request_id) ON DELETE RESTRICT);
                        INSERT INTO "extra""table"(payload, request_id) VALUES(zeroblob(4000000), 'same-id');
                        INSERT INTO store_metadata VALUES('old_experiment', 'old value');
                    ''')
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    before = path.stat().st_size
                cache = store._near_cache.invalidate
                owner = store._near_cache
                name = "invalidate"
                with patch.object(owner, name, wraps=cache) as invalidated:
                    result = store.clear()
                    invalidated.assert_called_once_with()
                self.assertEqual(result, dict(deleted_requests=1, deleted_memories=1, requests=0, memories=0, users=0))
                with closing(sqlite3.connect(path)) as connection:
                    tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    for (name,) in tables:
                        if name == "store_metadata":
                            continue
                        quoted = '"' + name.replace('"', '""') + '"'
                        self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0], 0, name)
                    self.assertEqual(connection.execute("SELECT * FROM store_metadata").fetchall(), [("embedding_identity", store.embedding_identity)])
                    self.assertEqual(connection.execute("PRAGMA freelist_count").fetchone()[0], 0)
                    self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertLess(path.stat().st_size, before // 2)
                wal = Path(str(path) + "-wal")
                self.assertTrue(not wal.exists() or wal.stat().st_size == 0)
                self.assertEqual(store.search("user", "old", 3), [])
                self.assertEqual(store.clear()["deleted_memories"], 0)
                store.add("same-id", "user", "session", [module.MemoryMessage("user", "new experiment")])
                hits = store.search("user", "new", 3)
                self.assertEqual(len(hits), 1)
                self.assertIn("new experiment", hits[0].content)
                self.assertNotIn("old experiment", hits[0].content)

    def test_delete_failure_rolls_back_and_restores_foreign_keys(self):
        for package, _, store, _ in self.stores():
            with self.subTest(package=package), closing(store._connect()) as connection:
                connection.executescript('''
                    CREATE TABLE legacy(payload TEXT);
                    INSERT INTO legacy VALUES('preserved');
                    CREATE TRIGGER refuse_clear BEFORE DELETE ON memories
                        BEGIN SELECT RAISE(ABORT, 'refuse reset'); END;
                ''')
                with self.assertRaises(sqlite3.IntegrityError):
                    reset_memory_database(connection, store.embedding_identity, lambda: self.fail("must not invalidate before commit"))
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT payload FROM legacy").fetchone()[0], "preserved")
                self.assertEqual(store.stats()["memories"], 1)

    def test_compaction_failure_reports_error_after_cache_invalidation(self):
        for package, _, store, _ in self.stores():
            with self.subTest(package=package), closing(store._connect()) as connection:
                # VACUUM attaches a temporary database internally.
                connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK)
                invalidated = []
                with self.assertRaises(sqlite3.DatabaseError):
                    reset_memory_database(connection, store.embedding_identity, lambda: invalidated.append(True))
                self.assertEqual(invalidated, [True])
                self.assertEqual(store.stats()["memories"], 0)
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                connection.set_authorizer(None)
                self.assertEqual(store.clear()["memories"], 0)

    def test_trigger_repopulation_is_detected_and_rolled_back(self):
        for package, _, store, _ in self.stores():
            with self.subTest(package=package), closing(store._connect()) as connection:
                connection.executescript('''
                    CREATE TABLE aaa_legacy(payload TEXT);
                    CREATE TRIGGER repopulate AFTER DELETE ON memories
                        BEGIN INSERT INTO aaa_legacy VALUES('stale'); END;
                ''')
                with self.assertRaisesRegex(RuntimeError, "not fully cleared"):
                    store.clear()
                self.assertEqual(store.stats()["memories"], 1)

    def test_busy_wal_checkpoint_fails_and_can_be_retried(self):
        for package, _, store, path in self.stores():
            with self.subTest(package=package), closing(sqlite3.connect(path)) as reader:
                reader.execute("BEGIN")
                reader.execute("SELECT * FROM memories").fetchall()
                with closing(store._connect()) as connection:
                    connection.execute("PRAGMA busy_timeout = 0")
                    invalidated = []
                    with self.assertRaisesRegex(RuntimeError, "checkpoint is busy"):
                        reset_memory_database(connection, store.embedding_identity, lambda: invalidated.append(True))
                    self.assertEqual(invalidated, [True])
                reader.rollback()
                self.assertEqual(store.clear()["memories"], 0)

    def test_both_clear_endpoints_and_failure_response(self):
        from fastapi.testclient import TestClient
        for package, _, store, path in self.stores():
            api = importlib.import_module(package + ".api")
            with self.subTest(package=package), patch.dict(os.environ, {"MEMORY_API_KEY": "reset-secret", "SKIP_MODEL_STARTUP": "true"}):
                api.app.dependency_overrides[api.get_store] = lambda: store
                try:
                    with TestClient(api.app, raise_server_exceptions=False) as client:
                        headers = {"X-Api-Key": "reset-secret"}
                        self.assertEqual(client.post("/clear").status_code, 401)
                        response = client.post("/clear", headers=headers)
                        self.assertEqual(response.status_code, 200)
                        self.assertTrue(response.json()["success"])
                        self.assertEqual(response.json()["deleted_memories"], 1)
                        self.assertEqual(client.get("/health").json()["memories"], 0)
                        with patch.object(store, "clear", side_effect=RuntimeError("compaction failed")):
                            self.assertEqual(client.post("/clear", headers=headers).status_code, 500)
                finally:
                    api.app.dependency_overrides.pop(api.get_store, None)


if __name__ == "__main__":
    unittest.main()
