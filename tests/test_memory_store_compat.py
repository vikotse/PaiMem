"""New stores and existing databases after retiring the preference strategy."""
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paimem.memory_store import MemoryMessage, SQLiteMemoryStore

LEGACY_TABLES = ("memory_turn_links", "memory_turns", "preference_profiles", "preference_profile_state")
LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_turns (
    request_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    source_timestamp INTEGER,
    PRIMARY KEY(request_id, turn_index),
    FOREIGN KEY(request_id) REFERENCES add_requests(request_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_turn_links (
    memory_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(request_id, turn_index)
        REFERENCES memory_turns(request_id, turn_index) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS preference_profile_state (
    user_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    revision INTEGER NOT NULL,
    last_memory_rowid INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS preference_profiles (
    user_id TEXT NOT NULL,
    role_key TEXT NOT NULL,
    role_name TEXT,
    profile_text TEXT NOT NULL,
    entries_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(user_id, role_key),
    FOREIGN KEY(user_id) REFERENCES preference_profile_state(user_id) ON DELETE CASCADE
);
"""


class Embedder:
    identity = "compat-test"

    def encode_documents(self, texts):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

    encode_queries = encode_documents


class StoreCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.db"
        self.store = SQLiteMemoryStore(self.path, Embedder())

    def tables(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def snapshot(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return {table: connection.execute(f'SELECT * FROM "{table}"').fetchall()
                    for table in self.tables()}

    def add_legacy_data(self):
        self.store.add("old", "u", "s", [MemoryMessage("user", "old memory")])
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(LEGACY_SCHEMA)
            mid = connection.execute("SELECT id FROM memories").fetchone()[0]
            connection.execute("INSERT INTO memory_turns VALUES ('old',0,'user','old memory',NULL)")
            connection.execute("INSERT INTO memory_turn_links VALUES (?,'old',0)", (mid,))
            connection.execute("INSERT INTO preference_profile_state VALUES ('u','global',1,1)")
            connection.execute("INSERT INTO preference_profiles VALUES ('u','',NULL,'old profile','[]','old')")
            connection.commit()

    def test_new_store_add_search_clear_do_not_create_retired_tables(self):
        self.store.add("r", "u", "s", [MemoryMessage("user", "I like basketball")])
        hits, detail = self.store.search("u", "What sport do I like?", 3, diagnostics=True)
        self.assertEqual(hits[0].recall_queue, "semantic")
        self.assertNotIn("preference_strategy", detail)
        self.assertFalse(any(k.startswith("preference_") for k in self.store.get_config()))
        self.assertFalse(set(LEGACY_TABLES) & self.tables())
        self.assertEqual(self.store.clear()["deleted_memories"], 1)
        self.assertEqual(self.store.stats(), dict(requests=0, memories=0, users=0))
        self.assertFalse(set(LEGACY_TABLES) & self.tables())

    def test_existing_database_load_search_and_add_preserve_legacy_data(self):
        self.add_legacy_data()
        before = self.snapshot()
        reopened = SQLiteMemoryStore(self.path, Embedder())
        self.assertEqual(reopened.search("u", "old", 3)[0].content, "[user] old memory")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(reopened.add("old", "u", "s", [MemoryMessage("user", "old memory")]), 1)
        self.assertEqual(before, self.snapshot())
        reopened.add("new", "u", "s", [MemoryMessage("user", "new memory")])
        after = self.snapshot()
        for table in LEGACY_TABLES:
            self.assertEqual(before[table], after[table])
        self.assertEqual(reopened.stats()["memories"], 2)

    def test_explicit_clear_removes_legacy_data_and_preserves_schema_and_identity(self):
        self.add_legacy_data()
        before = self.snapshot()
        result = self.store.clear()
        after = self.snapshot()
        self.assertEqual(result["deleted_requests"], 1)
        self.assertEqual(result["deleted_memories"], 1)
        self.assertEqual(set(before), set(after))
        self.assertEqual(before["store_metadata"], after["store_metadata"])
        self.assertTrue(all(not rows for table, rows in after.items() if table != "store_metadata"))
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.store.add("new", "u", "s", [MemoryMessage("user", "new memory")])
        self.assertTrue(all(not self.snapshot()[table] for table in LEGACY_TABLES))

    def test_clear_failure_rolls_back_legacy_and_active_data(self):
        self.add_legacy_data()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("""CREATE TRIGGER refuse_clear BEFORE DELETE ON memories
                                  BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
            connection.commit()
        before = self.snapshot()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.clear()
        self.assertEqual(before, self.snapshot())

    def test_partial_legacy_schema_clears_without_recreating_missing_tables(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE preference_profile_state (user_id TEXT)")
            connection.execute("INSERT INTO preference_profile_state VALUES ('u')")
            connection.commit()
        self.store.clear()
        self.assertEqual(self.snapshot()["preference_profile_state"], [])
        self.assertEqual(set(LEGACY_TABLES) & self.tables(), {"preference_profile_state"})

    def test_stale_environment_does_not_initialize_preference_llm(self):
        from paimem import api
        with patch.dict(os.environ, {"MEMORY_DB_PATH": str(self.path),
                "MEMORY_PREFERENCE_ENABLED": "true", "MEMORY_PREFERENCE_MODEL": "retired",
                "MEMORY_PREFERENCE_API_KEY": "test"}, clear=True), \
                patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), \
                patch.object(api, "build_embedder", return_value=Embedder()), \
                patch("openai.OpenAI") as client:
            api.get_store.cache_clear()
            try:
                store = api.get_store()
                store.add("r", "u", "s", [MemoryMessage("user", "favorite sport")])
                store.search("u", "favorite sport?", 3)
                client.assert_not_called()
                self.assertNotIn("preference_enabled", api._STORE_SETTING_KEYS)
                self.assertFalse(any(k.startswith("preference_") for k in store.get_config()))
            finally:
                api.get_store.cache_clear()


if __name__ == "__main__":
    unittest.main()
