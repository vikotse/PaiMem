"""Embedding preflight checks must not load models, call APIs, or mutate data."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from paimem.database_check import check_database, configured_identity
from paimem.configuration import prepare
from paimem.launcher import main

ROOT = Path(__file__).resolve().parents[1]


class DatabaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Exercise URI escaping and paths containing spaces.
        self.path = Path(self.tmp.name) / 'memory #1?.db'
        self.env = {
            'MEMORY_DB_PATH': str(self.path), 'EMBEDDING_BACKEND': 'remote',
            'EMBEDDING_MODEL': 'model-a', 'EMBEDDING_BASE_URL': 'https://example.com/v1',
            'EMBEDDING_API_KEY': 'secret-never-print', 'MEMORY_CONTEXT_MESSAGES': '0',
        }

    def create_database(self, *, memories=True, requests=True, metadata=True, identity=None):
        with sqlite3.connect(self.path) as db:
            db.executescript('CREATE TABLE memories(id TEXT, dimension INTEGER); '
                             'CREATE TABLE add_requests(request_id TEXT); '
                             'CREATE TABLE store_metadata(key TEXT PRIMARY KEY, value TEXT);')
            if memories:
                db.execute("INSERT INTO memories VALUES ('memory-1', 768)")
            if requests:
                db.execute("INSERT INTO add_requests VALUES ('request-1')")
            if metadata:
                db.execute('INSERT INTO store_metadata VALUES (?, ?)',
                           ('embedding_identity', identity or configured_identity(self.env)))

    def test_new_and_zero_length_database_are_not_created_or_populated(self):
        env = {**self.env, 'MEMORY_DB_PATH': str(self.path / 'missing' / 'memory.db')}
        self.assertEqual(check_database(env), 'new')
        self.assertFalse(self.path.exists())
        self.path.touch()
        self.assertEqual(check_database(self.env), 'empty')
        self.assertEqual(self.path.read_bytes(), b'')

    def test_compatible_database_is_unchanged(self):
        self.create_database()
        before = self.path.read_bytes()
        self.assertEqual(check_database(self.env), 'compatible')
        self.assertEqual(self.path.read_bytes(), before)

    def test_each_identity_change_is_actionable_and_preserves_data(self):
        self.create_database()
        before = self.path.read_bytes()
        for key, value, field in (
            ('EMBEDDING_MODEL', 'model-b', 'embedding.remote.model'),
            ('EMBEDDING_BASE_URL', 'https://other.example.com/v1', 'embedding.remote.base_url'),
            ('EMBEDDING_QUERY_PREFIX', 'query: ', 'embedding.remote.query_prefix'),
            ('EMBEDDING_BACKEND', 'local', 'embedding.backend'),
            ('MEMORY_CONTEXT_MESSAGES', '1', 'context.messages'),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                check_database({**self.env, key: value})
            message = str(caught.exception)
            self.assertIn(field, message)
            self.assertIn(str(self.path), message)
            self.assertIn('service.db_path', message)
            self.assertIn('re-ingest', message)
            self.assertNotIn(self.env['EMBEDDING_API_KEY'], message)
            self.assertEqual(self.path.read_bytes(), before)

    def test_operational_settings_and_endpoint_trailing_slash_can_change(self):
        self.create_database()
        self.assertEqual(check_database({**self.env,
            'EMBEDDING_API_KEY': 'new-secret', 'EMBEDDING_BATCH_SIZE': '2',
            'EMBEDDING_RPM': '100', 'EMBEDDING_TPM': '1000',
            'EMBEDDING_DEVICE': 'cpu', 'EMBEDDING_BASE_URL': 'https://example.com/v1/'}), 'compatible')

    def test_empty_database_accepts_new_model_without_updating_metadata(self):
        self.create_database(memories=False, requests=False)
        before = self.path.read_bytes()
        self.assertEqual(check_database({**self.env, 'EMBEDDING_MODEL': 'model-b'}), 'empty')
        self.assertEqual(self.path.read_bytes(), before)

    def test_either_memories_or_requests_prevents_switch(self):
        for memories, requests in ((True, False), (False, True)):
            with self.subTest(memories=memories, requests=requests):
                self.create_database(memories=memories, requests=requests)
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    check_database({**self.env, 'EMBEDDING_MODEL': 'model-b'})
                self.path.unlink()

    def test_missing_metadata_and_corrupt_database_fail_without_repair(self):
        self.create_database(metadata=False)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'missing embedding_identity'):
            check_database(self.env)
        self.assertEqual(self.path.read_bytes(), before)
        self.path.write_bytes(b'not a sqlite database')
        with self.assertRaisesRegex(ValueError, 'read-only'):
            check_database(self.env)
        self.assertEqual(self.path.read_bytes(), b'not a sqlite database')

    def test_uncheckpointed_wal_is_included(self):
        self.create_database(memories=False, requests=False)
        with contextlib.closing(sqlite3.connect(self.path)) as writer:
            writer.execute('PRAGMA journal_mode = WAL')
            writer.execute('PRAGMA wal_autocheckpoint = 0')
            writer.execute("INSERT INTO memories VALUES ('wal-memory', 768)")
            writer.commit()
            with self.assertRaisesRegex(ValueError, 'embedding.remote.model'):
                check_database({**self.env, 'EMBEDDING_MODEL': 'model-b'})
            self.assertEqual(writer.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 1)

    def test_identity_matches_built_in_backends_without_model_loading(self):
        from paimem.embeddings import RemoteEmbedder, SentenceTransformerEmbedder, build_embedder
        fake_st = Mock()
        with patch.dict(sys.modules, {'sentence_transformers': SimpleNamespace(SentenceTransformer=fake_st)}), \
                patch.dict(os.environ, {}, clear=True):
            for model in ('models/bge-small-en-v1.5', 'org/other-model'):
                for prefix in (None, '', 'query: '):
                    env = {'EMBEDDING_BACKEND': 'local', 'EMBEDDING_MODEL': model}
                    if prefix is not None:
                        env['EMBEDDING_QUERY_PREFIX'] = prefix
                    actual = SentenceTransformerEmbedder(model, query_prefix=prefix)
                    for context in (0, 2):
                        suffix = f'|context=v1:{context}' if context else ''
                        self.assertEqual(configured_identity({**env, 'MEMORY_CONTEXT_MESSAGES': str(context)}),
                                         actual.identity + suffix)
            env = {**self.env, 'EMBEDDING_MODEL': ' model-a ',
                   'EMBEDDING_BASE_URL': ' https://example.com/v1/ '}
            actual = build_embedder(env['EMBEDDING_MODEL'], backend='remote',
                                    base_url=env['EMBEDDING_BASE_URL'], api_key='secret')
            self.assertIsInstance(actual, RemoteEmbedder)
            self.assertEqual(configured_identity(env), actual.identity)

    def test_effective_environment_overrides_file_and_local_auto_prefix(self):
        overrides = {**self.env, 'MEMORY_API_KEY': 'secret', 'MEMORY_RERANKER_API_KEY': ''}
        _, effective = prepare(ROOT / 'configs/default.toml', overrides)
        self.create_database()
        self.assertEqual(check_database(effective), 'compatible')
        _, effective = prepare(ROOT / 'configs/default.toml', {
            **overrides, 'EMBEDDING_BACKEND': 'local', 'EMBEDDING_MODEL': 'models/bge-small-en-v1.5'})
        self.assertNotIn('EMBEDDING_QUERY_PREFIX', effective)
        self.assertIn('Represent this sentence', configured_identity(effective))

    def test_check_and_startup_fail_before_uvicorn_or_model_loading(self):
        self.create_database()
        uvicorn = SimpleNamespace(run=Mock())
        for args in (['--check'], []):
            with self.subTest(args=args), patch('paimem.launcher.prepare', return_value=(
                    {}, {**self.env, 'EMBEDDING_MODEL': 'model-b'})), \
                    patch.dict(sys.modules, {'uvicorn': uvicorn}), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as caught:
                    main(args)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn('embedding.remote.model', err.getvalue())
                uvicorn.run.assert_not_called()

    def test_check_success_reports_database_status(self):
        self.create_database()
        with patch('paimem.launcher.prepare', return_value=({}, self.env)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--check'])
        self.assertEqual(json.loads(output.getvalue())['database_status'], 'compatible')
        self.assertNotIn(self.env['EMBEDDING_API_KEY'], output.getvalue())
