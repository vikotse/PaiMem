"""Access controls and filesystem boundaries for management and diagnostics."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from paimem import api


class ApiAccessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'diagnostics'
        self.output.mkdir()
        self.store = Mock()
        self.store.get_config.return_value = {'mempursuit_enabled': False}
        self.store.search.return_value = ([], {'candidates': []})
        self.enterContext(patch.dict(os.environ, {
            'MEMORY_API_KEY': 'test', 'SKIP_MODEL_STARTUP': 'true',
            'MEMORY_DIAGNOSTICS_DIR': str(self.output),
        }, clear=True))
        self.enterContext(patch.dict(api.app.dependency_overrides, {api.get_store: lambda: self.store}))
        self.client = self.enterContext(TestClient(api.app, client=('127.0.0.1', 12345)))

    def diagnostic(self, path, **kwargs):
        return self.client.post('/search', json={
            'query': 'q', 'user_id': 'u', 'top_k': 1, 'debug_output_path': str(path),
        }, headers={'X-Api-Key': 'test'}, **kwargs)

    def test_config_requires_valid_key_for_reads_and_writes(self):
        for headers in ({}, {'X-Api-Key': 'wrong'}):
            self.assertEqual(self.client.get('/config', headers=headers).status_code, 401)
            self.assertEqual(self.client.post('/config', headers=headers, json={'mempursuit_enabled': True}).status_code, 401)
        self.store.get_config.assert_not_called()
        self.store.update_config.assert_not_called()
        for headers in ({'X-Api-Key': 'test'}, {'Authorization': 'Bearer test'}):
            self.assertEqual(self.client.get('/config', headers=headers).status_code, 200)
            self.assertEqual(self.client.post('/config', headers=headers, json={'mempursuit_enabled': True}).status_code, 200)
        self.assertEqual(self.store.update_config.call_count, 2)

    def test_management_and_diagnostics_do_not_allow_auth_bypass(self):
        os.environ.update(MEMORY_API_KEY='', ALLOW_UNAUTHENTICATED='true')
        self.assertEqual(self.client.get('/config').status_code, 503)
        self.assertEqual(self.client.post('/config', json={}).status_code, 503)
        self.assertEqual(self.diagnostic(self.output / 'trace.json').status_code, 503)
        self.store.search.assert_not_called()

    def test_diagnostics_require_key_and_loopback(self):
        payload = dict(query='q', user_id='u', top_k=1, debug_output_path=str(self.output/'trace.json'))
        self.assertEqual(self.client.post('/search', json=payload).status_code, 401)
        with TestClient(api.app, client=('192.0.2.1', 12345)) as remote:
            self.assertEqual(remote.post('/search', json=payload, headers={'X-Api-Key': 'test'}).status_code, 403)
        self.store.search.assert_not_called()

    def test_nested_output_and_atomic_replacement(self):
        nested = self.output / 'run'
        nested.mkdir()
        path = nested / 'trace.json'
        path.write_text('previous report')
        self.assertEqual(self.diagnostic(path).status_code, 200)
        self.assertEqual(json.loads(path.read_text()), {'candidates': []})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(nested.iterdir()), [path])

    def test_outside_traversal_relative_missing_and_wrong_extension(self):
        outside = self.root / 'outside.json'
        outside.write_text('unchanged')
        sibling = self.root / 'diagnostics-other'
        sibling.mkdir()
        for path in (outside, sibling/'trace.json', self.output/'..'/'outside.json',
                     Path('trace.json'), self.output/'missing'/'trace.json', self.output/'trace.txt'):
            with self.subTest(path=path):
                self.assertEqual(self.diagnostic(path).status_code, 422)
        self.assertEqual(outside.read_text(), 'unchanged')
        self.store.search.assert_not_called()

    def test_symlink_file_and_directory_are_rejected(self):
        outside = self.root / 'outside.json'
        outside.write_text('unchanged')
        (self.output/'file.json').symlink_to(outside)
        (self.output/'link').symlink_to(self.root, target_is_directory=True)
        for path in (self.output/'file.json', self.output/'link'/'outside.json'):
            self.assertEqual(self.diagnostic(path).status_code, 422)
        self.assertEqual(outside.read_text(), 'unchanged')
        self.store.search.assert_not_called()

    def test_directory_swap_during_search_does_not_redirect_write(self):
        nested = self.output / 'run'
        nested.mkdir()
        target = nested / 'trace.json'
        def search(*args, **kwargs):
            nested.rename(self.output/'saved-run')
            nested.symlink_to(self.root, target_is_directory=True)
            return [], {'candidates': []}
        self.store.search.side_effect = search
        self.assertEqual(self.diagnostic(target).status_code, 422)
        self.assertFalse((self.root/'trace.json').exists())

    def test_serialization_failure_cleans_temporary_file_and_preserves_report(self):
        path = self.output / 'trace.json'
        path.write_text('previous report')
        self.store.search.return_value = ([], {'value': float('nan')})
        self.assertEqual(self.diagnostic(path).status_code, 422)
        self.assertEqual(path.read_text(), 'previous report')
        self.assertEqual(list(self.output.iterdir()), [path])
