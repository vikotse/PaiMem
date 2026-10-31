import copy
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch
from paimem.configuration import SCHEMA, default_path, prepare

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.env = {'MEMORY_API_KEY': 'test', 'EMBEDDING_BACKEND': 'local',
                    'MEMORY_RERANKER_API_KEY': '', 'VLLM_DEPLOYMENT': 'remote'}

    def test_profiles_and_external_overrides(self):
        env = {**self.env, 'MEMORY_LEXICAL_WEIGHT': '0.7'}
        service, result = prepare(ROOT / 'configs/default.toml', env)
        self.assertEqual(result['MEMORY_LEXICAL_WEIGHT'], '0.7')
        self.assertEqual(result['MEMORY_MEMPURSUIT_ENABLED'], 'true')
        self.assertEqual(result['EMBEDDING_BATCH_SIZE'], '64')
        self.assertEqual(result['EMBEDDING_MODEL'], 'models/bge-small-en-v1.5')
        self.assertNotIn('EMBEDDING_QUERY_PREFIX', result)
        service, result = prepare(ROOT / 'configs/baseline.toml', env)
        self.assertFalse(service['reranker'])
        self.assertNotIn('MEMORY_RERANKER_MODEL', result)
        self.assertEqual(result['MEMORY_RERANKER_BACKEND'], 'cross-encoder')
        self.assertEqual(result['MEMORY_PACK_SPAN'], '1')

    def test_remote_defaults_and_overrides(self):
        env = {**self.env, 'EMBEDDING_BACKEND': 'remote', 'EMBEDDING_BASE_URL': 'https://example.com/v1',
               'EMBEDDING_API_KEY': 'test'}
        _, result = prepare(ROOT / 'configs/default.toml', env)
        self.assertEqual(result['EMBEDDING_MODEL'], 'text-embedding-v4')
        self.assertEqual(result['EMBEDDING_BATCH_SIZE'], '10')
        self.assertNotIn('EMBEDDING_REPO_ID', result)
        _, result = prepare(ROOT / 'configs/default.toml', {**env, 'EMBEDDING_BATCH_SIZE': '3'})
        self.assertEqual(result['EMBEDDING_BATCH_SIZE'], '3')

    def test_remote_max_input_tokens_configuration(self):
        env = {**self.env, 'EMBEDDING_BACKEND': 'remote',
               'EMBEDDING_BASE_URL': 'https://example.com/v1', 'EMBEDDING_API_KEY': 'test'}
        for profile in ('default', 'baseline'):
            _, result = prepare(ROOT / f'configs/{profile}.toml', env)
            self.assertEqual(result['EMBEDDING_MAX_INPUT_TOKENS'], '8192')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.toml'
            path.write_text('[embedding.remote]\nmax_input_tokens = 4096\n')
            _, result = prepare(path, env)
            self.assertEqual(result['EMBEDDING_MAX_INPUT_TOKENS'], '4096')
            _, result = prepare(path, {**env, 'EMBEDDING_MAX_INPUT_TOKENS': '2048'})
            self.assertEqual(result['EMBEDDING_MAX_INPUT_TOKENS'], '2048')
            for value in ('0', '-1', '1.5', 'bad'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    prepare(path, {**env, 'EMBEDDING_MAX_INPUT_TOKENS': value})
        _, result = prepare(ROOT / 'configs/default.toml', self.env)
        self.assertNotIn('EMBEDDING_MAX_INPUT_TOKENS', result)

    def test_remote_failure_log_path_configuration(self):
        env = {**self.env, 'EMBEDDING_BACKEND': 'remote',
               'EMBEDDING_BASE_URL': 'https://example.com/v1', 'EMBEDDING_API_KEY': 'test'}
        for profile in ('default', 'baseline'):
            _, result = prepare(ROOT / f'configs/{profile}.toml', env)
            self.assertEqual(result['EMBEDDING_FAILURE_LOG_PATH'], '')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.toml'
            path.write_text('[embedding.remote]\nfailure_log_path = "artifacts/embedding-failures.jsonl"\n')
            _, result = prepare(path, env)
            self.assertEqual(result['EMBEDDING_FAILURE_LOG_PATH'], 'artifacts/embedding-failures.jsonl')
            for value in ('artifacts/custom-failures.jsonl', ''):
                _, result = prepare(path, {**env, 'EMBEDDING_FAILURE_LOG_PATH': value})
                self.assertEqual(result['EMBEDDING_FAILURE_LOG_PATH'], value)
            path.write_text('[embedding.remote]\nfailure_log_path = false\n')
            with self.assertRaises(ValueError):
                prepare(path, env)
        _, result = prepare(ROOT / 'configs/default.toml', self.env)
        self.assertNotIn('EMBEDDING_FAILURE_LOG_PATH', result)

    def test_active_placeholder_credentials_only(self):
        with self.assertRaisesRegex(ValueError, r'\[service\].api_key'):
            prepare(ROOT / 'configs/default.toml', {})
        with self.assertRaisesRegex(ValueError, 'EMBEDDING_BASE_URL'):
            prepare(ROOT / 'configs/baseline.toml', {**self.env, 'EMBEDDING_BACKEND': 'remote'})
        # Inactive remote embedding / LLM placeholders do not prevent local startup.
        prepare(ROOT / 'configs/default.toml', self.env)
        for feature in ('SIRA', 'ROLE_NAME', 'CONFLICT'):
            with self.assertRaisesRegex(ValueError, feature + '_API_BASE'):
                prepare(ROOT / 'configs/default.toml', {**self.env, f'MEMORY_{feature}_ENABLED': 'true'})
        # vLLM-only setup does not need memory/embedding credentials.
        prepare(ROOT / 'configs/default.toml', {'MEMORY_RERANKER_API_KEY': ''}, scope='vllm')

    def test_invalid_combination_and_numbers(self):
        with self.assertRaisesRegex(ValueError, 'Rank fusion'):
            prepare(ROOT / 'configs/baseline.toml', {**self.env, 'MEMORY_RANK_FUSION_ENABLED': 'true'})
        for key, value in [('EMBEDDING_BATCH_SIZE', '0'), ('PAIMEM_PORT', '65536'),
                           ('PAIMEM_RERANKER_ENABLED', 'bad'), ('MEMORY_RANK_FUSION_DENSE_WEIGHT', 'NaN')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare(ROOT / 'configs/default.toml', {**self.env, key: value})

    def test_unknown_fields_and_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.toml'
            for content in ('[embedding]\nbackend="local"\nmodle="typo"',
                            '[service]\nport="8000"', '[embedding]\nlocal="oops"'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    prepare(path, self.env)

    def test_no_secret_leak_and_shell_quoting(self):
        secret = "a '$() ; `do not run` secret"
        env = {**os.environ, **self.env, 'MEMORY_API_KEY': secret}
        result = subprocess.run([sys.executable, str(ROOT / 'src/paimem/configuration.py'),
                                 '--config', str(ROOT / 'configs/default.toml')], env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(secret, result.stdout + result.stderr)
        exported = subprocess.run([sys.executable, str(ROOT / 'src/paimem/configuration.py'),
                                   '--config', str(ROOT / 'configs/default.toml'), '--shell'], env=env,
                                  text=True, capture_output=True, check=True)
        checked = subprocess.run(['bash', '-c', exported.stdout + '\nprintf "%s" "$MEMORY_API_KEY"'],
                                 text=True, capture_output=True, check=True)
        self.assertEqual(checked.stdout, secret)

    def test_local_file_selection_and_export_override(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'is_file', return_value=True):
            self.assertEqual(default_path(), Path('configs/config.local.toml'))
        with patch.dict(os.environ, {'PAIMEM_CONFIG': 'custom.toml'}, clear=True):
            self.assertEqual(default_path(), Path('custom.toml'))

    def test_example_keys_all_map_and_strategy_values_are_stable(self):
        for profile in ('default', 'baseline'):
            cfg = tomllib.loads((ROOT / f'configs/{profile}.toml').read_text())
            for section, fields in SCHEMA.items():
                table = cfg
                for part in section.split('.'):
                    table = table[part]
                # The endpoint example is commented out so setup can auto-select it.
                expected = set(fields) - ({'hf_endpoint'} if section == 'environment' else set())
                self.assertEqual(set(table) - {'local', 'remote', 'vllm'}, expected)
            _, env = prepare(ROOT / f'configs/{profile}.toml', self.env)
            self.assertEqual(env['MEMORY_MEMPURSUIT_CANDIDATES'], '150')
            self.assertEqual(env['MEMORY_RRF_K'], '60')
            self.assertEqual(env['MEMORY_RERANK_CANDIDATES'], '400')

    def test_vllm_batch_size_reaches_api_and_can_be_overridden(self):
        from paimem import api
        default_size = tomllib.loads((ROOT / 'configs/default.toml').read_text())['reranker']['vllm']['batch_size']
        for size in (default_size, 7):
            overrides = self.env if size == default_size else {**self.env, 'MEMORY_RERANKER_BATCH_SIZE': str(size)}
            _, env = prepare(ROOT / 'configs/default.toml', overrides)
            with patch.dict(os.environ, env, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), \
                    patch.object(api, 'build_embedder'), patch.object(api, 'build_reranker') as build, \
                    patch.object(api, 'SQLiteMemoryStore'):
                api.get_store.cache_clear()
                try:
                    api.get_store()
                    self.assertEqual(build.call_args.kwargs['batch_size'], size)
                    self.assertEqual(build.call_args.kwargs['max_length'], 1024)
                finally:
                    api.get_store.cache_clear()
        with self.assertRaisesRegex(ValueError, 'MEMORY_RERANKER_BATCH_SIZE'):
            prepare(ROOT / 'configs/default.toml', {**self.env, 'MEMORY_RERANKER_BATCH_SIZE': '0'})

    def test_long_question_config_reaches_request_and_validates_json(self):
        from paimem import api
        env = {**self.env, 'MEMORY_LONG_QUESTION_ENABLED': 'true',
               'MEMORY_LONG_QUESTION_API_BASE': 'https://example.com/v1',
               'MEMORY_LONG_QUESTION_API_KEY': 'test-key',
               'MEMORY_LONG_QUESTION_EXTRA_BODY': '{"thinking":{"type":"disabled"}}'}
        _, effective = prepare(ROOT / 'configs/default.toml', env)
        with patch.dict(os.environ, effective, clear=True), patch.dict(api.STORE_INIT_OVERRIDES, {}, clear=True), \
                patch('openai.OpenAI') as client:
            config = api._build_long_question_config()
            self.assertEqual((config.threshold_chars, config.head_chars, config.tail_chars, config.timeout),
                             (8000, 2000, 2000, 30))
            extractor = api._build_long_question_extractor(config)
            choice = client.return_value.chat.completions.create.return_value.choices = [unittest.mock.Mock(finish_reason='stop')]
            choice[0].message.content = '{"query":"target"}'
            self.assertEqual(extractor.extract('head', 'tail'), 'target')
            self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs['extra_body'],
                             {'thinking': {'type': 'disabled'}})
        for key, value in [('MEMORY_LONG_QUESTION_EXTRA_BODY', '[]'),
                           ('MEMORY_LONG_QUESTION_EXTRA_BODY', '{"x":NaN}'),
                           ('MEMORY_LONG_QUESTION_EXTRA_BODY', 'broken'),
                           ('MEMORY_LONG_QUESTION_THRESHOLD_CHARS', '0'),
                           ('MEMORY_LONG_QUESTION_HEAD_CHARS', '7000'),
                           ('MEMORY_LONG_QUESTION_TIMEOUT', '-1')]:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'long_question'):
                prepare(ROOT / 'configs/default.toml', {**env, key: value})

    def test_template_fields_match_schema_types(self):
        # Profiles may tune values independently of the loader's fallback defaults.
        for profile in ('default', 'baseline'):
            config = tomllib.loads((ROOT / f'configs/{profile}.toml').read_text())
            for section, fields in SCHEMA.items():
                table = config
                for part in section.split('.'):
                    table = table[part]
                for field, (_, fallback) in fields.items():
                    with self.subTest(profile=profile, section=section, field=field):
                        if section == 'environment' and field == 'hf_endpoint':
                            self.assertNotIn(field, table)
                            continue
                        allowed = (int, float) if type(fallback) is float else (type(fallback),)
                        self.assertIn(type(table[field]), allowed)

    def test_missing_fields_use_loader_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'minimal.toml'
            path.write_text('')
            _, env = prepare(path, {**self.env, 'EMBEDDING_BACKEND': 'remote',
                                   'EMBEDDING_BASE_URL': 'https://example.com/v1',
                                   'EMBEDDING_API_KEY': 'test'})
        self.assertEqual(env['EMBEDDING_RPM'], '0')
        self.assertEqual(env['EMBEDDING_TPM'], '0')
        self.assertEqual(env['MEMORY_RERANKER_BATCH_SIZE'], '16')

    def test_diagnostics_directory_configuration(self):
        _, env = prepare(ROOT / 'configs/default.toml', self.env)
        self.assertEqual(env['MEMORY_DIAGNOSTICS_DIR'], 'artifacts/diagnostics')
        _, env = prepare(ROOT / 'configs/default.toml', {**self.env, 'MEMORY_DIAGNOSTICS_DIR': '/tmp/traces'})
        self.assertEqual(env['MEMORY_DIAGNOSTICS_DIR'], '/tmp/traces')
        with self.assertRaisesRegex(ValueError, 'MEMORY_DIAGNOSTICS_DIR'):
            prepare(ROOT / 'configs/default.toml', {**self.env, 'MEMORY_DIAGNOSTICS_DIR': ''})
