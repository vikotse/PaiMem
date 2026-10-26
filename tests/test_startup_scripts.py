"""Exercise Bash workflows without network, model downloads, or GPU access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StartupScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='paimem scripts ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('scripts', 'configs', 'src/paimem', 'fake-bin'):
            (self.root / name).mkdir(parents=True)
        for name in ('scripts/config.sh', 'scripts/setup_env.sh', 'scripts/start_vllm.sh',
                     'start_svr.sh', 'configs/default.toml',
                     'configs/baseline.toml', 'src/paimem/launcher.py', 'src/paimem/database_check.py', 'src/paimem/configuration.py', 'src/paimem/long_question.py'):
            shutil.copyfile(ROOT / name, self.root / name)
        (self.root / 'src/paimem/__init__.py').write_text('')
        self.log = self.root / 'commands.jsonl'
        self.env = {
            'PATH': str(self.root / 'fake-bin') + os.pathsep + os.environ['PATH'],
            'HOME': os.environ['HOME'], 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': str(self.root / 'src'), 'TEST_LOG': str(self.log),
        }
        self.install_mock('uv', '''
args = sys.argv[1:]
if args[:2] == ['python', 'find']:
    print(sys.executable)
    raise SystemExit(0)
with open(os.environ['TEST_LOG'], 'a') as f:
    f.write(json.dumps(['uv', *args]) + '\\n')
if args and args[0] in ('sync', 'venv'):
    directory = pathlib.Path('.venv' if args[0] == 'sync' else args[-1])
    (directory / 'bin').mkdir(parents=True, exist_ok=True)
    (directory / 'bin/python').symlink_to(sys.executable) if not (directory / 'bin/python').exists() else None
    vllm = directory / 'bin/vllm'
    vllm.write_text('#!' + sys.executable + '\\nimport sys,os,json\\nwith open(os.environ["TEST_LOG"], "a") as f: f.write(json.dumps(["vllm", *sys.argv[1:]]) + chr(10))\\n')
    vllm.chmod(0o755)
if args[:2] == ['tool', 'run']:
    target = pathlib.Path(args[args.index('--local-dir') + 1])
    target.mkdir(parents=True, exist_ok=True)
    (target / 'config.json').write_text('{}')
''')
        self.install_mock('uname', "print('Linux')")
        self.install_mock('nvidia-smi', "print('GPU 0: mock')")
        self.install_mock('curl', '''
with open(os.environ['TEST_LOG'], 'a') as f:
    f.write(json.dumps(['curl', *sys.argv[1:]]) + '\\n')
if '-o' in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text('mock template')
else:
    raise SystemExit('Unexpected network/installer request')
''')
        self.settings = {
            'MEMORY_API_KEY': 'private memory key', 'EMBEDDING_BACKEND': 'remote',
            'EMBEDDING_MODEL': 'remote-model', 'EMBEDDING_BASE_URL': 'https://example.com/v1',
            'EMBEDDING_API_KEY': 'private embedding key', 'MEMORY_RERANKER_BACKEND': 'vllm',
            'MEMORY_RERANKER_MODEL': 'served-model', 'MEMORY_RERANKER_BASE_URL': 'http://localhost:8089',
            'VLLM_DEPLOYMENT': 'remote', 'MEMORY_RERANKER_API_KEY': '',
        }
        self.write_settings()

    def install_mock(self, name, body):
        path = self.root / 'fake-bin' / name
        path.write_text('#!' + sys.executable + '\nimport sys,os,json,pathlib\n' + body)
        path.chmod(0o755)

    def write_settings(self, **extra):
        self.settings.update(extra)
        import tomllib
        from paimem.configuration import SCHEMA
        config = tomllib.loads((ROOT / 'configs/default.toml').read_text())
        lines = []
        for section, fields in SCHEMA.items():
            table = config
            for part in section.split('.'):
                table = table[part]
            lines.append('[' + section + ']')
            for field, (key, default) in fields.items():
                value = self.settings.get(key, table.get(field, default))
                if type(default) is bool and isinstance(value, str):
                    value = value.lower() in ('true', '1')
                elif type(default) in (int, float):
                    value = type(default)(value)
                lines.append(field + ' = ' + json.dumps(value))
        (self.root / 'configs/config.local.toml').write_text('\n'.join(lines) + '\n')

    def run_script(self, script, *args, ok=True, env=None):
        result = subprocess.run(['bash', str(self.root / script), *args], cwd='/tmp',
                                env={**self.env, **(env or {})}, text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        for secret in ('private memory key', 'private embedding key'):
            self.assertNotIn(secret, result.stdout + result.stderr)
        return result

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_remote_embedding_existing_vllm_never_downloads(self):
        # Stale local download settings must not cause downloads in remote mode.
        self.write_settings(EMBEDDING_REPO_ID='should-not-download', RERANKER_REPO_ID='also-not-download')
        self.run_script('scripts/setup_env.sh')
        commands = self.commands()
        self.assertTrue(any(c[1] == 'sync' and '--frozen' in c for c in commands))
        self.assertFalse(any('download' in c or 'venv' in c or 'install' in c[1:2] for c in commands))
        result = self.run_script('start_svr.sh', '--check')
        self.assertEqual(json.loads(result.stdout)['embedding_backend'], 'remote')
        self.run_script('scripts/start_vllm.sh', ok=False)

    def test_local_models_download_to_separate_paths_with_revision(self):
        self.write_settings(EMBEDDING_BACKEND='local', EMBEDDING_MODEL='models/embedding with space',
                            EMBEDDING_REPO_ID='org/embedding', EMBEDDING_REVISION='embedding-sha',
                            MEMORY_RERANKER_BACKEND='cross-encoder',
                            MEMORY_RERANKER_MODEL='models/reranker', RERANKER_REPO_ID='org/reranker',
                            RERANKER_REVISION='reranker-sha')
        self.run_script('scripts/setup_env.sh')
        downloads = [c for c in self.commands() if 'download' in c]
        self.assertEqual(len(downloads), 2)
        self.assertEqual(downloads[0][-4:], ['--local-dir', 'models/embedding with space', '--revision', 'embedding-sha'])
        self.assertEqual(downloads[1][-4:], ['--local-dir', 'models/reranker', '--revision', 'reranker-sha'])
        self.assertFalse(any('vllm==0.32.0' in c for c in self.commands()))
        self.write_settings(EMBEDDING_REPO_ID='', RERANKER_REPO_ID='')
        self.run_script('scripts/setup_env.sh')
        self.assertEqual(len([c for c in self.commands() if 'download' in c]), 2)

    def test_baseline_skips_reranker_and_cli_config_wins(self):
        self.write_settings(VLLM_DEPLOYMENT='local', MEMORY_RERANKER_MODEL='', RERANKER_REPO_ID='unused')
        self.run_script('scripts/setup_env.sh', '--config', 'configs/baseline.toml', env=self.settings)
        self.assertFalse(any('download' in c or 'venv' in c for c in self.commands()))
        result = self.run_script('start_svr.sh', '--config', 'configs/baseline.toml', '--check', env=self.settings)
        self.assertFalse(json.loads(result.stdout)['reranker'])

    def test_local_vllm_and_template_are_prepared_and_start_receives_settings(self):
        self.write_settings(VLLM_DEPLOYMENT='local', RERANKER_REPO_ID='org/reranker',
                            RERANKER_MODEL_PATH='models/reranker', RERANKER_CHAT_TEMPLATE='models/template.jinja',
                            RERANKER_CHAT_TEMPLATE_URL='https://example.com/template.jinja',
                            MEMORY_RERANKER_API_KEY='rerank-key', VLLM_PORT='9001', VLLM_MAX_MODEL_LEN='2048')
        self.run_script('scripts/setup_env.sh')
        commands = self.commands()
        self.assertEqual(len([c for c in commands if 'download' in c]), 1)
        self.assertTrue(any('vllm==0.32.0' in c for c in commands))
        self.assertEqual(len([c for c in commands if c[0] == 'curl']), 1)
        self.run_script('scripts/start_vllm.sh', '--disable-log-requests')
        args = self.commands()[-1]
        self.assertEqual(args[:3], ['vllm', 'serve', 'models/reranker'])
        for key, value in [('--port', '9001'), ('--max-model-len', '2048'),
                           ('--served-model-name', 'served-model'), ('--api-key', 'rerank-key')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertEqual(args[-1], '--disable-log-requests')
        self.run_script('scripts/setup_env.sh')
        self.assertEqual(len([c for c in self.commands() if c[0] == 'curl']), 1)

    def test_vllm_only_does_not_need_memory_or_embedding_secrets(self):
        self.settings = {'MEMORY_RERANKER_BACKEND': 'vllm', 'VLLM_DEPLOYMENT': 'local',
                         'MEMORY_RERANKER_MODEL': 'served', 'RERANKER_REPO_ID': 'org/reranker',
                         'RERANKER_MODEL_PATH': 'models/reranker',
                         'RERANKER_CHAT_TEMPLATE': 'models/template.jinja',
                         'RERANKER_CHAT_TEMPLATE_URL': 'https://example.com/template', 'MEMORY_RERANKER_API_KEY': ''}
        self.write_settings()
        self.run_script('scripts/setup_env.sh', '--only', 'vllm')
        self.assertFalse(any('sync' in c for c in self.commands()))
        self.assertEqual(len([c for c in self.commands() if 'download' in c]), 1)

    def test_export_precedence_empty_values_and_missing_file(self):
        self.run_script('scripts/setup_env.sh')
        # Explicitly exported baseline beats config.local.toml; empty secret must not fall back to the file.
        result = self.run_script('start_svr.sh', '--check', env={**self.settings, 'PAIMEM_CONFIG': 'configs/baseline.toml'})
        self.assertFalse(json.loads(result.stdout)['reranker'])
        failure = self.run_script('start_svr.sh', '--check', env={'MEMORY_API_KEY': ''}, ok=False)
        self.assertIn('MEMORY_API_KEY', failure.stderr)
        failure = self.run_script('start_svr.sh', '--check', env={'PAIMEM_CONFIG': 'configs/missing.toml'}, ok=False)
        self.assertIn('missing.toml', failure.stderr)
        (self.root / 'configs/config.local.toml').unlink()
        self.run_script('start_svr.sh', '--check', env=self.settings)

    def test_missing_local_model_and_bad_gpu_fail_clearly(self):
        self.write_settings(EMBEDDING_BACKEND='local', EMBEDDING_MODEL='missing', EMBEDDING_REPO_ID='')
        failure = self.run_script('scripts/setup_env.sh', ok=False)
        self.assertIn('config.json', failure.stderr)
        self.install_mock('uname', "print('Darwin')")
        self.write_settings(VLLM_DEPLOYMENT='local')
        failure = self.run_script('scripts/setup_env.sh', '--only', 'vllm', ok=False)
        self.assertIn('Linux', failure.stderr)

    def test_missing_vllm_field_and_service_only(self):
        self.write_settings(VLLM_DEPLOYMENT='local')
        self.run_script('scripts/setup_env.sh', '--only', 'service')
        self.assertFalse(any('download' in c or 'venv' in c for c in self.commands()))
        failure = self.run_script('scripts/setup_env.sh', '--only', 'vllm', env={'RERANKER_MODEL_PATH': ''}, ok=False)
        self.assertIn('RERANKER_MODEL_PATH', failure.stderr)
        self.assertNotIn('unbound variable', failure.stderr)

    def test_vllm_cannot_install_into_service_environment_alias(self):
        self.run_script('scripts/setup_env.sh')
        (self.root / 'service-alias').symlink_to(self.root / '.venv', target_is_directory=True)
        self.write_settings(VLLM_DEPLOYMENT='local', VLLM_ENV='service-alias',
                            RERANKER_MODEL_PATH='models/reranker', RERANKER_CHAT_TEMPLATE='models/template.jinja')
        failure = self.run_script('scripts/setup_env.sh', '--only', 'vllm', ok=False)
        self.assertIn('separate', failure.stderr)
        self.assertFalse(any('vllm==0.32.0' in c for c in self.commands()))

    def test_custom_runtime_file_and_empty_backend_validation(self):
        self.run_script('scripts/setup_env.sh')
        custom = self.root / 'configs/custom config.local.toml'
        shutil.copyfile(self.root / 'configs/baseline.toml', custom)
        result = self.run_script('start_svr.sh', '--check', env={
            **self.settings, 'PAIMEM_CONFIG': str(custom)})
        self.assertFalse(json.loads(result.stdout)['reranker'])
        failure = self.run_script('scripts/setup_env.sh', env={'EMBEDDING_BACKEND': ''}, ok=False)
        self.assertIn('EMBEDDING_BACKEND', failure.stderr)
