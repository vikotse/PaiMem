"""NEAR input selection and reproducibility without model downloads."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scripts import run_near_ablation as runner


class FixedEmbedder:
    identity = 'near-input-test'
    model = SimpleNamespace(tokenizer=None)

    def encode_documents(self, texts):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return self.encode_documents(texts)


class NearInputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.data = self.root / 'custom splits'
        self.data.mkdir()
        self.enterContext(patch.dict(os.environ))

    def write_inputs(self, phase):
        for dataset in runner.DATASETS:
            record = dict(id=f'{dataset}-{phase}', session_id=f'session-{phase}',
                          messages=[dict(role='user', content='I like tea.')],
                          questions=[dict(id='q1', question='What do I like?',
                                          answer='tea', evidence_texts=['I like tea.'])])
            (self.data / f'{dataset}.{phase}.jsonl').write_text(json.dumps(record) + '\n')

    def args(self, phase, output):
        return SimpleNamespace(phase=phase, data_dir=self.data, output=output,
                               selection=None, embedding_model='test-model', device='cpu',
                               locomo_dev_records=3, locomo_dev_questions=8,
                               longmemeval_dev_records=16)

    def test_default_and_explicit_cli_directory(self):
        with patch.object(runner, 'run_retrieval') as run:
            runner.main(['--phase', 'dev', '--output', str(self.root/'out')])
            self.assertEqual(run.call_args.args[0].data_dir, runner.ROOT/'data/splits')
            runner.main(['--phase', 'validation', '--output', str(self.root/'out'),
                         '--data-dir', str(self.data)])
            self.assertEqual(run.call_args.args[0].data_dir, self.data)

    def test_relative_directory_resolves_from_working_directory(self):
        self.write_inputs('dev')
        with patch('os.getcwd', return_value=str(self.root)):
            paths = runner.retrieval_inputs(Path('custom splits'), 'dev')
        self.assertEqual(paths, {d: self.data/f'{d}.dev.jsonl' for d in runner.DATASETS})

    def test_missing_inputs_fail_before_output_or_model_loading(self):
        output = self.root/'output'
        for missing_directory in (True, False):
            if missing_directory:
                self.data.rmdir()
            else:
                self.data.mkdir()
                (self.data/'locomo_refined.dev.jsonl').write_text('{}\n')
                # A directory with the expected name must not count as an input file.
                (self.data/'longmemeval_s.dev.jsonl').mkdir()
            with self.subTest(missing_directory=missing_directory), patch.object(runner, 'build_embedder') as build:
                with self.assertRaisesRegex(ValueError, 'longmemeval_s.dev.jsonl') as error:
                    runner.run_retrieval(self.args('dev', output))
                self.assertIn('--data-dir', str(error.exception))
                self.assertFalse(output.exists())
                build.assert_not_called()

    def test_cli_reports_missing_inputs_without_traceback(self):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as exit_status:
            runner.main(['--phase', 'dev', '--data-dir', str(self.data),
                         '--output', str(self.root/'output')])
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn('locomo_refined.dev.jsonl', errors.getvalue())
        self.assertIn('longmemeval_s.dev.jsonl', errors.getvalue())
        self.assertNotIn('Traceback', errors.getvalue())
        self.assertFalse((self.root/'output').exists())

    def test_custom_inputs_drive_both_records_and_manifest_hashes(self):
        for phase in ('dev', 'validation'):
            self.write_inputs(phase)
        with patch.object(runner, 'build_embedder', return_value=FixedEmbedder()), redirect_stdout(io.StringIO()):
            for phase in ('dev', 'validation'):
                args = self.args(phase, self.root/phase)
                if phase == 'validation':
                    args.selection = self.root/'dev/selection.json'
                runner.run_retrieval(args)
                manifest = json.loads((args.output/'manifest.json').read_text())
                for dataset in runner.DATASETS:
                    raw = (self.data/f'{dataset}.{phase}.jsonl').read_bytes()
                    self.assertEqual(manifest['input_sha256'][dataset], hashlib.sha256(raw).hexdigest())
                    records = runner.adapters.read_rows(args.output/dataset/'selected.jsonl')
                    self.assertEqual(records[0]['id'], f'{dataset}-{phase}')
                    if phase == 'validation':
                        for arm in ('hybrid', 'near'):
                            items = runner.adapters.read_rows(args.output/dataset/arm/'official-input.jsonl')
                            self.assertEqual(items[0]['id'], f'{dataset}-{phase}:q1')
        path = self.data/'locomo_refined.dev.jsonl'
        path.write_text(path.read_text() + '\n')
        with patch.object(runner, 'build_embedder') as build:
            with self.assertRaisesRegex(ValueError, 'Configuration/source/input changed'):
                runner.run_retrieval(self.args('dev', self.root/'dev'))
            build.assert_not_called()

    def test_e2e_uses_saved_outputs_without_reading_split_directory(self):
        with patch.object(runner, 'run_e2e') as run, patch.object(runner, 'retrieval_inputs') as inputs:
            runner.main(['--phase', 'e2e', '--output', str(self.root/'validation'),
                         '--data-dir', str(self.root/'absent')])
            run.assert_called_once()
            inputs.assert_not_called()
