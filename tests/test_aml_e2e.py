"""Exercise real official pipeline entry points against a controlled HTTP boundary."""
import asyncio
import os
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from scripts import run_aml_e2e as runner


PIPELINES = Path(os.environ.get("AML_PIPELINE_ROOT", str(runner.ROOT / "agent-memory-leaderboard/data")))

@unittest.skipUnless((PIPELINES / "locomo-refined/pipeline.py").is_file(), "Clone the documented AML revision or export AML_PIPELINE_ROOT")
class E2ETests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root/'data.jsonl'
        self.output = self.root/'run'
        self.calls = []
        self.events = io.StringIO()
        self.row = dict(id='q1', question='Where?\u2028中文', gold_answer='Paris', speaker_1_memories='Paris')
        self.answer = 'Paris'
        self.judge = '{"label":"CORRECT"}'
        self.behavior = 'normal'
        self.search_response = {'data': [{'id': 'm1', 'content': 'Paris'}]}
        self.active_memory_requests = 0
        self.max_active_memory_requests = 0

    def tearDown(self):
        self.temp.cleanup()

    async def post(self, client, url, **kwargs):
        body = kwargs['json']
        self.calls.append(body)
        memory_request = url.endswith('/add') or url.endswith('/search')
        if self.behavior == 'slow_memory' and memory_request:
            self.active_memory_requests += 1
            self.max_active_memory_requests = max(
                self.max_active_memory_requests,
                self.active_memory_requests,
            )
            try:
                await asyncio.sleep(0.03)
            finally:
                self.active_memory_requests -= 1
        # Request-start and input estimate must already be logged before POST.
        if '/chat/completions' in url:
            event = json.loads(self.events.getvalue().strip().splitlines()[-1])
            self.assertEqual(event['event'], 'request_start')
            self.assertGreater(event['input_tokens_estimate'], 0)
        if self.behavior == 'hang':
            await asyncio.sleep(10)
        if url.endswith('/add'):
            payload = {'success': True}
        elif url.endswith('/search'):
            payload = self.search_response
            if body.get('debug_output_path') and self.behavior != 'ignore_diagnostics':
                Path(body['debug_output_path']).write_text(json.dumps(dict(
                    schema_version=1, stored_count=2, top_k=body['top_k'],
                    candidates=[dict(id='m1', content='Paris', channels=dict(
                        dense=dict(score=.9, rank=1), fusion=dict(score=.02, rank=1)), returned_rank=1),
                        dict(id='m2', content='Rome', channels=dict(
                        dense=dict(score=.2, rank=2), fusion=dict(score=.01, rank=2)), returned_rank=None)])))
        else:
            content = self.judge if body['model'] == 'judge' else self.answer
            if self.behavior == 'judge_once_bad' and body['model'] == 'judge':
                self.behavior = 'normal'
                content = 'invalid JSON'
            payload = {'choices':[{'message':{'content':content}}]}
            if self.behavior != 'missing_usage':
                payload['usage'] = {'prompt_tokens':100, 'completion_tokens':10}
        return httpx.Response(200, json=payload, request=httpx.Request('POST', url))

    def run_cli(self, family='locomo-refined/pipeline.py', extra=None):
        if not self.input.exists():
            runner.a.write_rows(self.input, [self.row])
        argv = ['--input',str(self.input),'--output',str(self.output),'--input-format','official',
                '--pipeline',str(PIPELINES/family),
                '--answer-base-url','http://mock/v1','--answer-model','answer',
                '--judge-base-url','http://mock/v1','--judge-model','judge',
                '--answer-price-in','2','--answer-price-out','4','--judge-price-in','1','--judge-price-out','3',
                '--retries','0','--heartbeat','0.01'] + (extra or [])
        async def mocked(client, url, **kw):
            return await self.post(client, url, **kw)
        with contextlib.redirect_stderr(self.events), patch.object(httpx.AsyncClient, 'post', mocked):
            return runner.main(argv)

    def report(self):
        return json.loads((self.output/'report.json').read_text())

    def test_binary_resume_totals_and_cap(self):
        self.assertEqual(self.run_cli(extra=['--answer-max-tokens','77']), 0)
        self.assertEqual(self.calls[0]['max_tokens'], 77)
        self.assertNotIn('max_tokens', self.calls[1])
        r = self.report()
        self.assertEqual(r['score_mean'], 1)
        self.assertEqual(r['usage']['input_tokens'], 200)
        self.assertEqual(r['usage']['output_tokens'], 20)
        self.assertAlmostEqual(r['usage']['estimated_cost'], .00037)
        self.assertGreater(r['usage']['request_seconds'], 0)
        self.assertEqual(self.run_cli(extra=['--answer-max-tokens','77']), 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.report()['usage'], r['usage'])

    def test_timeout_is_bounded_and_resume_retries_failure(self):
        self.behavior = 'hang'
        self.assertEqual(self.run_cli(extra=['--call-timeout','.04','--item-timeout','.08']), 1)
        self.assertIn('"event": "waiting"', self.events.getvalue())
        self.assertEqual(self.report()['usage']['unknown_usage'], 1)
        self.assertEqual(self.report()['answered'], 0)
        self.behavior = 'normal'
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.report()['usage']['requests'], 3)

    def test_invalid_judge_retries_and_counts_paid_attempt(self):
        self.behavior = 'judge_once_bad'
        self.assertEqual(self.run_cli(extra=['--retries','1']), 0)
        self.assertEqual(self.report()['usage']['requests'], 3)

    def test_missing_usage_marked_estimated(self):
        self.behavior = 'missing_usage'
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.report()['usage']['estimated_usage'], 2)

    def test_changed_config_refused(self):
        self.run_cli()
        with self.assertRaises(SystemExit):
            self.run_cli(extra=['--top-k','3'])
        self.assertEqual(len(self.calls), 2)

    def test_longmemeval(self):
        self.assertEqual(self.run_cli('longmemeval-s/pipeline.py'), 0)

    def test_beam(self):
        self.row.update(rubric=['Says Paris'], question_type='fact_recall')
        self.judge = '{"scores":[{"index":0,"score":1,"reason":"yes"}]}'
        self.assertEqual(self.run_cli('beam/pipeline.py'), 0)
        self.assertEqual(self.report()['score_mean'], 1)

    def test_clbench(self):
        self.row.update(retrieval={'selected':[{'text':'Paris'}]}, rubrics=['Says Paris'])
        self.judge = '{"Overall Score":1,"List of Requirement Satisfaction Status":["yes"]}'
        self.assertEqual(self.run_cli('clbench/pipeline.py'), 0)
        self.assertEqual(self.report()['score_mean'], 1)

    def test_clbench_failed_judge_not_zero_score(self):
        self.row.update(retrieval={'selected':[]}, rubrics=['Says Paris'])
        self.judge = 'broken'
        self.assertEqual(self.run_cli('clbench/pipeline.py'), 1)
        self.assertIsNone(self.report()['score_mean'])
        self.assertEqual(self.report()['judged'], 0)

    def test_personamem_v1_local_judge(self):
        self.row.update(context_messages=[], all_options='A. Paris\nB. Rome', correct_answer='A. Paris')
        self.answer = '(a)'
        self.assertEqual(self.run_cli('personamem/pipeline_v1.py'), 0)
        self.assertEqual(len(self.calls), 1)

    def test_personamem_v2_local_judge(self):
        self.row.update(chat_history=[], user_query='Where?', correct_answer='Paris', incorrect_answers=['Rome'])
        self.answer = 'A'
        self.assertEqual(self.run_cli('personamem/pipeline_v2.py'), 0)
        self.assertEqual(len(self.calls), 1)

    def test_scriptmem_official_rule_scorer_subset(self):
        gold = self.root/'gold'
        gold.mkdir()
        for name in ('angry','enemy','friends','man_earth'):
            (gold/f'{name}.json').write_text(json.dumps([{'sample_id':'s1','qa':[
                {'qa_type':'single_choice','question':'Where?', 'answer':'A. Paris'}]}]))
        self.row['id'] = 'angry:s1#q0000'
        self.answer = 'A'
        self.assertEqual(self.run_cli('scriptmem/pipeline.py', ['--scriptmem-data-dir',str(gold)]), 0)
        self.assertEqual(self.report()['judged'], 1)
        self.assertEqual(self.report()['score_mean'], 1)
        self.assertEqual(len(self.calls), 1)

    def test_official_clbench_idx(self):
        self.row.pop('id')
        self.row.update(idx='official-id', retrieval={'selected':[]}, rubrics=['Says Paris'])
        self.judge = '{"Overall Score":1,"List of Requirement Satisfaction Status":["yes"]}'
        self.assertEqual(self.run_cli('clbench/pipeline.py'), 0)

    def test_normalized_memory_replay(self):
        runner.a.write_rows(self.input, [dict(id='r1', messages=[{'role':'user','content':'Paris'}],
            questions=[dict(id='q1',question='Where?',answer='Paris', evidence_texts=['Paris'])])])
        self.assertEqual(self.run_cli(extra=['--input-format','normalized']), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertFalse((self.output/'search').exists())
        # Enabling capture on an old run is allowed but does not re-search.
        self.assertEqual(self.run_cli(extra=['--input-format','normalized','--save-search-results']), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertFalse((self.output/'search').exists())
        # The same rule applies when resuming from per-question replay only.
        (self.output/'input.jsonl').unlink()
        self.assertEqual(self.run_cli(extra=['--input-format','normalized','--save-search-results']), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertFalse((self.output/'search').exists())

    def test_normalized_replay_parallelizes_records_and_preserves_output_order(self):
        self.behavior = 'slow_memory'
        runner.a.write_rows(self.input, [
            dict(id='r1', messages=[{'role':'user','content':'Paris'}],
                 questions=[dict(id='q1',question='Where?',answer='Paris')]),
            dict(id='r2', messages=[{'role':'user','content':'Rome'}],
                 questions=[dict(id='q2',question='Where else?',answer='Rome')]),
        ])

        self.assertEqual(
            self.run_cli(extra=['--input-format', 'normalized', '--concurrency', '2']),
            0,
        )

        self.assertGreaterEqual(self.max_active_memory_requests, 2)
        replayed = runner.a.read_rows(self.output / 'input.jsonl')
        self.assertEqual([row['id'] for row in replayed], ['r1:q1', 'r2:q2'])

    def test_replay_concurrency_override_and_legacy_fallback(self):
        parsed = runner.parser().parse_args([
            '--input', str(self.input), '--output', str(self.output),
            '--concurrency', '3',
        ])
        self.assertIsNone(parsed.replay_concurrency)
        self.assertIsNone(parsed.model_concurrency)

        self.behavior = 'slow_memory'
        runner.a.write_rows(self.input, [
            dict(id='r1', messages=[], questions=[dict(id='q1', question='One?', answer='1')]),
            dict(id='r2', messages=[], questions=[dict(id='q2', question='Two?', answer='2')]),
        ])
        self.assertEqual(self.run_cli(extra=[
            '--input-format', 'normalized', '--concurrency', '1',
            '--replay-concurrency', '2', '--model-concurrency', '1',
        ]), 0)
        self.assertGreaterEqual(self.max_active_memory_requests, 2)

    def test_duplicate_normalized_record_ids_fail_before_memory_requests(self):
        runner.a.write_rows(self.input, [
            dict(id='same', messages=[], questions=[dict(id='q1', question='One?', answer='1')]),
            dict(id='same', messages=[], questions=[dict(id='q2', question='Two?', answer='2')]),
        ])
        with self.assertRaisesRegex(ValueError, 'duplicate normalized record IDs'):
            self.run_cli(extra=['--input-format', 'normalized'])
        self.assertEqual(self.calls, [])

    def test_optional_full_search_results_preserve_order_and_metadata(self):
        self.search_response = dict(data=[dict(id='m1', content='Paris', score=.9, metadata={'source': 's1'}),
                                          dict(id='m2', content='Rome', score=.8)], elapsed_ms=15)
        runner.a.write_rows(self.input, [dict(id='r1', messages=[{'role':'user','content':'Paris'}],
            questions=[dict(id='q1',question='Where?',answer='Paris', evidence_texts=['Paris'])])])
        extra = ['--input-format', 'normalized', '--save-search-results']
        self.assertEqual(self.run_cli(extra=extra), 0)
        filename = hashlib.sha256(b'r1:q1').hexdigest() + '.json'
        saved = json.loads((self.output/'search'/filename).read_text())
        self.assertEqual(saved, self.search_response)
        replay = json.loads((self.output/'replay'/filename).read_text())
        self.assertEqual(replay['question'], 'Where?')
        self.assertEqual(replay['retrieved_context'], 'Paris\nRome')
        self.assertNotIn('search_response', replay)
        self.assertNotIn('save_search_results', json.loads((self.output/'manifest.json').read_text()))
        self.assertEqual(self.run_cli(extra=['--input-format', 'normalized']), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(json.loads((self.output/'search'/filename).read_text()), saved)

    def test_optional_local_retrieval_diagnostics(self):
        runner.a.write_rows(self.input, [dict(id='r1', messages=[{'role':'user','content':'Paris'}],
            questions=[dict(id='q1', question='Where?', answer='Paris', evidence_texts=['Paris'])])])
        extra = ['--input-format', 'normalized', '--save-retrieval-diagnostics']
        self.assertEqual(self.run_cli(extra=extra), 0)
        filename = hashlib.sha256(b'r1:q1').hexdigest() + '.json'
        detail = json.loads((self.output/'search_diagnostics'/filename).read_text())
        self.assertEqual(detail['stored_count'], 2)
        self.assertEqual(len(detail['candidates']), 2)
        self.assertFalse((self.output/'search').exists())
        search_request = next(call for call in self.calls if call.get('query') == 'Where?')
        self.assertEqual(search_request['debug_output_path'], str((self.output/'search_diagnostics'/filename).resolve()))
        self.assertNotIn('save_retrieval_diagnostics', json.loads((self.output/'manifest.json').read_text()))
        self.assertEqual(self.run_cli(extra=['--input-format', 'normalized']), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(json.loads((self.output/'search_diagnostics'/filename).read_text()), detail)

    def test_diagnostic_capture_fails_loudly_with_old_memory_service(self):
        self.behavior = 'ignore_diagnostics'
        stale = self.output / 'search_diagnostics' / (hashlib.sha256(b'r1:q1').hexdigest() + '.json')
        stale.parent.mkdir(parents=True)
        stale.write_text('{"stale": true}')
        runner.a.write_rows(self.input, [dict(id='r1', messages=[],
            questions=[dict(id='q1', question='Where?', answer='Paris')])])
        with self.assertRaisesRegex(ValueError, 'did not write search diagnostics'):
            self.run_cli(extra=['--input-format', 'normalized', '--save-retrieval-diagnostics'])

    def test_search_capture_does_not_invent_results_in_oracle_or_empty_mode(self):
        runner.a.write_rows(self.input, [dict(id='r1', messages=[],
            questions=[dict(id='q1',question='Where?',answer='Paris', evidence_texts=['Paris'])])])
        for mode in ('oracle', 'empty'):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                self.assertEqual(self.run_cli(extra=['--input-format', 'normalized', '--mode', mode,
                                                    '--save-search-results']), 0)
                self.assertFalse((self.output/'search').exists())

    def test_recover_orphan_and_torn_ledger(self):
        path = self.root/'ledger.jsonl'
        args = runner.parser().parse_args(['--input', str(self.input), '--output', str(self.output)])
        with contextlib.redirect_stderr(self.events):
            accounting = runner.Accounting(path, args)
            accounting.start('answer', 'q1', 1, {'messages':[]}, runner.TokenCounter())
            with path.open('ab') as stream:
                stream.write(b'{"broken')
            recovered = runner.Accounting(path, args)
            self.assertEqual(recovered.totals['unknown_usage'], 1)
            again = runner.Accounting(path, args)
            self.assertEqual(again.totals['requests'], 1)


if __name__ == '__main__':
    unittest.main()
