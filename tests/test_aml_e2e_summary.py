from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode
from http.server import ThreadingHTTPServer

from scripts.summarize_aml_e2e import (aggregate, case_detail, case_diagnostics_page, case_score, compare_cases,
                                     load_versions, make_handler, markdown, single_cases, display_search_results, conflict_display)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def report(self, name, count=2, score=.5, currency='CNY', **updates):
        row = dict(questions=count, answered=count, judged=count, score_mean=score,
                   complete=True, contract='binary_judge', wall_seconds_this_run=10,
                   usage=dict(request_seconds=12, input_tokens=100, output_tokens=20,
                              requests=4, unknown_usage=0, estimated_usage=0, estimated_cost=2),
                   cost_currency=currency, prices_configured={'answer': True, 'judge': True})
        row.update(updates)
        path = self.root / name / 'report.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row))

    def test_weighted_and_separate_currencies(self):
        self.report('a', count=2, score=0)
        self.report('b', count=6, score=1, currency='USD')
        report = aggregate(self.root)
        self.assertEqual(report['totals']['score_mean'], .75)
        self.assertEqual(report['totals']['questions'], 8)
        self.assertEqual(report['totals']['wall_seconds_this_run'], 20)
        self.assertEqual(report['totals']['costs']['CNY']['estimated_cost'], 2)
        self.assertEqual(report['totals']['costs']['USD']['estimated_cost'], 2)
        self.assertIn('75.00%', markdown(report))

    def test_missing_values_not_zero(self):
        self.report('a', judged=0, score_mean=None, usage={}, prices_configured={})
        report = aggregate(self.root)
        self.assertIsNone(report['totals']['score_mean'])
        self.assertIsNone(report['totals']['input_tokens'])
        self.assertIsNone(report['totals']['costs']['CNY']['estimated_cost'])
        self.assertFalse(report['totals']['costs']['CNY']['complete'])

    def test_unknown_usage_excluded_from_successful_request_cost(self):
        self.report('a', usage=dict(request_seconds=12, input_tokens=100, output_tokens=20,
                                    requests=6, unknown_usage=2, estimated_usage=0, estimated_cost=2))
        report = aggregate(self.root)
        self.assertEqual(report['datasets'][0]['unknown_usage'], 2)
        self.assertEqual(report['datasets'][0]['estimated_cost'], 2)
        self.assertTrue(report['datasets'][0]['cost_complete'])
        self.assertTrue(report['totals']['costs']['CNY']['complete'])
        self.assertTrue(report['totals']['complete'])
        self.assertFalse(any('cost unavailable or incomplete' in w for w in report['warnings']))
        self.assertNotIn('（不完整）', markdown(report))

    def test_partial_cost_and_corrupt_report(self):
        self.report('a')
        self.report('b', prices_configured={'answer': False})
        (self.root / 'bad').mkdir()
        (self.root / 'bad/report.json').write_text('{')
        report = aggregate(self.root)
        self.assertEqual(report['totals']['costs']['CNY']['estimated_cost'], 2)
        self.assertFalse(report['totals']['costs']['CNY']['complete'])
        self.assertTrue(any('bad:' in w for w in report['warnings']))

    def test_live_scan_and_http(self):
        self.report('run/a')
        run = self.root / 'run'
        (run / 'summary.json').write_text(json.dumps(aggregate(run)))
        self.assertEqual(len(load_versions(self.root)['reports']), 1)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.root))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}'
            with urlopen(url) as response:
                page = response.read().decode('utf-8')
                self.assertIn('PaiMem 评测结果对比', page)
                self.assertIn('PaiMem Evaluation Comparison', page)
                self.assertIn('id="language"', page)
            self.report('next/a')
            (self.root / 'next/summary.json').write_text(json.dumps(aggregate(self.root / 'next')))
            with urlopen(url + '/api/reports') as response:
                self.assertEqual(len(json.load(response)['reports']), 2)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_empty_directory(self):
        with self.assertRaises(ValueError):
            aggregate(self.root)

    def case_run(self, version, inputs, judges, answers=None, contract='binary_judge', **metadata):
        self.report(version + '/a', contract=contract)
        folder = self.root / version / 'a'
        for filename, rows in (('input.jsonl', inputs), ('judged.jsonl', judges), ('answers.jsonl', answers or [])):
            folder.joinpath(filename).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        folder.joinpath('manifest.json').write_text(json.dumps(dict(input_sha256='same', pipeline_sha256='same', **metadata)))
        return folder

    def checkpoint(self, folder, stage, row):
        ident = row.get('id', row.get('idx'))
        path = folder / stage / (hashlib.sha256(ident.encode()).hexdigest() + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(row))
        return path

    @contextmanager
    def server_url(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.root))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}'
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_category_mapping_aggregation_and_filters(self):
        from unittest.mock import patch
        from scripts.summarize_aml_e2e import category_summary, category_mapping
        splits = self.root / 'splits'
        splits.mkdir()
        mapping = splits / 'question_categories.eval_240.json'
        entries = [dict(source_dataset='a', item_id=q, type=t, subtype='子类')
                   for q, t in [('q1', '时间型'), ('q2', '时间型'), ('q3', '事实型')]]
        mapping.write_text(json.dumps(dict(questions=entries)))
        items = [dict(id=q, question=q) for q in ('q1', 'q2', 'unknown')]
        self.case_run('base_eval_240', items, [dict(id='q1', is_correct=True)])
        self.case_run('next_eval_240', items, [dict(id='q1', is_correct=False)])
        with patch('scripts.summarize_aml_e2e.SPLITS_DIR', splits):
            self.assertIsNone(category_mapping('base_validation')[0])
            report = dict(version='base_eval_240', datasets=[dict(dataset='a')])
            rows, warnings = category_summary(self.root, report)
            rows = {r['category']: r for r in rows}
            self.assertEqual(rows['时间型']['questions'], 2)
            self.assertEqual(rows['时间型']['scored_questions'], 1)
            self.assertEqual(rows['时间型']['score_mean'], 1)
            self.assertIsNone(rows['事实型']['score_mean'])
            self.assertEqual(rows['未分类']['questions'], 1)
            result = single_cases(self.root, 'base_eval_240', 'a', category='时间型', page_size=1, page=2)
            self.assertEqual(result['filtered_count'], 2)
            self.assertEqual(result['cases'][0]['id'], 'q2')
            self.assertEqual(result['cases'][0]['subtype'], '子类')
            result = compare_cases(self.root, 'base_eval_240', 'next_eval_240', 'a', category='时间型')
            self.assertEqual([r['id'] for r in result['cases']], ['q1'])
            self.assertEqual(result['counts']['regressed'], 1)
            with self.server_url() as url:
                params = urlencode(dict(version='base_eval_240', dataset='a', category='未分类'))
                result = json.load(urlopen(url + '/api/cases?' + params))
                self.assertEqual([r['id'] for r in result['cases']], ['unknown'])
            self.case_run('rubric_eval_240', items, [dict(id='q1', llm_judge_score=.4)], contract='beam_rubric')
            rows, _ = category_summary(self.root, dict(report, version='rubric_eval_240'))
            self.assertEqual(next(r for r in rows if r['category']=='时间型')['score_mean'], .4)
            entries.append(entries[0])
            mapping.write_text(json.dumps(dict(questions=entries)))
            self.assertIn('duplicate', category_mapping('base_eval_240')[1][0])
            mapping.write_text('{')
            self.assertIsNone(category_mapping('base_eval_240')[0])
            self.assertTrue(single_cases(self.root, 'base_eval_240', 'a')['warnings'])

    def test_all_datasets_categories_pagination_and_detail_identity(self):
        from unittest.mock import patch
        import shutil
        splits = self.root / 'splits'
        splits.mkdir()
        entries = [dict(source_dataset=ds, item_id=q, type=kind)
                   for ds, q, kind in [('a', 'same-id', '时间型'), ('b', 'same-id', '时间型'),
                                       ('a', 'other', '事实型'), ('b', 'other', '事实型')]]
        (splits / 'question_categories.eval240.json').write_text(json.dumps(dict(questions=entries)))
        items = [dict(id='same-id', question='When?', answer='Monday'),
                 dict(id='other', question='Who?', answer='Sam')]
        for version, score in [('base_eval240', True), ('next_eval240', False)]:
            folder = self.case_run(version, items, [dict(id='same-id', is_correct=score)])
            shutil.copytree(folder, folder.with_name('b'))
        # The same ID in b has a different answer, and must retain its own detail.
        b = self.root / 'base_eval240' / 'b'
        (b / 'answers.jsonl').write_text(json.dumps(dict(id='same-id', generated_answer='Tuesday'))+'\n')
        with patch('scripts.summarize_aml_e2e.SPLITS_DIR', splits):
            first = single_cases(self.root, 'base_eval240', '*', category='时间型', page_size=1)
            second = single_cases(self.root, 'base_eval240', '*', category='时间型', page_size=1, page=2)
            self.assertEqual(first['total'], 4)
            self.assertEqual(first['filtered_count'], 2)
            self.assertEqual(first['cases'][0]['dataset'], 'a')
            self.assertEqual(second['cases'][0]['dataset'], 'b')
            self.assertEqual(first['cases'][0]['id'], second['cases'][0]['id'])
            self.assertEqual(set(first['categories']), {'时间型', '事实型'})
            comparison = compare_cases(self.root, 'base_eval240', 'next_eval240', '*', category='时间型',
                                       filter='regressed', page_size=1, page=2)
            self.assertEqual(comparison['counts']['regressed'], 2)
            self.assertEqual(comparison['filtered_count'], 2)
            self.assertEqual(comparison['cases'][0]['dataset'], 'b')
            self.assertEqual(single_cases(self.root, 'base_eval240', '*', category='时间型', query='Who')['filtered_count'], 0)
            with self.server_url() as url:
                params = urlencode(dict(version='base_eval240', dataset='*', category='时间型'))
                listing = json.load(urlopen(url + '/api/cases?' + params))
                self.assertEqual([r['dataset'] for r in listing['cases']], ['a', 'b'])
                row = listing['cases'][1]
                params = urlencode(dict(version='base_eval240', dataset=row['dataset'], id=row['id']))
                detail = json.load(urlopen(url + '/api/case?' + params))
                self.assertEqual(detail['versions'][0]['answer']['generated_answer'], 'Tuesday')
            # A dataset absent from one version is still listed as missing.
            shutil.rmtree(self.root / 'next_eval240' / 'b')
            result = compare_cases(self.root, 'base_eval240', 'next_eval240', '*', category='时间型', filter='missing')
            self.assertEqual([(r['dataset'], r['id']) for r in result['cases']], [('b', 'same-id')])
        self.assertEqual(single_cases(self.root, 'base_eval240', '*')['total'], 4)
        self.assertEqual(single_cases(self.root, 'absent', '*')['cases'], [])

    def test_score_filters_boundaries_and_http(self):
        values = dict(full=1, zero=0, partial=.5, near_full=.999999, near_zero=.000001,
                      invalid=2, nonfinite=float('nan'), missing=None)
        items = [dict(id=q, question=q) for q in values]
        judges = [dict(id=q, llm_judge_score=value) for q, value in values.items() if value is not None]
        self.case_run('scores', items, judges, contract='beam_rubric')
        expected = dict(all=set(values), full={'full'}, zero={'zero'},
                        partial={'partial', 'near_full', 'near_zero'},
                        not_full={'zero', 'partial', 'near_full', 'near_zero'},
                        unscored={'invalid', 'nonfinite', 'missing'})
        for kind, ids in expected.items():
            with self.subTest(kind=kind):
                result = single_cases(self.root, 'scores', '*', score_filter=kind)
                self.assertEqual({r['id'] for r in result['cases']}, ids)
                self.assertEqual(result['filtered_count'], len(ids))
        result = single_cases(self.root, 'scores', '*', score_filter='partial', query='near', page_size=1, page=2)
        self.assertEqual(result['filtered_count'], 2)
        self.assertEqual(result['cases'][0]['id'], 'near_zero')
        with self.assertRaises(ValueError):
            single_cases(self.root, 'absent', '*', score_filter='bad')
        with self.server_url() as url:
            params = dict(version='scores', dataset='*', score_filter='zero')
            result = json.load(urlopen(url + '/api/cases?' + urlencode(params)))
            self.assertEqual([r['id'] for r in result['cases']], ['zero'])
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + '/api/cases?' + urlencode(dict(params, score_filter='bad')))
            self.assertEqual(caught.exception.code, 400)

    def test_score_filters_combine_versions_categories_and_datasets(self):
        import shutil
        from unittest.mock import patch
        splits = self.root / 'splits'
        splits.mkdir()
        entries = [dict(source_dataset=ds, item_id=q, type=kind)
                   for ds in ('a', 'b') for q, kind in [('q1', '时间型'), ('q2', '时间型'), ('q3', '事实型')]]
        (splits / 'question_categories.eval240.json').write_text(json.dumps(dict(questions=entries)))
        items = [dict(id=q, question=q) for q in ('q1', 'q2', 'q3')]
        for version, scores in [('base_eval240', (True, False, True)), ('next_eval240', (False, True, False))]:
            folder = self.case_run(version, items, [dict(id=q['id'], is_correct=v) for q, v in zip(items, scores)])
            shutil.copytree(folder, folder.with_name('b'))
        with patch('scripts.summarize_aml_e2e.SPLITS_DIR', splits):
            args = dict(root=self.root, baseline='base_eval240', comparison='next_eval240', dataset='*',
                        category='时间型', filter='all', score_filter='zero')
            result = compare_cases(**args)
            self.assertEqual([(r['dataset'], r['id']) for r in result['cases']], [('a', 'q1'), ('b', 'q1')])
            self.assertEqual(result['counts']['regressed'], 2)
            result = compare_cases(**args, score_version='baseline', page_size=1, page=2)
            self.assertEqual(result['filtered_count'], 2)
            self.assertEqual((result['cases'][0]['dataset'], result['cases'][0]['id']), ('b', 'q2'))
            self.assertEqual(compare_cases(**dict(args, filter='improved'))['filtered_count'], 0)
            self.assertEqual(compare_cases(**dict(args, dataset='a'))['filtered_count'], 1)
            with self.server_url() as url:
                params = dict(baseline='base_eval240', comparison='next_eval240', dataset='*',
                              category='时间型', filter='all', score_filter='zero', score_version='baseline')
                result = json.load(urlopen(url + '/api/cases?' + urlencode(params)))
                self.assertEqual([r['id'] for r in result['cases']], ['q2', 'q2'])
                with self.assertRaises(HTTPError) as caught:
                    urlopen(url + '/api/cases?' + urlencode(dict(params, score_version='bad')))
                self.assertEqual(caught.exception.code, 400)

    def test_case_score_contracts_and_invalid_values(self):
        fixtures = [('binary_judge', {'is_correct': False}, 0),
                    ('personamem_v1', {'is_correct': True}, 1),
                    ('personamem_v2', {'is_correct': False}, 0),
                    ('beam_rubric', {'llm_judge_score': .5}, .5),
                    ('clbench_rubric', {'rubric_clbench_score': 0, 'rubric_clbench_requirement_ratio': .8}, 0),
                    ('scriptmem_rule', {'score': '0.5'}, .5)]
        for contract, row, expected in fixtures:
            with self.subTest(contract=contract):
                self.assertEqual(case_score(row, contract), expected)
        for value in (-1, 2, float('nan'), float('inf'), True, 'invalid'):
            self.assertIsNone(case_score({'llm_judge_score': value}, 'beam_rubric'))
        self.assertIsNone(case_score({'is_correct': 'false'}, 'binary_judge'))
        self.assertIsNone(case_score({}, 'binary_judge'))
        self.assertIsNone(case_score({'score': 1}, 'unknown'))
        self.assertIsNone(case_score({'score': 1}, []))

    def test_case_alignment_filters_missing_and_pagination(self):
        inputs = [dict(id=ident, question='问题 ' + ident, gold_answer='Paris') for ident in ('q1', 'q2', 'q3')]
        self.case_run('base', inputs, [dict(id='q2', is_correct=True), dict(id='q1', is_correct=False)])
        self.case_run('next', inputs[::-1] + [dict(id='q4', question='new')],
                      [dict(id='q1', is_correct=True), dict(id='q2', is_correct=False), dict(id='q3', is_correct=False)])
        data = compare_cases(self.root, 'base', 'next', 'a')
        self.assertEqual(data['counts'], dict(improved=1, regressed=1, same=0, missing=2, incomparable=0))
        self.assertEqual([r['id'] for r in data['cases']], ['q1', 'q2'])
        self.assertEqual([r['delta'] for r in data['cases']], [1, -1])
        data = compare_cases(self.root, 'base', 'next', 'a', filter='missing')
        self.assertEqual([r['id'] for r in data['cases']], ['q3', 'q4'])
        self.assertIsNone(data['cases'][0]['baseline_score'])
        self.assertEqual(data['cases'][0]['comparison_score'], 0)
        self.assertIsNone(data['cases'][0]['delta'])
        self.assertTrue(any('ID 集合不同' in note for note in data['warnings']))
        data = compare_cases(self.root, 'base', 'next', 'a', filter='all', query='问题', page_size=1, page=99)
        self.assertEqual(data['filtered_count'], 3)
        self.assertEqual(data['page'], 3)
        self.assertEqual(data['cases'][0]['id'], 'q3')
        self.assertEqual(compare_cases(self.root, 'base', 'next', 'a', filter='regressed')['cases'][0]['id'], 'q2')

    def test_case_content_and_contract_mismatch(self):
        inputs = [dict(id='q1', question='Where?', gold_answer='Paris')]
        self.case_run('base', inputs, [dict(id='q1', is_correct=True)])
        folder = self.case_run('next', [dict(inputs[0], gold_answer='Rome')], [dict(id='q1', is_correct=False)])
        data = compare_cases(self.root, 'base', 'next', 'a', filter='all')
        self.assertEqual(data['cases'][0]['status'], 'incomparable')
        self.assertIsNone(data['cases'][0]['delta'])
        self.case_run('rubric', inputs, [dict(id='q1', llm_judge_score=.5)], contract='beam_rubric')
        data = compare_cases(self.root, 'base', 'rubric', 'a', filter='all')
        self.assertEqual(data['cases'][0]['status'], 'incomparable')
        self.assertTrue(any('评测口径不同' in note for note in data['warnings']))
        (folder / 'manifest.json').write_text(json.dumps(dict(input_sha256='other', pipeline_sha256='other')))
        data = compare_cases(self.root, 'base', 'next', 'a')
        self.assertTrue(any('源数据指纹不同' in note for note in data['warnings']))
        self.assertTrue(any('pipeline指纹不同' in note for note in data['warnings']))

    def test_original_messages_are_linked_per_version_and_preserved(self):
        messages = [dict(role='user', content='<script>alert(1)</script>\u2028原文', timestamp='2026-01-01'),
                    dict(role='assistant', content='reply')]
        for version, content in [('base', messages), ('next', [])]:
            folder = self.case_run(version, [dict(id='q1', x_record_id='record-1')], [])
            source = self.root / (version + '.jsonl')
            source.write_text(json.dumps(dict(id='unrelated', messages=['wrong'])) + '\n' +
                              json.dumps(dict(id='record-1', messages=content), ensure_ascii=False) + '\n')
            (folder / 'manifest.json').write_text(json.dumps(dict(input=str(source), input_format='normalized',
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
        detail = case_detail(self.root, 'base', 'next', 'a', 'q1')
        self.assertEqual(detail['versions'][0]['original_messages']['messages'], messages)
        self.assertEqual(detail['versions'][1]['original_messages']['messages'], [])
        self.assertTrue(detail['versions'][0]['original_messages']['source'].endswith('base.jsonl:2'))
        source = self.root / 'base.jsonl'
        source.write_text('{}')
        result = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['original_messages']
        self.assertIsNone(result['messages'])
        self.assertIn('指纹不符', result['reason'])
        source.unlink()
        result = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['original_messages']
        self.assertIn('无法读取', result['reason'])

    def test_original_messages_missing_ambiguous_and_official(self):
        folder = self.case_run('base', [dict(id='q1', x_record_id='r1')], [])
        result = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['original_messages']
        self.assertIn('未记录', result['reason'])
        source = self.root / 'source.jsonl'
        for rows, reason in [([dict(id='other', messages=[])], '未找到'),
                             ([dict(id='r1'), dict(id='r1')], '重复'),
                             ([dict(id='r1')], '未保存 messages')]:
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            (folder / 'manifest.json').write_text(json.dumps(dict(input=str(source),
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            result = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['original_messages']
            self.assertIsNone(result['messages'])
            self.assertIn(reason, result['reason'])
        (folder / 'input.jsonl').write_text(json.dumps(dict(id='q1')) + '\n')
        source.write_text(json.dumps(dict(id='q1', messages=[dict(role='user', content='original')])) + '\n')
        (folder / 'manifest.json').write_text(json.dumps(dict(input=str(source), input_format='official',
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
        result = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['original_messages']
        self.assertEqual(result['messages'], [dict(role='user', content='original')])

    def test_checkpoints_live_and_detail_legacy_and_full_search(self):
        item = dict(id='q1', question='Where?', gold_answer='Paris', retrieved_context='Paris\nRome', x_n_retrieved=2)
        base = self.case_run('base', [item], [dict(id='q1', is_correct=False)])
        other = self.case_run('next', [], [])
        self.checkpoint(other, 'replay', dict(item, retrieved_context='Paris'))
        self.checkpoint(other, 'answer', dict(id='q1', generated_answer='Paris'))
        self.checkpoint(other, 'judge', dict(id='q1', is_correct=True, judge_response='correct'))
        search = other / 'search' / (hashlib.sha256(b'q1').hexdigest() + '.json')
        search.parent.mkdir()
        response = dict(data=[dict(id='m1', content='Paris', score=.9, extra=dict(source='memory'))], trace='t1')
        search.write_text(json.dumps(response))
        data = case_detail(self.root, 'base', 'next', 'a', 'q1')
        self.assertEqual(data['case']['delta'], 1)
        self.assertIsNone(data['versions'][0]['search_response'])
        self.assertEqual(data['versions'][0]['input']['retrieved_context'], 'Paris\nRome')
        self.assertEqual(data['versions'][1]['search_response'], response)
        self.assertTrue(data['versions'][1]['sources']['input'].startswith('replay/'))
        self.assertEqual(data['versions'][1]['answer']['generated_answer'], 'Paris')
        # A newer checkpoint must supersede an older stage merge on every request.
        self.checkpoint(base, 'judge', dict(id='q1', is_correct=True))
        self.assertEqual(compare_cases(self.root, 'base', 'next', 'a', filter='all')['cases'][0]['status'], 'same')
        search.write_text('{')
        data = case_detail(self.root, 'base', 'next', 'a', 'q1')
        self.assertIsNone(data['versions'][1]['search_response'])
        self.assertTrue(data['versions'][1]['warnings'])

    def test_final_results_join_original_ranks_without_collapsing_duplicate_ids(self):
        rows = [dict(id='same', content='[Positive-related evidence] yes', recall_queue='positive'),
                dict(id='same', content='[Negative-related evidence] no', recall_queue='negative'),
                dict(id='missing', content='not in semantic top k', recall_queue='positive'),
                dict(id='conflict:x', content='conflict', recall_queue='conflict_summary')]
        diagnostics = dict(final_results=rows, candidates=[
            dict(id='same', semantic_returned_rank=7, returned_rank=81,
                 channels=dict(reranker=dict(rank=12, score=.9), dense=dict(rank=21))),
            dict(id='missing', semantic_returned_rank=None, returned_rank=90, channels={})])
        result, source = display_search_results(None, diagnostics)
        self.assertEqual(source, 'diagnostics')
        self.assertEqual([r['final_rank'] for r in result], [1, 2, 3, 4])
        self.assertEqual([r['semantic_rank'] for r in result], [7, 7, None, None])
        self.assertEqual(result[0]['original_channels']['reranker']['rank'], 12)
        self.assertEqual(result[2]['semantic_rank_state'], 'not_in_top_k')
        self.assertEqual(result[3]['semantic_rank_state'], 'not_applicable')
        del diagnostics['candidates'][0]['semantic_returned_rank']
        result, _ = display_search_results(None, diagnostics)
        self.assertEqual(result[0]['semantic_rank_state'], 'unrecorded')
        self.assertIsNone(result[0]['semantic_rank'])
        self.assertEqual(result[0]['original_channels']['reranker']['rank'], 12)
        self.assertEqual(rows[0]['content'], '[Positive-related evidence] yes')

    def test_conflict_panel_distinguishes_stage_results_and_legacy_failures(self):
        cases = [('assessment_failed', 'judgment', 'failed'), ('conflict', 'judgment', 'conflict'),
                 ('no_conflict', 'judgment', 'no_conflict'), ('not_judgment', 'non_judgment', 'not_run'),
                 ('classification_failed', 'failed', 'not_run'), ('bypassed', 'not_run', 'not_run'),
                 ('retrieval_failed', 'judgment', 'not_run'), ('insufficient_evidence', 'judgment', 'not_run')]
        for status, classification, assessment in cases:
            with self.subTest(status=status):
                panel = conflict_display(dict(conflict_strategy=dict(status=status)), [])
                self.assertEqual(panel['classification'], classification)
                self.assertEqual(panel['assessment'], assessment)
        self.assertEqual(conflict_display(None, None)['assessment'], 'unknown')
        result = conflict_display(dict(conflict_strategy=dict(status='conflict', is_judgment=True)),
                                  [dict(recall_queue='conflict_summary', content='both claims')])
        self.assertEqual(result['explanation'], ['both claims'])

    def test_scope_queue_and_diagnostic_only_conflict_explanation(self):
        rows = [dict(id='s', content='[Scope/correction evidence] explicit correction')]
        diagnostics = dict(final_results=rows, conflict_strategy=dict(
            strategy='three_queue_v1', status='conflict', is_judgment=True, has_conflict=True,
            scope_query='scope and corrections', explanation='both claims; which is correct?',
            queues=dict(scope=dict(selected_count=1))))
        results, _ = display_search_results(None, diagnostics)
        self.assertEqual(results[0]['recall_queue'], 'scope')
        panel = conflict_display(diagnostics, results)
        self.assertEqual(panel['scope_query'], 'scope and corrections')
        self.assertEqual(panel['scope_count'], 1)
        self.assertEqual(panel['assessment'], 'conflict')
        self.assertEqual(panel['explanation'], ['both claims; which is correct?'])
        self.assertEqual(len(results), 1)

    def test_first_result_conflict_note_is_not_duplicated_in_panel(self):
        note = '[Conflicting evidence] Inferred correct description: implementation occurred.'
        rows = [dict(id='conflict:x', content=note, recall_queue='conflict_summary'),
                dict(id='p', content='[Positive-related evidence] implemented', recall_queue='positive')]
        diagnostics = dict(final_results=rows, conflict_strategy=dict(
            status='conflict', explanation=note, explanation_placement='first_result'),
            candidates=[dict(id='p', semantic_returned_rank=7, returned_rank=2, channels={})])
        results, _ = display_search_results(None, diagnostics)
        self.assertEqual(results[0]['final_rank'], 1)
        self.assertEqual(results[0]['semantic_rank_state'], 'not_applicable')
        self.assertEqual(results[1]['final_rank'], 2)
        self.assertEqual(results[1]['semantic_rank'], 7)
        panel = conflict_display(diagnostics, results)
        self.assertEqual(panel['explanation'], [note])
        self.assertEqual(panel['explanation_placement'], 'first_result')
        self.assertEqual(panel['positive_count'], 1)

    def test_display_queue_fields_legacy_inference_and_raw_response_preservation(self):
        folder = self.case_run('base', [dict(id='q1', question='Where?')], [])
        rows = [dict(id='s', content='ordinary', recall_queue='semantic'),
                dict(id='same', content='[Positive-related evidence] supported'),
                dict(id='same', content='[Negative-related evidence] denied'),
                dict(id='conflict:123', content='explanation'),
                dict(id='old', content='legacy'),
                dict(id='bad', content='legacy', recall_queue=['invalid'])]
        path = folder / 'search' / (hashlib.sha256(b'q1').hexdigest() + '.json')
        path.parent.mkdir()
        path.write_text(json.dumps(dict(data=rows)))
        version = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]
        self.assertEqual(version['search_response'], dict(data=rows))
        self.assertEqual(version['search_results_source'], 'search')
        self.assertEqual([r['recall_queue'] for r in version['search_results']],
                         ['semantic', 'positive', 'negative', 'conflict_summary', 'unknown', 'unknown'])
        self.assertEqual([r['id'] for r in version['search_results']], [r['id'] for r in rows])

    def test_diagnostics_only_top_k_and_nested_candidates_stay_separate(self):
        folder = self.case_run('base', [dict(id='q1', question='Where?')], [])
        rows = [dict(id='s', content='selected evidence', recall_queue='semantic'),
                dict(id='p', content='positive evidence', recall_queue='positive')]
        path = folder / 'search_diagnostics' / (hashlib.sha256(b'q1').hexdigest() + '.json')
        path.parent.mkdir()
        payload = dict(candidates=[], final_results=rows, conflict_strategy=dict(queues=dict(
            positive=dict(selected_ids=['p'], pipeline=dict(candidates=[dict(content='LARGE CANDIDATE')])))))
        path.write_text(json.dumps(payload))
        version = case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]
        self.assertIsNone(version['search_response'])
        self.assertEqual([{k: row[k] for k in ('id', 'content', 'recall_queue')}
                          for row in version['search_results']], rows)
        self.assertEqual(version['search_results_source'], 'diagnostics')
        self.assertNotIn('LARGE CANDIDATE', json.dumps(version))
        self.assertNotIn('final_results', version['search_diagnostics'])
        payload['final_results'] = []
        path.write_text(json.dumps(payload))
        self.assertEqual(case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['search_results'], [])
        del payload['final_results']
        path.write_text(json.dumps(payload))
        self.assertIsNone(case_detail(self.root, 'base', None, 'a', 'q1')['versions'][0]['search_results'])

    def test_full_diagnostics_are_paginated_before_http_response(self):
        item = dict(id='q1', question='Where?', gold_answer='Paris')
        self.case_run('base', [item], [dict(id='q1', is_correct=False)])
        other = self.case_run('next', [item], [dict(id='q1', is_correct=True)])
        filename = hashlib.sha256(b'q1').hexdigest() + '.json'
        path = other / 'search_diagnostics' / filename
        path.parent.mkdir()
        candidates = [dict(id=f'm{i}', content=f'Paris detail {i}',
                           channels=dict(dense=dict(rank=i+1, score=1/(i+1)),
                                         fusion=dict(rank=3-i, score=(3-i)/100)),
                           returned_rank=1 if i == 2 else None) for i in range(3)]
        path.write_text(json.dumps(dict(schema_version=1, stored_count=3, candidates=candidates)))
        detail = case_detail(self.root, 'base', 'next', 'a', 'q1')
        self.assertEqual(detail['versions'][1]['search_diagnostics']['candidate_count'], 3)
        self.assertNotIn('candidates', detail['versions'][1]['search_diagnostics'])
        page = case_diagnostics_page(self.root, 'base', 'next', 'a', 'q1', 'next', page_size=1)
        self.assertEqual(page['candidates'][0]['id'], 'm2')
        self.assertEqual(page['filtered_count'], 3)
        page = case_diagnostics_page(self.root, 'base', 'next', 'a', 'q1', 'next', channel='dense', page=2, page_size=1)
        self.assertEqual(page['candidates'][0]['id'], 'm1')
        page = case_diagnostics_page(self.root, 'base', 'next', 'a', 'q1', 'next', query='detail 2')
        self.assertEqual([r['id'] for r in page['candidates']], ['m2'])
        with self.assertRaises(ValueError):
            case_diagnostics_page(self.root, 'base', 'next', 'a', 'q1', 'other')

    def test_single_version_includes_incomplete_cases_and_one_detail_column(self):
        inputs = [dict(id='q1', question='Only input'), dict(id='q2', question='Answered')]
        self.case_run('only', inputs, [dict(id='q3', is_correct=False)],
                      answers=[dict(id='q2', generated_answer='Paris')])
        data = single_cases(self.root, 'only', 'a')
        self.assertEqual(data['total'], 3)
        self.assertEqual([row['score'] for row in data['cases']], [None, None, 0])
        self.assertEqual(data['cases'][0]['stages'], dict(input=True, answer=False, judge=False))
        page = single_cases(self.root, 'only', 'a', query='ANSWERED', page_size=1, page=99)
        self.assertEqual(page['page'], 1)
        self.assertEqual(page['cases'][0]['id'], 'q2')
        detail = case_detail(self.root, 'only', None, 'a', 'q2')
        self.assertEqual(detail['mode'], 'single')
        self.assertEqual(len(detail['versions']), 1)
        self.assertNotIn('delta', detail['case'])
        with self.server_url() as url:
            params = dict(version='only', dataset='a')
            with urlopen(url + '/api/cases?' + urlencode(params)) as response:
                self.assertEqual(json.load(response)['total'], 3)
            with urlopen(url + '/api/case?' + urlencode(dict(params, id='q3'))) as response:
                self.assertEqual(json.load(response)['case']['score'], 0)
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + '/api/cases?' + urlencode(dict(params, page=0)))
            self.assertEqual(caught.exception.code, 400)

    def test_retrieval_sort_dimensions_directions_and_missing_ranks(self):
        folder = self.case_run('only', [dict(id='q1')], [])
        path = folder / 'search_diagnostics' / (hashlib.sha256(b'q1').hexdigest() + '.json')
        path.parent.mkdir()
        candidates = [
            dict(id='a', content='match', returned_rank=None, channels=dict(fusion=dict(rank=1))),
            dict(id='b', content='match', returned_rank=2, channels=dict(fusion=dict(rank=2))),
            dict(id='c', content='other', returned_rank=1, channels=dict(fusion=dict(rank=3), dense=dict(rank=1))),
            dict(id='d', returned_rank=False, channels=dict(fusion=dict(rank=None))),
        ]
        path.write_text(json.dumps(dict(candidates=candidates)))
        def page(**kwargs):
            return case_diagnostics_page(self.root, 'only', None, 'a', 'q1', 'only', **kwargs)
        self.assertEqual([r['id'] for r in page()['candidates']], ['c', 'b', 'a', 'd'])
        self.assertEqual([r['id'] for r in page(order='desc')['candidates']], ['b', 'c', 'a', 'd'])
        self.assertEqual(page(page_size=1, page=2)['candidates'][0]['id'], 'b')
        self.assertEqual([r['id'] for r in page(channel='fusion')['candidates']], ['a', 'b', 'c', 'd'])
        self.assertEqual([r['id'] for r in page(channel='fusion', order='desc')['candidates']], ['c', 'b', 'a', 'd'])
        self.assertEqual([r['id'] for r in page(query='match')['candidates']], ['b', 'a'])
        self.assertEqual(page(channel='dense')['candidates'][0]['id'], 'c')
        detail = case_detail(self.root, 'only', None, 'a', 'q1')
        self.assertEqual(detail['versions'][0]['search_diagnostics']['channels'], ['dense', 'fusion'])
        for kwargs in (dict(order='bad'), dict(channel='bad')):
            with self.assertRaises(ValueError):
                page(**kwargs)
        with self.server_url() as url:
            params = dict(version='only', dataset='a', id='q1', order='desc', page_size=1)
            with urlopen(url + '/api/case/diagnostics?' + urlencode(params)) as response:
                self.assertEqual(json.load(response)['candidates'][0]['id'], 'b')

    def test_clbench_idx_and_scriptmem_qa_id(self):
        item = dict(id='q1', idx='q1', question='Where?', gold_answer='Paris')
        self.case_run('base', [item], [dict(idx='q1', rubric_clbench_score=0)],
                      [dict(idx='q1', model_output='Paris')], contract='clbench_rubric')
        self.case_run('next', [item], [dict(idx='q1', rubric_clbench_score=1)], contract='clbench_rubric')
        self.assertEqual(case_detail(self.root, 'base', 'next', 'a', 'q1')['versions'][0]['answer']['model_output'], 'Paris')
        self.case_run('rule', [item], [dict(qa_id='q1', score=.5)], contract='scriptmem_rule')
        self.assertEqual(case_detail(self.root, 'rule', 'rule', 'a', 'q1')['case']['baseline_score'], .5)

    def test_corrupt_and_duplicate_rows_do_not_create_false_differences(self):
        item = dict(id='q1', question='Where?')
        folder = self.case_run('base', [item], [dict(id='q1', is_correct=True)])
        self.case_run('next', [item], [dict(id='q1', is_correct=False)])
        with (folder / 'judged.jsonl').open('a') as stream:
            stream.write('{broken\n' + json.dumps(dict(id='q1', is_correct=False)) + '\n')
        data = compare_cases(self.root, 'base', 'next', 'a', filter='all')
        self.assertIsNone(data['cases'][0]['baseline_score'])
        self.assertEqual(data['cases'][0]['status'], 'missing')
        self.assertTrue(any('invalid row' in note for note in data['warnings']))
        self.assertTrue(any('duplicate' in note for note in data['warnings']))
        (folder / 'report.json').write_text(json.dumps(dict(contract=['invalid'])))
        data = compare_cases(self.root, 'base', 'next', 'a', filter='all')
        self.assertTrue(any('invalid contract' in note for note in data['warnings']))

    def test_case_http_validation_and_no_arbitrary_file_access(self):
        item = dict(id='q/中文', question='<script>alert(1)</script>', gold_answer='Paris')
        self.case_run('base', [item], [dict(id=item['id'], is_correct=False)], answer_api_key='do-not-expose')
        self.case_run('next', [item], [dict(id=item['id'], is_correct=True)])
        diagnostic_file = self.root / 'next' / 'a' / 'search_diagnostics' / (
            hashlib.sha256(item['id'].encode()).hexdigest() + '.json')
        diagnostic_file.parent.mkdir()
        diagnostic_file.write_text(json.dumps(dict(schema_version=1, candidates=[
            dict(id='m1', content='large-content', channels=dict(fusion=dict(score=.1, rank=1)))])))
        params = dict(baseline='base', comparison='next', dataset='a')
        with self.server_url() as url:
            with urlopen(url + '/api/cases?' + urlencode(params)) as response:
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertEqual(json.load(response)['cases'][0]['delta'], 1)
            with urlopen(url + '/api/case?' + urlencode(dict(params, id=item['id']))) as response:
                text = response.read().decode()
                self.assertNotIn('do-not-expose', text)
                self.assertEqual(json.loads(text)['case']['id'], item['id'])
                self.assertNotIn('large-content', text)
            with urlopen(url + '/api/case/diagnostics?' + urlencode(dict(params, id=item['id'], version='next'))) as response:
                self.assertEqual(json.load(response)['candidates'][0]['content'], 'large-content')
            invalid = [('/api/cases', dict(params, baseline='../base'), 400),
                       ('/api/cases', dict(params, dataset='/tmp'), 400),
                       ('/api/cases', dict(params, comparison='a\\b'), 400),
                       ('/api/cases', dict(params, page='0'), 400),
                       ('/api/cases', dict(params, filter='bad'), 400),
                       ('/api/case', dict(params, id='../../manifest.json'), 404),
                       ('/api/case', params, 400)]
            for route, query, status in invalid:
                with self.subTest(query=query), self.assertRaises(HTTPError) as caught:
                    urlopen(url + route + '?' + urlencode(query))
                self.assertEqual(caught.exception.code, status)
            with self.assertRaises(HTTPError) as caught:
                urlopen(url + '/base/a/input.jsonl')
            self.assertEqual(caught.exception.code, 404)

    def test_symlink_dataset_and_artifacts_rejected(self):
        item = dict(id='q1', question='Where?')
        base = self.case_run('base', [item], [dict(id='q1', is_correct=True)])
        other = self.case_run('next', [item], [dict(id='q1', is_correct=False)])
        (self.root / 'alias').symlink_to(self.root / 'base', target_is_directory=True)
        with self.assertRaises(ValueError):
            compare_cases(self.root, 'alias', 'next', 'a')
        (base / 'input.jsonl').unlink()
        (base / 'input.jsonl').symlink_to(other / 'input.jsonl')
        with self.assertRaises(ValueError):
            compare_cases(self.root, 'base', 'next', 'a')

if __name__ == '__main__':
    unittest.main()
