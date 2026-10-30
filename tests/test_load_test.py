import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('load_test', SCRIPTS / 'load_test.py')
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)


def args(**changes):
    result = load.mixed_parser().parse_args([])
    result.mode = 'mixed'
    result.users = 2
    result.seed_documents = 2
    result.seed_batch_size = 1
    result.seed_concurrency = 2
    result.messages_per_add = 2
    result.duration = .01
    result.timeout = 1
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def reply(request):
    body = json.loads(request.content)
    if request.url.path == '/add':
        return httpx.Response(200, json={**{k: body[k] for k in ('request_id', 'user_id', 'session_id')}, 'success': True})
    return httpx.Response(200, json={'data': [dict(id='m', content='answer', score=.5)]})


def test_add_search_each_16_overlap_seed_barrier_and_drain(monkeypatch):
    monkeypatch.setenv('MEMORY_API_KEY', 'SECRET')
    active = {'add': 0, 'search': 0}
    peak = active.copy()
    max_total = 0
    seeded = 0
    preflights = 0
    request_ids = []
    async def handler(request):
        nonlocal seeded, preflights, max_total
        body = json.loads(request.content)
        assert request.headers['X-Api-Key'] == 'SECRET'
        assert request.url.path in ('/add', '/search')
        if request.url.path == '/add':
            request_ids.append(body['request_id'])
            if ':seed:' in body['request_id']:
                await asyncio.sleep(.001)
                seeded += len(body['messages'])
                return reply(request)
        elif preflights < 2:
            preflights += 1
            return reply(request)
        assert seeded == 4 and preflights == 2
        kind = request.url.path[1:]
        active[kind] += 1
        peak[kind] = max(peak[kind], active[kind])
        max_total = max(max_total, sum(active.values()))
        await asyncio.sleep(.04)
        active[kind] -= 1
        return reply(request)
    report = asyncio.run(load.run_mixed(args(), load.load_workload(args()), httpx.MockTransport(handler)))
    assert peak == {'add': 16, 'search': 16} and max_total == 32
    assert active == {'add': 0, 'search': 0}
    assert report['wall_seconds'] >= .04
    assert report['add']['successful'] == report['search']['successful'] == 16
    assert report['add']['successful_qps'] == pytest.approx(16 / report['wall_seconds'])
    assert len(request_ids) == len(set(request_ids))
    assert report['seed']['confirmed_source_documents'] == 4
    assert 'SECRET' not in json.dumps(report)


def test_seed_failure_never_starts_measured_load():
    paths = []
    async def handler(request):
        body = json.loads(request.content)
        paths.append(request.url.path)
        assert ':seed:' in body['request_id']
        return httpx.Response(500, text='PRIVATE_ERROR')
    report = asyncio.run(load.run_mixed(args(), load.load_workload(args()), httpx.MockTransport(handler)))
    assert report['status'] == 'seed_failed'
    assert report['add'] == report['search'] == {}
    assert set(paths) == {'/add'}
    assert 'PRIVATE_ERROR' not in json.dumps(report)


def test_empty_seed_search_aborts():
    async def handler(request):
        if request.url.path == '/search':
            return httpx.Response(200, json={'data': []})
        return reply(request)
    report = asyncio.run(load.run_mixed(args(), load.load_workload(args()), httpx.MockTransport(handler)))
    assert report['status'] == 'seed_failed'
    assert 'EmptySeedSearch' in report['seed_errors']


def test_timeout_and_malformed_response_not_counted_as_success():
    async def handler(request):
        if json.loads(request.content)['user_id'] == 'slow':
            await asyncio.sleep(1)
        return httpx.Response(200, json={'data': [{'content': 'PRIVATE_TEXT'}]})
    async def run():
        async with httpx.AsyncClient(base_url='http://localhost', transport=httpx.MockTransport(handler)) as client:
            return [await load.call_api(client, '/search', dict(user_id=user, top_k=2), .01)
                    for user in ('slow', 'fast')]
    samples = asyncio.run(run())
    stats = load.summarize(samples, .1)
    assert stats['successful_qps'] == 0 and stats['mean_ms'] is None
    assert stats['errors'] == {'TimeoutError': 1, 'ValueError': 1}
    assert 'PRIVATE_TEXT' not in json.dumps(stats)


def test_mixed_failures_preserve_success_latency():
    stats = load.summarize([dict(seconds=1, error=None, result_count=1),
                            dict(seconds=9, error='HTTP_429', result_count=None)], 10)
    assert stats['mean_ms'] == 1000 and stats['p95_ms'] == 1000
    assert stats['error_rate'] == .5 and stats['failed_mean_ms'] == 9000
    assert stats['successful_qps'] == .1


@pytest.mark.parametrize('mode', ['add', 'search'])
def test_single_endpoint_modes(mode):
    async def handler(request):
        await asyncio.sleep(.02)
        return reply(request)
    settings = args(mode=mode)
    report = asyncio.run(load.run_mixed(settings, load.load_workload(settings), httpx.MockTransport(handler)))
    other = 'search' if mode == 'add' else 'add'
    assert report[mode]['successful'] == 16
    assert report[other]['attempted'] == 0


def test_dry_run_validates_input_without_network_or_files(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'input.jsonl'
    path.write_text(json.dumps({'query': 'PRIVATE_QUERY', 'documents': ['d1', 'd2']}) + '\n')
    def forbidden(*a, **kw):
        raise AssertionError('dry-run must not contact service')
    monkeypatch.setattr(load.httpx, 'AsyncClient', forbidden)
    output = tmp_path / 'report.json'
    load.main(['--mode', 'mixed', '--input', str(path), '--seed-documents', '2',
               '--output', str(output), '--dry-run'])
    assert json.loads(capsys.readouterr().out)['input_kind'] == 'jsonl'
    assert not output.exists()
    with pytest.raises(SystemExit):
        load.main(['--mode', 'mixed', '--input', str(path), '--seed-documents', '3', '--dry-run'])


def test_cli_report_and_existing_output_protection(tmp_path, monkeypatch, capsys):
    real_client = httpx.AsyncClient
    async def handler(request):
        await asyncio.sleep(.003)
        return reply(request)
    monkeypatch.setattr(load.httpx, 'AsyncClient', lambda **kw: real_client(
        **{**kw, 'transport': httpx.MockTransport(handler)}))
    output = tmp_path / 'report.json'
    command = ['--mode', 'mixed', '--users', '1', '--seed-documents', '2', '--duration', '.01',
               '--add-concurrency', '2', '--search-concurrency', '2', '--output', str(output)]
    load.main(command)
    report = json.loads(output.read_text())
    assert report['status'] == 'passed'
    assert report['add']['successful'] and report['search']['successful']
    previous = output.read_text()
    with pytest.raises(SystemExit):
        load.main(command)
    assert output.read_text() == previous


def test_legacy_mode_remains_available(monkeypatch):
    calls = []
    monkeypatch.setattr(load, 'legacy_main', lambda argv: calls.append(argv))
    load.main(['--concurrency', '16'])
    assert calls == [['--concurrency', '16']]


@pytest.mark.parametrize('failure', ['response', 'exception'])
def test_sequential_add_errors_exclude_private_details(monkeypatch, failure):
    def request(*args):
        if failure == 'exception':
            raise RuntimeError('PRIVATE_ERROR')
        return 500, {'detail': 'PRIVATE_RESPONSE'}
    monkeypatch.setattr(load, 'request_json', request)
    with pytest.raises(RuntimeError) as caught:
        load.main(['--users', '1'])
    assert 'PRIVATE' not in str(caught.value)


def test_sequential_search_errors_exclude_private_details(monkeypatch, capsys):
    def request(method, url, *args):
        if url.endswith('/add'):
            return 200, {'success': True}
        raise RuntimeError('PRIVATE_ERROR')
    monkeypatch.setattr(load, 'request_json', request)
    with pytest.raises(SystemExit):
        load.main(['--users', '1', '--searches-per-user', '1'])
    report = json.loads(capsys.readouterr().out)
    assert report['failures'] == ['RuntimeError']
    assert 'PRIVATE' not in json.dumps(report)


def test_cancellation_reports_unfinished_requests_and_drains_tasks():
    async def run():
        timed_started = asyncio.Event()
        active = 0
        async def handler(request):
            nonlocal active
            body = json.loads(request.content)
            if request.url.path == '/add' and ':timed:' in body['request_id']:
                active += 1
                timed_started.set()
                try:
                    await asyncio.sleep(100)
                finally:
                    active -= 1
            return reply(request)
        settings = args(mode='add', duration=100, add_concurrency=2)
        task = asyncio.create_task(load.run_mixed(settings, load.load_workload(settings), httpx.MockTransport(handler)))
        await asyncio.wait_for(timed_started.wait(), 2)
        task.cancel()
        report = await task
        assert active == 0
        return report
    report = asyncio.run(run())
    assert report['status'] == 'interrupted'
    assert report['cancelled_inflight']['add'] == 2
    assert report['add']['successful'] == 0
