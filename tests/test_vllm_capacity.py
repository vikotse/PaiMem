import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location('capacity', Path(__file__).resolve().parents[1] / 'scripts/benchmark_vllm_capacity.py')
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)


def settings(**kwargs):
    return SimpleNamespace(candidates=5, model='test', max_length=1024, timeout=1,
                           **kwargs)


def test_concurrency_remainder_auth_and_drain():
    active = peak = calls = 0
    sizes = []

    async def handler(request):
        nonlocal active, peak, calls
        body = json.loads(request.content)
        assert body['truncate_prompt_tokens'] == 1024
        assert body['queries'] == ['q'] * len(body['documents'])
        assert request.headers['Authorization'] == 'Bearer secret'
        active += 1
        peak = max(peak, active)
        calls += 1
        sizes.append(len(body['documents']))
        await asyncio.sleep(.03)
        active -= 1
        return httpx.Response(200, json={'data': [dict(index=i, score=.5) for i in reversed(range(len(body['documents'])))]})

    async def run():
        async with httpx.AsyncClient(base_url='http://localhost', transport=httpx.MockTransport(handler),
                                     headers={'Authorization': 'Bearer secret'}) as client:
            return await capacity.measure(client, settings(), [dict(query='q', documents=['d'] * 5)], 2, 3, 2, .01)

    row = asyncio.run(run())
    assert row['successful_queries'] == 3
    assert peak == 6 and active == 0
    assert calls == 9 and sizes.count(1) == 3
    assert row['wall_seconds'] >= .06  # Includes all work admitted before the deadline.
    assert row['useful_pairs_per_second'] == pytest.approx(15 / row['wall_seconds'])


def test_partial_failure_does_not_count_as_successful_query():
    async def handler(request):
        docs = json.loads(request.content)['documents']
        if len(docs) == 1:
            return httpx.Response(503, text='private document data')
        return httpx.Response(200, json={'data': [dict(index=i, score=.5) for i in range(len(docs))]})

    async def run():
        async with httpx.AsyncClient(base_url='http://localhost', transport=httpx.MockTransport(handler)) as client:
            return await capacity.measure(client, settings(), [dict(query='q', documents=['d'] * 5)], 2, 1, 1, .01)

    row = asyncio.run(run())
    assert row['successful_queries'] == 0
    assert row['useful_pairs_per_second'] == 0
    assert row['valid_pairs_per_second'] > 0
    assert row['errors'] == {'HTTP_503': 1}
    assert row['query_p95_ms'] is None
    assert 'private' not in json.dumps(row)


def test_timeout_is_counted_and_all_workers_finish():
    async def handler(request):
        await asyncio.sleep(1)
    async def run():
        async with httpx.AsyncClient(base_url='http://localhost', transport=httpx.MockTransport(handler)) as client:
            args = settings()
            args.timeout = .01
            return await capacity.measure(client, args, [dict(query='q', documents=['d'] * 5)], 5, 2, 1, .005)
    row = asyncio.run(run())
    assert row['failed_queries'] == 2
    assert row['errors'] == {'TimeoutError': 2}


@pytest.mark.parametrize('data', [
    [dict(index=0, score=.5), dict(index=0, score=.6)],
    [dict(index=0, score=.5), dict(index=1, score=float('nan'))],
    [dict(index=0, score=.5), dict(index=True, score=.6)],
])
def test_invalid_scores(data):
    with pytest.raises(ValueError):
        capacity.validate_scores({'data': data}, 2)


def test_recommendations_require_all_repeats_slo_samples_and_current_client():
    args = settings(repeats=2, max_error_rate=0, min_success=100, p95_ms=1000)
    def trial(batch_parallel, throughput, **kw):
        return dict(server='dp4', batch_size=64, concurrency=8, batch_concurrency=batch_parallel,
                    error_rate=0, successful_queries=100, wall_seconds=500 / throughput,
                    query_p95_ms=500, **kw)
    rows = [trial(1, 100), trial(1, 120), trial(4, 200), trial(4, 210)]
    result = capacity.recommendations(rows, args)
    assert result['best']['batch_concurrency'] == 4
    assert result['best_current_client']['batch_concurrency'] == 1
    rows[-1]['error_rate'] = .1
    assert capacity.recommendations(rows, args)['best']['batch_concurrency'] == 1
    rows[0]['query_p95_ms'] = 1001
    assert capacity.recommendations(rows, args)['best'] is None
    assert capacity.recommendations([trial(1, 100)], args)['best'] is None


def test_dry_run_does_not_contact_server_or_write(tmp_path):
    output = tmp_path / 'report'
    assert capacity.main(['--synthetic', '--candidates', '5', '--output', str(output), '--dry-run']) == 0
    assert not list(tmp_path.iterdir())


def test_owned_server_cleanup_signals_only_its_group(monkeypatch):
    calls = []
    monkeypatch.setattr(capacity.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    process = SimpleNamespace(pid=12345, wait=lambda timeout: 0)
    capacity.stop_owned(process)
    assert calls == [(12345, capacity.signal.SIGTERM), (12345, capacity.signal.SIGKILL)]


def test_complete_cli_report_and_output_protection(tmp_path, monkeypatch):
    real_client = httpx.AsyncClient
    async def handler(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'test'}]})
        body = json.loads(request.content)
        await asyncio.sleep(.001)
        return httpx.Response(200, json={'data': [dict(index=i, score=.7) for i in range(len(body['documents']))]})
    monkeypatch.setattr(capacity.httpx, 'AsyncClient', lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv('MEMORY_RERANKER_API_KEY', 'secret-not-for-report')
    prefix = tmp_path / 'results'
    argv = ['--synthetic', '--model', 'test', '--candidates', '5', '--batch-sizes', '2',
            '--concurrency', '2', '--batch-concurrency', '1', '2', '--duration', '.02',
            '--warmup', '.005', '--cooldown', '.001', '--repeats', '2', '--min-success', '1',
            '--output', str(prefix)]
    assert capacity.main(argv) == 0
    raw = prefix.with_suffix('.json').read_text()
    report = json.loads(raw)
    assert len(report['results']) == 4
    assert report['recommendations']['best_current_client']['batch_concurrency'] == 1
    assert 'secret-not-for-report' not in raw
    assert prefix.with_suffix('.csv').exists()
    with pytest.raises(SystemExit):
        capacity.main(argv)
    assert prefix.with_suffix('.json').read_text() == raw
