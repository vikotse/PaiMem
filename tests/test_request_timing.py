"""Request timing is independent of detailed stage timing and excludes private data."""
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from paimem import api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('MEMORY_API_KEY', 'PRIVATE_KEY')
    monkeypatch.setenv('SKIP_MODEL_STARTUP', 'true')
    monkeypatch.setenv('MEMORY_TIMING_ENABLED', '0')
    monkeypatch.delenv('MEMORY_REQUEST_TIMING_ENABLED', raising=False)
    store = Mock()
    store.search.return_value = []
    monkeypatch.setitem(api.app.dependency_overrides, api.get_store, lambda: store)
    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        yield test_client, store


def records(caplog):
    return [json.loads(r.getMessage().split(' ', 1)[1]) for r in caplog.records
            if r.name == 'paimem.request_timing']


def search(client, **kwargs):
    return client.post('/search', json={'user_id': 'PRIVATE_USER', 'query': 'PRIVATE_QUERY',
                                      'top_k': 1}, **kwargs)


def test_default_off(client, caplog):
    assert search(client[0], headers={'X-Api-Key': 'PRIVATE_KEY'}).status_code == 200
    assert records(caplog) == []


@pytest.mark.parametrize('switch', ['1', 'true'])
def test_success_and_privacy(client, monkeypatch, caplog, switch):
    monkeypatch.setenv('MEMORY_REQUEST_TIMING_ENABLED', switch)
    headers = {'X-Api-Key': 'PRIVATE_KEY'}
    assert search(client[0], headers=headers).status_code == 200
    assert client[0].post('/add', headers=headers, json={
        'request_id': 'PRIVATE_REQUEST', 'user_id': 'PRIVATE_USER', 'session_id': 'PRIVATE_SESSION',
        'messages': [{'role': 'user', 'content': 'PRIVATE_CONTENT'}],
    }).status_code == 200
    rows = records(caplog)
    assert [r['path'] for r in rows] == ['/search', '/add']
    assert all(r['status_code'] == 200 and r['elapsed_ms'] >= 0 for r in rows)
    assert len({r['trace_id'] for r in rows}) == 2
    assert 'PRIVATE' not in caplog.text
    client[0].get('/does-not-exist')
    assert len(records(caplog)) == 2


def test_failures_logged_once(client, monkeypatch, caplog):
    monkeypatch.setenv('MEMORY_REQUEST_TIMING_ENABLED', '1')
    assert search(client[0]).status_code == 401
    assert client[0].post('/add', json={}, headers={'X-Api-Key': 'PRIVATE_KEY'}).status_code == 422
    client[1].search.side_effect = RuntimeError('PRIVATE_ERROR')
    assert search(client[0], headers={'X-Api-Key': 'PRIVATE_KEY'}).status_code == 500
    assert [r['status_code'] for r in records(caplog)] == [401, 422, 500]
    assert 'PRIVATE_ERROR' not in caplog.text
