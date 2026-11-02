"""Search budgets are capped before retrieval and at the response boundary."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from paimem import api
from paimem.configuration import prepare


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def search_client(tmp_path):
    store = Mock()
    with patch.dict(os.environ, {
        'MEMORY_API_KEY': 'test', 'SKIP_MODEL_STARTUP': 'true',
        'MEMORY_DIAGNOSTICS_DIR': str(tmp_path),
    }, clear=True), patch.dict(api.app.dependency_overrides, {api.get_store: lambda: store}):
        with TestClient(api.app, client=('127.0.0.1', 12345)) as client:
            yield client, store


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('extra_summary', [False, True])
@pytest.mark.parametrize('cap,requested,expected', [
    (None, 1000, 100), (None, 100, 100), (None, 3, 3),
    (7, 101, 7), (7, 3, 3), (150, 200, 150), (1, 1000, 1),
])
def test_search_budget_and_response(search_client, tmp_path, monkeypatch,
                                    diagnostics, extra_summary, cap, requested, expected):
    client, store = search_client
    if cap is not None:
        monkeypatch.setenv('MEMORY_SEARCH_MAX_TOP_K', str(cap))

    def search(user, query, top_k, options, diagnostics=False):
        hits = [SimpleNamespace(id=str(i), content='memory', score=1., created_at='',
                                recall_queue='semantic')
                for i in range(top_k + int(extra_summary))]
        if not diagnostics:
            return hits
        return hits, {
            'top_k': top_k, 'returned_count': len(hits),
            'final_results': [{**hit.__dict__, 'rank': rank} for rank, hit in enumerate(hits, 1)],
            'candidates': [{'id': hit.id, 'returned_rank': rank, 'returned_ranks': [rank]}
                           for rank, hit in enumerate(hits, 1)],
        }

    store.search.side_effect = search
    payload = dict(query='q', user_id='u', top_k=requested, options=['option'])
    path = tmp_path / 'trace.json'
    if diagnostics:
        payload['debug_output_path'] = str(path)
    response = client.post('/search', json=payload, headers={'X-Api-Key': 'test'})
    assert response.status_code == 200
    data = response.json()['data']
    assert len(data) == expected
    assert [hit['id'] for hit in data] == [str(i) for i in range(expected)]
    store.search.assert_called_once_with('u', 'q', expected, ['option'],
                                         **({'diagnostics': True} if diagnostics else {}))
    if diagnostics:
        detail = json.loads(path.read_text())
        assert detail['top_k'] == detail['returned_count'] == expected
        assert [hit['id'] for hit in detail['final_results']] == [hit['id'] for hit in data]
        if extra_summary:
            assert detail['candidates'][-1]['returned_rank'] is None
            assert detail['candidates'][-1]['returned_ranks'] == []


@pytest.mark.parametrize('requested', [0, -1, 1.5, 'invalid'])
def test_invalid_request_still_rejected(search_client, requested):
    client, store = search_client
    response = client.post('/search', json=dict(query='q', user_id='u', top_k=requested),
                           headers={'X-Api-Key': 'test'})
    assert response.status_code == 422
    store.search.assert_not_called()


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'invalid', ''])
def test_invalid_limit_fails_at_startup(monkeypatch, value):
    monkeypatch.setenv('MEMORY_SEARCH_MAX_TOP_K', value)
    monkeypatch.setenv('SKIP_MODEL_STARTUP', 'true')
    with pytest.raises(ValueError, match='MEMORY_SEARCH_MAX_TOP_K'):
        with TestClient(api.app):
            pass


def test_config_defaults_toml_override_and_validation(tmp_path):
    env = {'MEMORY_API_KEY': 'test', 'EMBEDDING_BACKEND': 'local',
           'MEMORY_RERANKER_API_KEY': '', 'VLLM_DEPLOYMENT': 'remote'}
    for profile in ('default', 'baseline'):
        _, result = prepare(ROOT / f'configs/{profile}.toml', env)
        assert result['MEMORY_SEARCH_MAX_TOP_K'] == '100'
    path = tmp_path / 'config.toml'
    path.write_text('')
    assert prepare(path, env)[1]['MEMORY_SEARCH_MAX_TOP_K'] == '100'
    path.write_text('[service]\nsearch_max_top_k = 30\n')
    assert prepare(path, env)[1]['MEMORY_SEARCH_MAX_TOP_K'] == '30'
    assert prepare(path, {**env, 'MEMORY_SEARCH_MAX_TOP_K': '150'})[1]['MEMORY_SEARCH_MAX_TOP_K'] == '150'
    for value in ('0', '-1', '1.5', 'invalid', ''):
        with pytest.raises(ValueError, match='MEMORY_SEARCH_MAX_TOP_K'):
            prepare(path, {**env, 'MEMORY_SEARCH_MAX_TOP_K': value})
    path.write_text('[service]\nsearch_max_top_k = 1.5\n')
    with pytest.raises(ValueError, match='MEMORY_SEARCH_MAX_TOP_K'):
        prepare(path, env)
