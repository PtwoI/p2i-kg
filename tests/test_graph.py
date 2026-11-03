import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import seed_graph, GraphStore
from p2i_kg.app import create_app


def test_seed_relations_and_jsonld():
    graph = seed_graph()
    assert len(graph.nodes) == 384 and len(graph.edges) == 760
    assert len([node for node in graph.nodes if node.kind == 'model']) == 55
    assert all(edge.evidence for edge in graph.edges)
    data = graph.jsonld()
    assert len(data['@graph']) == len(graph.nodes)+len(graph.edges)
    assert all('evidence' in claim for claim in data['@graph'] if claim['@type'] == 'Claim')
    assert data['@context']['model']['@prefix']


@pytest.mark.parametrize('invalid', ['duplicate', 'dangling', 'range', 'evidence', 'paper'])
def test_invalid_catalog_rejected_atomically(tmp_path, invalid):
    store = GraphStore(tmp_path/'kg.sqlite3')
    before = store.load_optional()
    data = before[1].model_dump(mode='json')
    if invalid == 'duplicate':data['nodes'].append(data['nodes'][0])
    elif invalid == 'dangling':data['edges'][0]['target'] = 'missing'
    elif invalid == 'range':data['edges'][0]['target'] = 'model:mlp'
    elif invalid == 'evidence':data['edges'][0]['evidence'] = ['concept:activation']
    else:data['nodes'][0]['url'] = None
    with pytest.raises(ValueError):store.replace(data)
    assert store.load_optional() == before


def test_sqlite_reopen_and_idempotent_import(tmp_path):
    path = tmp_path/'kg.sqlite3'
    store = GraphStore(path)
    revision = store.replace(seed_graph())
    assert GraphStore(path).load_optional()[0] == revision
    graph = store.load()
    graph.title = 'My teaching catalog'
    changed = store.replace(graph)
    assert changed != revision
    assert GraphStore(path).load().title == 'My teaching catalog'


def test_search_compare_detail_and_exports(tmp_path):
    with TestClient(create_app(tmp_path/'kg.sqlite3')) as client:
        graph = client.get('/api/kg').json()
        assert len(graph['nodes']) == 384
        assert any(node['id'] == 'model:lstm' for node in client.get('/api/kg/search?q=LSTM').json()['nodes'])
        detail = client.get('/api/kg/nodes/model:transformer').json()
        assert detail['node']['kind'] == 'model' and detail['edges']
        result = client.get('/api/kg/compare?left=model:rnn&right=model:lstm').json()
        assert 'concept:recurrence' in {node['id'] for node in result['common']}
        assert 'concept:memory_cell' in {node['id'] for node in result['right_only']}
        assert client.get('/api/kg/compare?left=model:rnn&right=missing').status_code == 422
        assert client.get('/api/kg/nodes/missing').status_code == 404
        assert client.get('/api/kg/export?format=jsonld').json()['@graph']
        assert client.get('/api/kg/export?format=bad').status_code == 422
        assert client.post('/api/kg/runtime/model:mlp', json={}, headers={'Origin':'http://elsewhere.test'}).status_code == 403
        assert client.get('/').status_code == 200
