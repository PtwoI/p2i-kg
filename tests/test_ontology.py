import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from rdflib import Graph, Literal
from rdflib.namespace import RDF, OWL
from rdflib.compare import isomorphic
from fastapi.testclient import TestClient
from p2i_kg.app import create_app
from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph
from p2i_kg.ontology import BASE, identifier, data_graph, vocabulary, shapes, validate_rdf


def test_semantic_questions_and_explanations():
    graph = seed_graph()
    attended = {x['node']['id']: x['proof'] for x in graph.query(concept='concept:attention')}
    assert {'model:bert','model:gpt2','model:vit','model:gat'} <= attended.keys()
    assert 'model:mamba' not in attended
    assert len(attended['model:gpt2']) >= 2
    assert {x['node']['id'] for x in graph.query(concept='concept:position', objective='objective:autoregressive')} == {'model:gpt2','model:llama'}
    assert 'model:unet' in {x['node']['id'] for x in graph.query(task='task:segmentation')}
    assert graph.query(concept='concept:cross_attention', objective='objective:masked_lm') == []
    # extensions must not silently inherit architecture features.
    assert 'concept:self_attention' not in graph.effective_concepts('model:mamba')
    result = graph.compare('model:bert', 'model:gpt2')
    assert 'concept:causal_mask' in {n['id'] for n in result['right_only']}
    assert result['facets']['trained_with']['left'] == ['objective:masked_lm']


def test_rdf_shacl_and_lossless_jsonld_roundtrip():
    graph = seed_graph()
    assert validate_rdf(graph)['conforms']
    data = data_graph(graph)
    restored = Graph().parse(data=json.dumps(graph.jsonld()), format='json-ld')
    assert isomorphic(data, restored)
    assert (BASE.extends, RDF.type, OWL.TransitiveProperty) not in vocabulary()
    assert (BASE.subconcept_of, RDF.type, OWL.TransitiveProperty) in vocabulary()
    # Domain/range axioms alone are not validation: SHACL must reject malformed RDF.
    data.remove((identifier('paper:bert'), BASE.author, None))
    from pyshacl import validate
    assert not validate(data, shacl_graph=shapes(), inference='none')[0]


@pytest.mark.parametrize('relation,kind', [('subconcept_of','concept'),('has_component','component')])
def test_taxonomy_and_composition_cycles_fail(relation, kind):
    graph = seed_graph().model_dump(mode='json')
    keys = [n['id'] for n in graph['nodes'] if n['kind'] == kind][:2]
    for source,target in [keys,keys[::-1]]:
        graph['edges'].append(dict(source=source,relation=relation,target=target,basis='structural',explanation='cycle',evidence=['paper:transformer']))
    with pytest.raises(ValueError, match='Cycle'):
        KnowledgeGraph.model_validate(graph)


def fragment(key):
    return {'schema_version':'0.2', 'nodes':[dict(id='concept:'+key,kind='concept',label=key,summary='Curated extension')],
            'edges':[dict(source='model:transformer',relation='uses',target='concept:'+key,basis='pedagogical',
                explanation='Extension demonstration.',evidence=['paper:transformer'])]}


def test_incremental_updates_are_atomic_and_conflicts_preserve_data(tmp_path):
    store = GraphStore(tmp_path/'kg.sqlite3')
    old = store.load_optional()[0]
    result = store.extend(fragment('new-concept'), expected_revision=old)
    assert result['added_nodes'] == result['added_edges'] == 1
    assert store.extend(fragment('new-concept'))['revision'] == result['revision']
    changed = fragment('new-concept')
    changed['nodes'][0]['summary'] = 'Different'
    with pytest.raises(ValueError, match='Conflicting'):
        store.extend(changed)
    with pytest.raises(ValueError, match='Stale'):
        store.extend(fragment('another'), expected_revision=old)
    invalid = fragment('bad')
    invalid['edges'][0]['evidence'] = ['paper:missing']
    with pytest.raises(ValueError):
        store.extend(invalid)
    assert store.load_optional()[0] == result['revision']


def test_competing_extensions_use_revision_lock(tmp_path):
    store = GraphStore(tmp_path/'kg.sqlite3')
    revision = store.load_optional()[0]
    def update(key):
        try:
            return store.extend(fragment(key), expected_revision=revision)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(update, ['first','second']))
    assert sum(x is not None for x in outcomes) == 1


def test_legacy_import_and_upgrade_keep_local_curation(tmp_path):
    store = GraphStore(tmp_path/'kg.sqlite3')
    data = seed_graph().model_dump(mode='json')
    data['schema_version'] = '0.1'
    graph = KnowledgeGraph.model_validate(data)
    assert graph.schema_version == '0.2'
    graph.nodes[0].label = 'My note'
    store.replace(graph)
    result = store.extend(seed_graph().model_dump(mode='json'), keep_existing=True)
    assert result['retained_conflicts']
    assert store.load().nodes[0].label == 'My note'


def test_neighborhood_limits_and_api_extension(tmp_path):
    with TestClient(create_app(tmp_path/'kg.sqlite3')) as client:
        revision = client.get('/api/kg').json()['revision']
        neighborhood = client.get('/api/kg/neighborhood/model:transformer?hops=2&limit=5').json()
        assert len(neighborhood['nodes']) == 5 and neighborhood['truncated']
        assert client.get('/api/kg/neighborhood/model:transformer?hops=9').status_code == 422
        assert client.get('/api/kg/query?concept=concept:attention').json()['matches']
        assert client.get('/api/kg/query?concept=paper:bert').status_code == 422
        for route in ('ontology','shapes','export?format=turtle'):
            assert len(Graph().parse(data=client.get('/api/kg/'+route).text, format='turtle')) > 10
        payload = {'expected_revision':revision,'patch':fragment('api-addition')}
        assert client.post('/api/kg/extend',json=payload,headers={'Origin':'https://elsewhere.test'}).status_code == 403
        assert client.post('/api/kg/extend',json=payload).status_code == 200
        assert client.post('/api/kg/extend',json=payload).status_code == 409
