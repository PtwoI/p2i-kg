import json
import subprocess
import sys
from pathlib import Path
import pytest
from p2i_kg.store import GraphStore
from p2i_kg.viewer import export_viewer


def test_portable_viewer_has_catalog_exports_and_no_python_dependency(tmp_path):
    store = GraphStore(tmp_path/'db.sqlite3')
    target = tmp_path/'viewer'
    result = export_viewer(store, target)
    graph = store.load()
    assert result['catalog_revision'] == store.load_optional()[0]
    assert result['runtime_snapshots'] == []
    assert len(json.loads((target/'catalog.json').read_text())['nodes']) == len(graph.nodes)
    assert len(json.loads((target/'exports/catalog.jsonld').read_text())['@graph']) == len(graph.nodes)+len(graph.edges)
    html = (target/'index.html').read_text()
    assert 'data-mode="static"' in html and '/api/kg/export' not in html
    for asset in ('app.js', 'catalog-client.js', 'map.css'):
        assert (target/asset).stat().st_size > 0
    for export in ('catalog.ttl', 'ontology.ttl', 'shapes.ttl'):
        assert (target/'exports'/export).stat().st_size > 0
    with pytest.raises(FileExistsError):
        export_viewer(store, target)
    subprocess.run([sys.executable, '-c',
        'import sys; from p2i_kg.viewer import export_viewer; assert "torch" not in sys.modules'], check=True)


def test_default_view_is_a_map_without_course_screens():
    source = (Path(__file__).parents[1]/'src/p2i_kg/static/index.html').read_text()
    assert 'id="map"' in source and 'id="detail"' in source
    assert 'data-route="learn"' not in source and 'id="lesson"' not in source
    assert 'src="./app.js"' in source and 'href="./map.css"' in source


def test_static_client_preserves_semantic_results(tmp_path):
    """Hosted calculations agree with the live ontology, including closure."""
    store = GraphStore(tmp_path/'db.sqlite3')
    target = tmp_path/'viewer'
    export_viewer(store, target)
    graph = store.load()
    source = (target/'catalog-client.js').as_uri()
    script = f'''
import {{readFileSync}} from 'node:fs';
globalThis.document={{documentElement:{{dataset:{{mode:'static'}}}}}};
globalThis.location={{origin:'http://localhost'}};
globalThis.fetch=async()=>({{ok:true,json:async()=>JSON.parse(readFileSync({json.dumps(str(target/'catalog.json'))},'utf8'))}});
const {{read}}=await import({json.dumps(source)});
const compare=await read('/api/kg/compare?left=model:bert&right=model:gpt2');
const query=await read('/api/kg/query?concept=concept:attention');
const neighborhood=await read('/api/kg/neighborhood/model:transformer?hops=2&limit=40&relations=uses,has_component,subconcept_of');
console.log(JSON.stringify({{compare,query,neighborhood}}));
'''
    import shutil
    if not shutil.which('node'):
        pytest.skip('Node is required to check the hosted JavaScript client')
    result = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', script], text=True))
    comparison = graph.compare('model:bert', 'model:gpt2')
    for field in ('common', 'left_only', 'right_only', 'facets'):
        assert result['compare'][field] == comparison[field]
    assert [m['node']['id'] for m in result['query']['matches']] == [m['node']['id'] for m in graph.query(concept='concept:attention')]
    assert [m['proof_edges'] for m in result['query']['matches']] == [m['proof_edges'] for m in graph.query(concept='concept:attention')]
    assert result['neighborhood'] == graph.neighborhood('model:transformer', hops=2, limit=40,
        relations=['uses', 'has_component', 'subconcept_of'])
