"""Build a portable, read-only learning viewer from a validated catalog."""
from importlib.resources import files
from pathlib import Path
import json


def export_viewer(store, output, *, runtime=False):
    """Write static assets and optionally actual, untrained module snapshots.

    Existing output directories are rejected: an old snapshot must never appear
    to belong to a newer catalog. No database or session state is copied.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    revision, graph = store.load_optional()
    def copy_assets(source, target):
        for resource in source.iterdir():
            if resource.name in {'study.js', 'study.css', 'study-math.js'}:
                continue
            if resource.is_dir():
                nested = target/resource.name
                nested.mkdir()
                copy_assets(resource, nested)
            else:
                (target/resource.name).write_bytes(resource.read_bytes())
    copy_assets(files('p2i_kg').joinpath('static'), destination)
    index = destination/'index.html'
    html = index.read_text(encoding='utf-8').replace('<html lang="ko">', f'<html lang="ko" data-mode="static" data-revision="{revision}">')
    for old, new in [('/api/kg/export?format=turtle', './exports/catalog.ttl'),
                     ('/api/kg/export?format=jsonld', './exports/catalog.jsonld'),
                     ('/api/kg/export', './exports/catalog.json'),
                     ('/api/kg/ontology', './exports/ontology.ttl'),
                     ('/api/kg/shapes', './exports/shapes.ttl')]:
        html = html.replace(old, new)
    index.write_text(html, encoding='utf-8')

    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')

    write(destination/'catalog.json', {'revision': revision, **graph.model_dump(mode='json')})
    exports = destination/'exports'
    exports.mkdir()
    write(exports/'catalog.json', graph.model_dump(mode='json'))
    write(exports/'catalog.jsonld', graph.jsonld())
    from .ontology import data_graph, vocabulary, shapes
    for name, rdf in [('catalog', data_graph(graph)+vocabulary()), ('ontology', vocabulary()), ('shapes', shapes())]:
        (exports/(name+'.ttl')).write_text(rdf.serialize(format='turtle'), encoding='utf-8')
    snapshots = []
    if runtime:
        from .bindings import inspect_model
        (destination/'snapshots').mkdir()
        for runtime_id in sorted({node.runtime_id for node in graph.nodes if node.runtime_id}):
            result = inspect_model(runtime_id)
            node = next(n for n in graph.nodes if n.runtime_id == runtime_id)
            result['model_id'] = node.id
            result['label'] = node.label
            result['scope_note'] = node.runtime_note
            result['observed']['metadata']['name'] = node.label
            for module in result['observed']['modules']:
                origin = module['origin']
                if origin.get('file'):
                    origin['file'] = Path(origin['file']).name
            for operation in result['observed']['operations']:
                if operation.get('source') and operation['source'].get('file'):
                    operation['source']['file'] = Path(operation['source']['file']).name
            write(destination/'snapshots'/(runtime_id+'.json'), result)
            snapshots.append(runtime_id)
    result = {'catalog_revision': revision, 'runtime_snapshots': snapshots,
              'mode': 'read-only', 'weights': 'random initialization, not trained checkpoints'}
    write(destination/'viewer-manifest.json', result)
    return result
