"""Build and verify the complete map before publishing it to GitHub Pages."""
from hashlib import sha256
from pathlib import Path
import json
import tempfile

from .store import GraphStore
from .viewer import export_viewer


def verify_pages(directory):
    root = Path(directory)
    manifest = json.loads((root / 'viewer-manifest.json').read_text())
    catalog = json.loads((root / 'catalog.json').read_text())
    if manifest['catalog_revision'] != catalog['revision']:
        raise ValueError('Catalog and runtime snapshots have different revisions')
    models = [node for node in catalog['nodes'] if node['kind'] == 'model']
    missing = [node['id'] for node in models if not node.get('runtime_id')]
    if missing:
        raise ValueError(f'Models without P2I implementations: {missing}')
    expected = {node['runtime_id'] for node in models}
    actual = {path.stem for path in (root / 'snapshots').glob('*.json')}
    if expected != actual or expected != set(manifest['runtime_snapshots']):
        raise ValueError('P2I snapshot coverage does not match the current catalog')
    for node in models:
        snapshot = json.loads((root / 'snapshots' / (node['runtime_id'] + '.json')).read_text())
        if snapshot.get('model_id') != node['id']:
            raise ValueError(f'Incorrect model binding: {node["id"]}')
        if not snapshot.get('observed', {}).get('modules') or not snapshot.get('architecture', {}).get('nodes'):
            raise ValueError(f'Empty P2I structure: {node["id"]}')
    for name, checksum in manifest.get('files', {}).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Publication file differs from its manifest: {name}')
    return {'models': len(models), 'nodes': len(catalog['nodes']),
            'edges': len(catalog['edges']), 'snapshots': len(expected),
            'catalog_revision': catalog['revision'], 'source_revision': manifest.get('source_revision')}


def build_pages(output, *, source_revision):
    """Use a fresh bundled catalog, never the maintainer's older local database.

    The caller publishes only after this returns; failures leave the live site
    untouched. A new directory is required so obsolete model snapshots cannot
    survive a later catalog change.
    """
    root = Path(output)
    with tempfile.TemporaryDirectory(prefix='p2imap-catalog-') as temporary:
        store = GraphStore(Path(temporary) / 'catalog.sqlite3')
        manifest = export_viewer(store, root, runtime=True)
    manifest['source_repository'] = 'PtwoI/p2i-kg'
    manifest['source_revision'] = source_revision
    manifest['model_count'] = len([node for node in json.loads((root / 'catalog.json').read_text())['nodes']
                                   if node['kind'] == 'model'])
    # Version all HTML/JS/CSS references together, including nested P2I assets.
    version = sha256((manifest['catalog_revision'] + source_revision).encode()).hexdigest()[:16]
    for path in root.rglob('*'):
        if path.suffix in ('.html', '.js', '.css'):
            path.write_text(path.read_text().replace('neon-v1', version), encoding='utf-8')
    manifest['build_version'] = version
    manifest['files'] = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                         for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'viewer-manifest.json'}
    (root / 'viewer-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return verify_pages(root)
