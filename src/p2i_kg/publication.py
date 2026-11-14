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


def build_pages(output, *, source_revision, archive=None, regenerate_snapshots=False):
    """Use a fresh bundled catalog, never the maintainer's older local database.

    The caller publishes only after this returns; failures leave the live site
    untouched. A new directory is required so obsolete model snapshots cannot
    survive a later catalog change.
    """
    root = Path(output)
    with tempfile.TemporaryDirectory(prefix='p2imap-catalog-') as temporary:
        store = GraphStore(Path(temporary) / 'catalog.sqlite3')
        manifest = export_viewer(store, root, runtime=regenerate_snapshots)
        if not regenerate_snapshots:
            from .archive import DEFAULT_ARCHIVE, copy_archive
            archived = copy_archive(archive or DEFAULT_ARCHIVE, root,
                                    catalog_revision=manifest['catalog_revision'])
            manifest['runtime_snapshots'] = sorted(archived['snapshots'])
            manifest['canonical_archive'] = {
                'repository': 'PtwoI/p2i-kg', 'path': 'model-archive',
                'revision': source_revision,
                'generation_revision': archived['generation_revision'],
                'manifest_sha256': sha256((Path(archive or DEFAULT_ARCHIVE) / 'manifest.json').read_bytes()).hexdigest()}

    from .archive import source_hashes
    manifest['snapshot_source_files'] = source_hashes()
    manifest['source_repository'] = 'PtwoI/p2i-kg'
    manifest['source_revision'] = source_revision
    manifest['model_count'] = len([node for node in json.loads((root / 'catalog.json').read_text())['nodes']
                                   if node['kind'] == 'model'])
    # Version all HTML/JS/CSS references together, including nested P2I assets.
    version = sha256((manifest['catalog_revision'] + source_revision).encode()).hexdigest()[:16]
    for path in root.rglob('*'):
        if path.suffix in ('.html', '.js', '.css'):
            text = path.read_text().replace('neon-v1', version)
            if path == root / 'index.html':
                text = text.replace('data-mode="static"', f'data-mode="static" data-build="{version}"')
            path.write_text(text, encoding='utf-8')
    manifest['build_version'] = version
    manifest['files'] = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                         for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'viewer-manifest.json'}
    (root / 'viewer-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return verify_pages(root)
