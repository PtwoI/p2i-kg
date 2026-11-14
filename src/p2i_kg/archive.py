"""Versioned structure snapshots owned by KG, not by the deployment repository.

The archive contains untrained, reduced demonstration structures, not weights.
Its generation revision is intentionally distinct from the deployment revision:
committing an archive must not require its manifest to contain its own Git hash.
"""
from hashlib import sha256
from pathlib import Path
import json
import re
import shutil

REPOSITORY = 'PtwoI/p2i-kg'
DEFAULT_ARCHIVE = Path(__file__).resolve().parents[2] / 'model-archive'
SOURCES = ('catalog.json', 'runtime.py', 'bindings.py', 'models.py',
           'extended_models.py', 'extended_registry.py', 'rl_models.py', 'viewer.py')


def source_hashes():
    root = Path(__file__).parent
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}


def verify_archive(directory=DEFAULT_ARCHIVE, *, catalog_revision=None):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('repository') != REPOSITORY:
        raise ValueError('Unsupported canonical archive manifest')
    if manifest.get('source_files') != source_hashes():
        raise ValueError('Stale model archive: catalog/runtime sources changed; regenerate snapshots')
    from .store import seed_graph
    graph = seed_graph()
    current_revision = sha256(json.dumps(graph.model_dump(mode='json'), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if manifest['catalog_revision'] != current_revision or (catalog_revision is not None and catalog_revision != current_revision):
        raise ValueError('Archive and catalog have different revisions')
    snapshots = manifest['snapshots']
    expected = {node.runtime_id: node.id for node in graph.nodes if node.kind == 'model'}
    if {key: entry['model_id'] for key, entry in snapshots.items()} != expected:
        raise ValueError('Archive model coverage differs from bundled catalog')
    if not snapshots or any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in snapshots):
        raise ValueError('Unsafe or empty archive runtime IDs')
    actual = {p.stem for p in (root / 'snapshots').glob('*.json')}
    if actual != set(snapshots):
        raise ValueError('Archive snapshot coverage differs from manifest')
    for key, entry in snapshots.items():
        path = root / 'snapshots' / (key + '.json')
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Archive snapshot checksum mismatch: {key}')
        data = json.loads(path.read_text())
        if data.get('model_id') != entry['model_id'] or data.get('runtime_id') != key:
            raise ValueError(f'Archive model binding mismatch: {key}')
        if not data.get('observed', {}).get('modules') or not data.get('architecture', {}).get('nodes'):
            raise ValueError(f'Empty archive structure: {key}')
    return manifest


def archive_pages(build, output):
    """Import a verified runtime build, preserving exact bytes and provenance."""
    from .publication import verify_pages
    build, output = Path(build), Path(output)
    verify_pages(build)
    published = json.loads((build / 'viewer-manifest.json').read_text())
    if published.get('source_repository') != REPOSITORY or not published.get('source_revision'):
        raise ValueError('Archive import requires a KG source revision')
    if published.get('snapshot_source_files') != source_hashes():
        raise ValueError('Build runtime provenance differs from checked-out source files')
    # Caller must generate/import using the same checked-out runtime sources.
    from .store import GraphStore
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        revision = GraphStore(Path(temporary) / 'catalog.sqlite3').load_optional()[0]
    if published['catalog_revision'] != revision:
        raise ValueError('Cannot archive a stale catalog')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(build / 'snapshots', output / 'snapshots')
    manifest = {
        'schema_version': 1, 'repository': REPOSITORY,
        'generation_revision': published.get('canonical_archive', {}).get('generation_revision', published['source_revision']),
        'catalog_revision': revision, 'source_files': source_hashes(),
        'weights': 'random initialization, not trained checkpoints',
        'snapshots': {key: {'model_id': json.loads((build/'snapshots'/(key+'.json')).read_text())['model_id'],
                            'sha256': sha256((build/'snapshots'/(key+'.json')).read_bytes()).hexdigest()}
                      for key in sorted(published['runtime_snapshots'])}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    verify_archive(output, catalog_revision=revision)
    return {'snapshots': len(manifest['snapshots']), 'generation_revision': manifest['generation_revision']}


def copy_archive(directory, destination, *, catalog_revision):
    manifest = verify_archive(directory, catalog_revision=catalog_revision)
    shutil.copytree(Path(directory) / 'snapshots', Path(destination) / 'snapshots')
    return manifest
