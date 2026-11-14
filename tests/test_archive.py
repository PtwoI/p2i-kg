import json
import shutil
from pathlib import Path
import pytest
from p2i_kg.archive import DEFAULT_ARCHIVE, verify_archive, archive_pages
from p2i_kg.publication import build_pages


def test_canonical_archive_covers_every_model():
    manifest = verify_archive()
    assert len(manifest['snapshots']) >= 79


def test_pages_uses_exact_canonical_bytes_without_runtime(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torch', None)
    output = tmp_path/'pages'
    result = build_pages(output, source_revision='deployment-sha')
    manifest = verify_archive()
    assert result['snapshots'] == len(manifest['snapshots'])
    for key in manifest['snapshots']:
        assert (output/'snapshots'/(key+'.json')).read_bytes() == (DEFAULT_ARCHIVE/'snapshots'/(key+'.json')).read_bytes()
    published = json.loads((output/'viewer-manifest.json').read_text())
    assert published['canonical_archive']['revision'] == 'deployment-sha'
    assert published['canonical_archive']['generation_revision'] == manifest['generation_revision']
    archive_pages(output, tmp_path/'roundtrip')
    assert verify_archive(tmp_path/'roundtrip')['snapshots'] == manifest['snapshots']


@pytest.mark.parametrize('damage', ['checksum', 'coverage', 'sources', 'catalog'])
def test_archive_fails_closed(tmp_path, damage):
    root = tmp_path/'archive'
    shutil.copytree(DEFAULT_ARCHIVE, root)
    path = root/'manifest.json'
    manifest = json.loads(path.read_text())
    key = next(iter(manifest['snapshots']))
    if damage == 'checksum':
        (root/'snapshots'/(key+'.json')).write_text('{}')
    elif damage == 'coverage':
        manifest['snapshots'].pop(key)
    elif damage == 'sources':
        manifest['source_files']['runtime.py'] = 'stale'
    elif damage == 'catalog':
        manifest['catalog_revision'] = 'stale'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_archive(root, catalog_revision=verify_archive()['catalog_revision'])
