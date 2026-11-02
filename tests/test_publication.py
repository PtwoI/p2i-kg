import json
import pytest
from p2i_kg.publication import verify_pages


def test_publication_rejects_new_model_without_runtime_snapshot(tmp_path):
    (tmp_path / 'catalog.json').write_text(json.dumps({'revision': 'new', 'nodes': [
        {'id': 'model:new', 'kind': 'model', 'runtime_id': 'new'}], 'edges': []}))
    (tmp_path / 'viewer-manifest.json').write_text(json.dumps({
        'catalog_revision': 'new', 'runtime_snapshots': []}))
    (tmp_path / 'snapshots').mkdir()
    with pytest.raises(ValueError, match='coverage'):
        verify_pages(tmp_path)


def test_publication_rejects_stale_catalog_even_with_existing_snapshot(tmp_path):
    (tmp_path / 'catalog.json').write_text(json.dumps({'revision': 'new', 'nodes': [], 'edges': []}))
    (tmp_path / 'viewer-manifest.json').write_text(json.dumps({'catalog_revision': 'old'}))
    with pytest.raises(ValueError, match='different revisions'):
        verify_pages(tmp_path)


def test_publication_rejects_changed_asset(tmp_path):
    (tmp_path / 'catalog.json').write_text(json.dumps({'revision': 'same', 'nodes': [], 'edges': []}))
    (tmp_path / 'snapshots').mkdir()
    (tmp_path / 'app.js').write_text('unexpected replacement')
    (tmp_path / 'viewer-manifest.json').write_text(json.dumps({
        'catalog_revision': 'same', 'runtime_snapshots': [], 'files': {'app.js': 'outdated checksum'}}))
    with pytest.raises(ValueError, match='differs'):
        verify_pages(tmp_path)
