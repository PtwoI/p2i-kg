import json
import shutil
import subprocess
from hashlib import sha256

import pytest
from p2i_kg.publication_plan import publication_plan

SOURCE = 'a' * 40


def package(root, source=SOURCE, asset='new'):
    root.mkdir(parents=True, exist_ok=True)
    (root/'exports').mkdir(exist_ok=True)
    (root/'snapshots').mkdir(exist_ok=True)
    catalog = {'revision': 'revision', 'nodes': [], 'edges': []}
    (root/'catalog.json').write_text(json.dumps(catalog))
    (root/'exports/catalog.json').write_text(json.dumps(catalog))
    (root/'app.js').write_text(asset)
    manifest = {'source_repository': 'PtwoI/p2i-kg', 'source_revision': source,
        'catalog_revision': 'revision', 'runtime_snapshots': [], 'model_count': 0,
        'files': {p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest()
                  for p in root.rglob('*') if p.is_file() and p.name != 'viewer-manifest.json'}}
    (root/'viewer-manifest.json').write_text(json.dumps(manifest))


@pytest.fixture
def setup(tmp_path):
    site, build = tmp_path/'site', tmp_path/'build'
    package(site/'apps/p2imap', source='b'*40, asset='old')
    (site/'index.html').write_text('preserve homepage')
    (site/'apps/p2imap/custom.css').write_text('unmanaged UI')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(site), *args], text=True).strip()
    git('init', '-q'); git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'initial')
    package(build)
    return site, build, git('rev-parse', 'HEAD')


def plan(setup):
    site, build, home = setup
    return publication_plan(build, site, expected_source=SOURCE, expected_home=home)


def test_plan_is_scoped_readonly_and_repeatable(setup):
    site, build, _ = setup
    first = plan(setup)
    assert first == plan(setup)
    assert {c['path'] for c in first['changes']} == {'apps/p2imap/app.js', 'apps/p2imap/viewer-manifest.json'}
    assert (site/'apps/p2imap/app.js').read_text() == 'old'
    assert (site/'index.html').read_text() == 'preserve homepage'
    assert (site/'apps/p2imap/custom.css').read_text() == 'unmanaged UI'


def test_rejects_changed_home_parent(setup):
    site, build, _ = setup
    with pytest.raises(ValueError, match='HEAD changed'):
        publication_plan(build, site, expected_source=SOURCE, expected_home='f'*40)


def test_rejects_dirty_checkout(setup):
    (setup[0]/'index.html').write_text('another user working')
    with pytest.raises(ValueError, match='clean'):
        plan(setup)


def test_rejects_wrong_source(setup):
    package(setup[1], source='c'*40)
    with pytest.raises(ValueError, match='requested KG'):
        plan(setup)


def test_rejects_missing_checksums(setup):
    p = setup[1]/'viewer-manifest.json'
    m = json.loads(p.read_text()); m['files'] = {}; p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='complete'):
        plan(setup)


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'a/../../outside', 'a\\b', '.git/config'])
def test_rejects_traversal(setup, name):
    p = setup[1]/'viewer-manifest.json'
    m = json.loads(p.read_text()); m['files'][name] = '0'*64; p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Unsafe'):
        plan(setup)


def test_rejects_symlink(setup):
    p = setup[1]/'app.js'; p.unlink(); p.symlink_to(setup[0]/'index.html')
    with pytest.raises(ValueError, match='Symlink'):
        plan(setup)


def test_rejects_unmanaged_collision(setup):
    (setup[1]/'custom.css').write_text('overwrite?'); package(setup[1])
    with pytest.raises(ValueError, match='unmanaged'):
        plan(setup)


def test_rejects_unchecksummed_file(setup):
    (setup[1]/'extra.js').write_text('not in manifest')
    with pytest.raises(ValueError, match='unchecksummed'):
        plan(setup)


def test_removes_only_old_managed_files(setup):
    p = setup[1]/'app.js'; p.unlink()
    mpath = setup[1]/'viewer-manifest.json'; m=json.loads(mpath.read_text())
    del m['files']['app.js']; mpath.write_text(json.dumps(m))
    changes = plan(setup)['changes']
    assert [c['path'] for c in changes if c['operation']=='delete'] == ['apps/p2imap/app.js']


def test_duplicate_delivery_is_noop(setup):
    site, build, _ = setup
    package(build, source='b'*40)
    result = publication_plan(build, site, expected_source='b'*40, expected_home=setup[2])
    assert result['changes'] == []
