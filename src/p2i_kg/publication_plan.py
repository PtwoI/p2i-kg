"""Read-only, compare-and-swap publication plans for the GitHub publisher.

This module never pushes or changes a checkout. The caller must commit the plan
against its expected homepage parent and recheck KG main before merging.
"""
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
import re
import subprocess

from .publication import verify_pages

PREFIX = 'apps/p2imap/'


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _safe_file(root, name):
    relative = PurePosixPath(name)
    if (not name or '\\' in name or relative.is_absolute()
            or any(part in ('.', '..', '.git') for part in name.split('/'))):
        raise ValueError(f'Unsafe publication path: {name}')
    path = root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f'Symlink in publication path: {name}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Publication path escapes root: {name}')
    return path


def _manifest(root):
    manifest_path = _safe_file(root, 'viewer-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or 'viewer-manifest.json' in files:
        raise ValueError('A complete publication checksum manifest is required')
    for name, digest in files.items():
        path = _safe_file(root, name)
        if not path.is_file() or _digest(path) != digest:
            raise ValueError(f'Managed file modified or missing; preserve local UI: {name}')
    required = {'catalog.json', 'exports/catalog.json'} | {
        f'snapshots/{runtime}.json' for runtime in manifest.get('runtime_snapshots', [])}
    if not required <= files.keys():
        raise ValueError('Manifest does not cover catalog, exports and snapshots')
    return manifest


def publication_plan(build, homepage, *, expected_source, expected_home):
    """Return only authorized file changes, bound to a clean homepage commit.

    Unmanaged files are preserved; collisions fail. Modified generated UI fails
    closed rather than silently reverting a customization. Duplicate deliveries
    for the same source/catalog return no changes, even if a rebuild has different
    random weights. Caller must validate CI for expected_source separately.
    """
    for value in (expected_source, expected_home):
        if not re.fullmatch(r'[0-9a-f]{40}', value):
            raise ValueError('Full lowercase Git commit SHAs are required')
    build, homepage = Path(build).resolve(), Path(homepage).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(homepage), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != expected_home:
        raise ValueError('Homepage HEAD changed; refresh and rebuild the plan')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Homepage checkout must be clean')
    current = homepage / PREFIX
    if current.is_symlink() or (homepage / 'apps').is_symlink():
        raise ValueError('Homepage publication directory cannot be a symlink')
    new, old = _manifest(build), _manifest(current)
    if new.get('source_repository') != 'PtwoI/p2i-kg' or new.get('source_revision') != expected_source:
        raise ValueError('Build does not match the requested KG source')
    result = verify_pages(build)
    if new.get('model_count') != result['models'] or len(new.get('runtime_snapshots', [])) != result['models']:
        raise ValueError('Publication model count or unique runtime count differs')
    if len(set(new['runtime_snapshots'])) != len(new['runtime_snapshots']):
        raise ValueError('Duplicate runtime snapshots')
    # No extra unchecksummed assets may sneak into a publication artifact.
    actual = {p.relative_to(build).as_posix() for p in build.rglob('*') if p.is_file()}
    if actual != set(new['files']) | {'viewer-manifest.json'}:
        raise ValueError('Build contains unchecksummed files')
    plan = {'source_repository': 'PtwoI/p2i-kg', 'source_revision': expected_source,
            'homepage_repository': 'K0ykwon/k0ykwon.github.io',
            'expected_homepage_revision': expected_home, 'catalog_revision': new['catalog_revision'],
            'changes': [], 'verification': result}
    if old.get('source_revision') == expected_source and old['catalog_revision'] == new['catalog_revision']:
        return plan
    for name in sorted(set(new['files']) | set(old['files']) | {'viewer-manifest.json'}):
        target = _safe_file(current, name)
        source = _safe_file(build, name)
        if name not in old['files'] and name != 'viewer-manifest.json' and target.exists():
            raise ValueError(f'New generated file collides with unmanaged file: {name}')
        before = _digest(target) if target.is_file() else None
        after = _digest(source) if name in new['files'] or name == 'viewer-manifest.json' else None
        if before != after:
            plan['changes'].append({'path': PREFIX + name, 'before_sha256': before,
                                    'after_sha256': after, 'operation': 'delete' if after is None else 'write'})
    return plan
