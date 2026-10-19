"""Verify and unpack the bundled experiment inputs; no network or API calls."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    for item in manifest['files']:
        target = (ROOT / item['path']).resolve()
        assert target.is_relative_to(ROOT), item['path']
        archive = target.with_name(target.name + '.gz')
        assert digest(archive) == item['compressed_sha256'], f'Archive hash mismatch: {archive}'
        if target.exists():
            assert digest(target) == item['sha256'], f'Existing data differs: {target}'
            print('Verified', item['path'])
            continue
        with tempfile.TemporaryDirectory(prefix='llm-data-') as temporary:
            unpacked = Path(temporary) / 'unpacked'
            with gzip.open(archive, 'rb') as source, unpacked.open('wb') as out:
                shutil.copyfileobj(source, out)
            assert unpacked.stat().st_size == item['bytes']
            assert digest(unpacked) == item['sha256'], f'Data hash mismatch: {archive}'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(unpacked, target)
        print('Unpacked', item['path'])


if __name__ == '__main__':
    main()
