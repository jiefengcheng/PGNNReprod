"""Fetch the specific published TxGNN raw assets into an ignored local folder.

No dataset is shipped in the repository. This acquisition module intentionally
writes files only to a caller-chosen local location.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone

ASSETS = {
    'kg.csv': (7144484, ',', {'x_index', 'y_index', 'x_type', 'x_id', 'relation', 'y_type', 'y_id'}),
    'node.csv': (7144482, '\t', {'node_index', 'node_id', 'node_source'}),
    'edges.csv': (7144483, ',', {'x_index', 'y_index', 'relation'}),
}

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def checked_header(path, delimiter, required):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream, delimiter=delimiter))
    missing = required.difference(header)
    if missing:
        raise ValueError('Invalid asset header: {} missing {}'.format(path, sorted(missing)))
    return header

def acquire_assets(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_path = raw_dir / 'asset_lock.json'
    prior = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    records = {}
    for filename, (datafile_id, delimiter, required) in ASSETS.items():
        destination = raw_dir / filename
        url = 'https://dataverse.harvard.edu/api/access/datafile/{}'.format(datafile_id)
        if not destination.exists():
            descriptor, temporary = tempfile.mkstemp(dir=str(raw_dir), prefix='download_', suffix='.part')
            temporary = Path(temporary)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        if response.status != 200:
                            raise RuntimeError('Unexpected HTTP status {}'.format(response.status))
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                checked_header(temporary, delimiter, required)
                if filename in prior and file_sha256(temporary) != prior[filename]['sha256']:
                    raise ValueError('Acquired bytes differ from the existing asset lock: ' + filename)
                os.replace(str(temporary), str(destination))
            finally:
                if temporary.exists():
                    temporary.unlink()
        header = checked_header(destination, delimiter, required)
        digest = file_sha256(destination)
        if filename in prior and digest != prior[filename]['sha256']:
            raise ValueError('Existing bytes differ from the asset lock: ' + filename)
        records[filename] = {
            'url': url, 'datafile_id': datafile_id, 'sha256': digest,
            'bytes': destination.stat().st_size, 'delimiter': delimiter, 'header': header,
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        }
    temporary_lock = raw_dir / 'asset_lock.json.part'
    temporary_lock.write_text(json.dumps(records, indent=2) + '\n')
    os.replace(str(temporary_lock), str(lock_path))
    return records

