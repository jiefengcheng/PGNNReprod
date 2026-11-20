"""Compare independently assembled graph and initialization with an upstream snapshot.

This stage does not assert independent raw-data normalization/splitting parity.
"""
from pathlib import Path
import json
import hashlib
import sys
import tempfile
import numpy as np
import pandas as pd
from .assets import file_sha256
from .graph import build_graph
from .inputs import make_owned_inputs

def compare_real_snapshot(snapshot_dir, output_parent):
    import subprocess
    snapshot_dir = Path(snapshot_dir)
    source_manifest = json.loads((snapshot_dir / 'manifest.json').read_text())
    for filename, expected in source_manifest['files'].items():
        if file_sha256(snapshot_dir / filename) != expected:
            raise ValueError('Reference snapshot changed: ' + filename)
    schema = json.loads((snapshot_dir / 'schema.json').read_text())
    config = source_manifest['config']

    def read_frame(filename):
        return pd.read_csv(snapshot_dir / filename,
                           dtype={'x_id': str, 'y_id': str, 'x_type': str,
                                  'y_type': str, 'relation': str}, low_memory=False)

    full = read_frame('processed.csv')
    splits = {'train': read_frame('train.csv'), 'valid': read_frame('valid.csv'),
              'test': read_frame('base_test.csv')}
    owned = build_graph(full, splits, schema)
    if owned.num_nodes != schema['node_counts']:
        raise AssertionError('Node inventory differs from the reference graph.')
    for i, etype in enumerate(owned.canonical_etypes):
        expected = np.load(snapshot_dir / ('edge_{:03d}.npy'.format(i)), allow_pickle=False)
        np.testing.assert_array_equal(owned.edges[etype], expected)

    inputs, before, after = make_owned_inputs(owned.num_nodes, owned.node_types,
                                            config['input_dim'], config['input_seed'])
    np.testing.assert_array_equal(before.cpu().numpy(), np.load(snapshot_dir / 'torch_cpu_rng_before_inputs.npy', allow_pickle=False))
    np.testing.assert_array_equal(after.cpu().numpy(), np.load(snapshot_dir / 'torch_cpu_rng_after_inputs.npy', allow_pickle=False))
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='adapter_check_', dir=str(output_parent)))
    input_digests = {}
    for i, ntype in enumerate(owned.node_types):
        values = inputs[ntype].detach().cpu().numpy()
        expected = np.load(snapshot_dir / ('input_{:03d}.npy'.format(i)), allow_pickle=False)
        np.testing.assert_array_equal(values, expected)
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise AssertionError('Invalid input matrix: ' + ntype)
        header = json.dumps([ntype, list(values.shape), 'float32'], separators=(',', ':')).encode()
        h = hashlib.sha256(len(header).to_bytes(8, 'little') + header)
        h.update(np.ascontiguousarray(values, dtype='<f4').tobytes())
        input_digests[ntype] = h.hexdigest()
        np.save(out / ('input_{:03d}.npy'.format(i)), values, allow_pickle=False)
    for i, etype in enumerate(owned.canonical_etypes):
        np.save(out / ('edge_{:03d}.npy'.format(i)), owned.edges[etype], allow_pickle=False)
    (out / 'node_reference_ids.json').write_text(json.dumps(owned.node_ids) + '\n')
    (out / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    report = {
        'scope': 'reference-prepared tables to independently built graph and input matrices',
        'full_owned_preparation_complete': False,
        'independent_raw_preparation_parity': 'pending',
        'ordered_graph_arrays_equal': True,
        'frozen_inputs_equal': True,
        'graph_ordered_hash': owned.logical_hash(),
        'graph_edge_multiset_hash': owned.logical_hash(ordered=False),
        'input_logical_hashes': input_digests,
        'input_dim': config['input_dim'],
        'artifact_sha256': {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')

    # A new process reopens the stored arrays and schema and verifies their content.
    verifier = """
from pathlib import Path
import hashlib, json, sys
import numpy as np
p = Path(sys.argv[1])
c = json.loads((p / 'checks.json').read_text())
s = json.loads((p / 'schema.json').read_text())
for name, wanted in c['artifact_sha256'].items():
    h = hashlib.sha256()
    with (p / name).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == wanted, name
for i, ntype in enumerate(s['node_types']):
    x = np.load(p / ('input_{:03d}.npy'.format(i)), allow_pickle=False)
    assert x.shape == (s['node_counts'][ntype], c['input_dim']) and x.dtype == np.float32
for i, (src, rel, dst) in enumerate(s['canonical_etypes']):
    e = np.load(p / ('edge_{:03d}.npy'.format(i)), allow_pickle=False)
    assert e.ndim == 2 and e.shape[0] == 2 and e.dtype == np.int64
    if e.shape[1]:
        assert e.min() >= 0 and e[0].max() < s['node_counts'][src] and e[1].max() < s['node_counts'][dst]
print('Fresh-process array and schema verification passed.')
"""
    subprocess.run([sys.executable, '-c', verifier, str(out)], check=True)
    report['fresh_process_array_and_schema_verification'] = True
    (out / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    return out, report
