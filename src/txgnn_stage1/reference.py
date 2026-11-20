"""Optional pinned-upstream reference snapshot exporter.

Run only in the separately managed legacy TxGNN reference environment. Outputs
include raw derived graph arrays and are ignored by git.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .assets import ASSETS, file_sha256

def capture_reference(raw_dir, repo_dir, work_dir, config):
    import inspect
    import platform
    import random
    import subprocess
    import torch
    import dgl
    from txgnn import TxData
    from txgnn.utils import initialize_node_embedding
    from txgnn.data_splits.datasplit import DataSplitter

    raw_dir, repo_dir, work_dir = map(Path, (raw_dir, repo_dir, work_dir))
    sha = subprocess.check_output(['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != config['reference_sha']:
        raise ValueError('Reference checkout does not match the configured SHA.')
    subprocess.run(['git', '-C', str(repo_dir), 'diff', '--exit-code', 'HEAD', '--', 'txgnn'], check=True)
    installed_file = Path(inspect.getfile(TxData)).resolve()
    if repo_dir.resolve() not in installed_file.parents:
        raise ValueError('TxData is imported from a different installation: ' + str(installed_file))
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('Reference input recipe requires default float32.')
    lock = json.loads((raw_dir / 'asset_lock.json').read_text())
    for filename in ASSETS:
        if file_sha256(raw_dir / filename) != lock[filename]['sha256']:
            raise ValueError('Raw asset digest mismatch: ' + filename)

    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='reference_', dir=str(work_dir)))
    data_dir, out = run_dir / 'data', run_dir / 'snapshot'
    data_dir.mkdir()
    out.mkdir()
    for filename in ASSETS:
        try:
            os.link(str(raw_dir / filename), str(data_dir / filename))
        except OSError:
            shutil.copy2(str(raw_dir / filename), str(data_dir / filename))
    shutil.copytree(str(repo_dir / 'data' / 'disease_files'), str(data_dir / 'disease_files'))

    def numpy_state():
        name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        return {'algorithm': name, 'keys': keys.tolist(), 'position': position,
                'has_gauss': has_gauss, 'cached_gaussian': cached_gaussian}

    # Seed after imports. This defines the owned protocol, not an unknown historical run.
    np.random.seed(config['preprocess_seed'])
    random.seed(config['preprocess_seed'])
    states = {'numpy_preparation_before': numpy_state(),
              'python_preparation_before': random.getstate()}
    original_area_selector = DataSplitter.get_test_kg_for_disease

    def record_area_selection(instance, *args, **kwargs):
        selected = original_area_selector(instance, *args, **kwargs)
        selected[['x_index', 'y_index']].drop_duplicates().to_csv(out / 'area_mask_pairs.csv', index=False)
        return selected

    DataSplitter.get_test_kg_for_disease = record_area_selection
    try:
        reference_data = TxData(data_folder_path=str(data_dir))
        reference_data.prepare_split(
            split=config['split'], seed=config['fold_seed'], no_kg=config['no_kg'],
            test_size=config['test_size'], one_hop=config['one_hop'], mask_ratio=config['mask_ratio'],
        )
    finally:
        DataSplitter.get_test_kg_for_disease = original_area_selector

    states['numpy_preparation_after'] = numpy_state()
    states['python_preparation_after'] = random.getstate()
    (out / 'preparation_rng_states.json').write_text(json.dumps(states) + '\n')
    for filename, frame in [
        ('processed.csv', reference_data.df), ('train.csv', reference_data.df_train),
        ('valid.csv', reference_data.df_valid), ('evaluation_test.csv', reference_data.df_test),
    ]:
        frame.to_csv(out / filename, index=False)
    cached_test_files = list(data_dir.rglob('test.csv'))
    if len(cached_test_files) != 1:
        raise ValueError('Expected exactly one base test cache in the fresh reference directory.')
    shutil.copy2(str(cached_test_files[0]), str(out / 'base_test.csv'))

    graph = reference_data.G.to('cpu')
    torch.manual_seed(config['input_seed'])
    np.save(out / 'torch_cpu_rng_before_inputs.npy', torch.get_rng_state().cpu().numpy(), allow_pickle=False)
    graph = initialize_node_embedding(graph, config['input_dim'])
    np.save(out / 'torch_cpu_rng_after_inputs.npy', torch.get_rng_state().cpu().numpy(), allow_pickle=False)
    node_types = list(graph.ntypes)
    canonical_etypes = [tuple(t) for t in graph.canonical_etypes]
    schema = {
        'node_types': node_types,
        'node_counts': {t: int(graph.number_of_nodes(t)) for t in node_types},
        'canonical_etypes': [list(t) for t in canonical_etypes],
        'edge_type_name_order': list(graph.etypes),
        'relation_parameter_rows': [{'canonical_etype': list(t), 'row': i} for i, t in enumerate(canonical_etypes)],
        'reverse_relations': {},
    }
    validation_etypes = list(canonical_etypes)
    base_types = reference_data.df[['x_type', 'relation', 'y_type']].drop_duplicates()
    for source_type, relation, target_type in base_types.itertuples(index=False, name=None):
        forward = (source_type, relation, target_type)
        inverse_name = relation if source_type == target_type else 'rev_' + relation
        backward = (target_type, inverse_name, source_type)
        for triple in [forward, backward]:
            if triple not in validation_etypes:
                validation_etypes.append(triple)
        if source_type != target_type:
            schema['reverse_relations'][inverse_name] = relation
    schema['validation_canonical_etypes'] = [list(t) for t in validation_etypes]
    for i, node_type in enumerate(node_types):
        values = graph.nodes[node_type].data['inp']
        if values.requires_grad or values.dtype != torch.float32:
            raise ValueError('Unexpected reference node-input contract.')
        np.save(out / ('input_{:03d}.npy'.format(i)), values.detach().cpu().numpy(), allow_pickle=False)
    for i, canonical_etype in enumerate(canonical_etypes):
        source, target = graph.edges(etype=canonical_etype, order='eid')
        array = np.stack([source.cpu().numpy(), target.cpu().numpy()]).astype(np.int64, copy=False)
        np.save(out / ('edge_{:03d}.npy'.format(i)), array, allow_pickle=False)
    (out / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'numpy': np.__version__, 'pandas': pd.__version__,
                   'torch': torch.__version__, 'dgl': dgl.__version__}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out / 'requirements_observed.txt').write_text(freeze)
    auxiliary_files = [repo_dir / 'txgnn' / 'data_splits' / name for name in
                       ['HumanDO.obo', 'mondo_references.csv', 'kg_grouped_diseases_bert_map.csv']]
    auxiliary_files.extend(sorted((repo_dir / 'data' / 'disease_files').glob('*.csv')))
    auxiliary_hashes = {str(p.relative_to(repo_dir)): file_sha256(p) for p in auxiliary_files}
    manifest = {'schema_version': 1, 'config': config, 'assets': lock,
                'auxiliary_sha256': auxiliary_hashes,
                'status': 'reference snapshot; independent full-data parity pending',
                'files': {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return out
