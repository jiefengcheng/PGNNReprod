"""Serialize and reload an owned Step 1 bundle. Loading does not draw random numbers."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from .assets import ASSETS, file_sha256
from .graph import build_graph
from .inputs import INPUT_RNG_ALGORITHM, make_numpy_inputs
from .preprocess import (PROTOCOL, prepare_tables, read_group_map, read_released_tables,
                         attach_node_file)

NODE_COLUMNS = ["node_type", "local_idx", "reference_id", "raw_id", "name", "source"]


def _write_frame(path, frame):
    frame.to_csv(path, index=False)


def _hash_tree(root):
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[str(path.relative_to(root))] = file_sha256(path)
    return hashes


def prepare_bundle(raw_dir, out_dir, group_map_path=None, fold_seed=42, preprocess_seed=42,
                   input_seed=0, input_dim=100):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Refusing to write into a non-empty bundle directory: " + str(out_dir))
    out_dir.mkdir(parents=True)
    lock = json.loads((raw_dir / "asset_lock.json").read_text())
    for filename in ASSETS:
        if file_sha256(raw_dir / filename) != lock[filename]["sha256"]:
            raise ValueError("Raw asset digest mismatch: " + filename)
    print("Reading released tables", flush=True)
    kg, node_file = read_released_tables(raw_dir)
    group_map = read_group_map(group_map_path)
    print("Preparing tables: {} KG rows".format(len(kg)), flush=True)
    prepared = prepare_tables(kg, group_map=group_map, fold_seed=fold_seed)
    print("Building graph and inputs", flush=True)
    nodes, node_audit = attach_node_file(prepared["nodes"], node_file)
    base = _text_frame(prepared["base"])
    splits = {name: _text_frame(frame) for name, frame in prepared["splits"].items()}
    graph = build_graph(base, splits, prepared["schema"])
    inputs = make_numpy_inputs(graph.num_nodes, graph.node_types, input_dim, input_seed)
    (out_dir / "graph").mkdir()
    (out_dir / "inputs").mkdir()
    for index, etype in enumerate(graph.canonical_etypes):
        np.save(out_dir / "graph" / "edge_{:03d}.npy".format(index), graph.edges[etype])
    for index, node_type in enumerate(graph.node_types):
        np.save(out_dir / "inputs" / "input_{:03d}.npy".format(index), inputs[node_type])
    _write_frame(out_dir / "nodes.csv", nodes[NODE_COLUMNS])
    _write_frame(out_dir / "source_node_crosswalk.csv", prepared["crosswalk"])
    _write_frame(out_dir / "disease_membership.csv", prepared["membership"])
    # Reverse rows repeat source_row inside the same split. The first assignment wins.
    split_of = {}
    for name, frame in prepared["splits"].items():
        # Reverse rows repeat source_row. The base fact's split is the frame name.
        for source_row in frame.source_row.to_numpy():
            split_of.setdefault(int(source_row), name)
    base_membership = prepared["base"][["source_row"]].copy()
    base_membership["split"] = base_membership.source_row.map(lambda value: split_of[int(value)])
    _write_frame(out_dir / "split_membership.csv", base_membership)
    relations = dict(prepared["schema"])
    relations["relation_parameter_rows"] = [
        {"canonical_etype": etype, "row": index}
        for index, etype in enumerate(relations["canonical_etypes"])]
    (out_dir / "relations.json").write_text(json.dumps(relations, indent=2) + "\n")
    checks = dict(prepared["checks"])
    checks["node_file"] = node_audit
    checks["graph_ordered_hash"] = graph.logical_hash()
    checks["graph_edge_multiset_hash"] = graph.logical_hash(ordered=False)
    checks["node_counts"] = graph.num_nodes
    checks["canonical_etype_count"] = len(graph.canonical_etypes)
    checks["training_edges"] = {str(etype): int(graph.edges[etype].shape[1]) for etype in graph.canonical_etypes}
    checks["input"] = {"algorithm": INPUT_RNG_ALGORITHM, "seed": int(input_seed),
                       "dimension": int(input_dim), "dtype": "float32",
                       "note": "NumPy PCG64 uniform draws cast to float32. Not Torch bit patterns."}
    checks["preprocess_seed_consumed"] = False
    checks["holdout"] = _holdout_report(prepared["splits"], prepared["schema"]["reverse_relations"])
    # Collision lists can contain identifiers. Keep only the count in the saved audit.
    checks["normalization_collision_groups"] = len(prepared["checks"]["normalization_collisions"])
    checks.pop("normalization_collisions", None)
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    group_hash = None if group_map_path is None else file_sha256(group_map_path)
    manifest = {"schema_version": 1, "protocol": PROTOCOL,
                "config": {"split": "complex_disease", "fold_seed": int(fold_seed),
                           "preprocess_seed": int(preprocess_seed), "input_seed": int(input_seed),
                           "input_dim": int(input_dim), "preprocess_seed_consumed": False},
                "assets": {name: {"datafile_id": lock[name]["datafile_id"], "sha256": lock[name]["sha256"],
                                  "bytes": lock[name]["bytes"]} for name in lock},
                "group_map_sha256": group_hash,
                "numpy": np.__version__, "pandas": pd.__version__,
                "files": _hash_tree(out_dir)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def _text_frame(frame):
    out = frame.copy()
    for column in ["x_type", "x_id", "relation", "y_type", "y_id"]:
        out[column] = out[column].map(lambda value: value if isinstance(value, str) else str(value))
    return out


def _holdout_report(splits, reverse_relations):
    therapeutic = {"indication", "contraindication", "off-label use"}
    therapeutic.update("rev_" + name for name in list(therapeutic))
    def facts(frame):
        selected = frame[frame.relation.isin(therapeutic)][
            ["x_type", "x_id", "relation", "y_type", "y_id"]]
        return set(map(tuple, selected.itertuples(index=False, name=None)))
    train, valid, test = facts(splits["train"]), facts(splits["valid"]), facts(splits["test"])
    return {"therapeutic_train_valid_overlap": len(train & valid),
            "therapeutic_train_test_overlap": len(train & test),
            "therapeutic_valid_test_overlap": len(valid & test),
            "therapeutic_train": len(train), "therapeutic_valid": len(valid),
            "therapeutic_test": len(test),
            "reverse_relations": len(reverse_relations)}


def verify_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    mismatches = []
    for relative, expected in manifest["files"].items():
        actual = file_sha256(bundle_dir / relative)
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError("Bundle files changed: {}".format(mismatches))
    script = r"""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
relations = json.loads((root / "relations.json").read_text())
checks = json.loads((root / "checks.json").read_text())
counts = checks["node_counts"]
for index, node_type in enumerate(relations["node_types"]):
    values = np.load(root / "inputs" / "input_{:03d}.npy".format(index), allow_pickle=False)
    assert values.dtype == np.float32
    assert values.shape == (counts[node_type], manifest["config"]["input_dim"])
    assert np.isfinite(values).all()
for index, (src, rel, dst) in enumerate(relations["canonical_etypes"]):
    edges = np.load(root / "graph" / "edge_{:03d}.npy".format(index), allow_pickle=False)
    assert edges.dtype == np.int64 and edges.ndim == 2 and edges.shape[0] == 2
    if edges.shape[1]:
        assert edges.min() >= 0
        assert edges[0].max() < counts[src]
        assert edges[1].max() < counts[dst]
print("fresh-process reload ok")
"""
    subprocess.run([sys.executable, "-c", script, str(bundle_dir)], check=True)
    return {"fresh_process_reload": True, "files_checked": len(manifest["files"]),
            "protocol": manifest["protocol"], "graph_ordered_hash": json.loads(
                (bundle_dir / "checks.json").read_text())["graph_ordered_hash"]}
