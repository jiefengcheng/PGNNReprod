"""Tiny on-disk bundle: hashes, holdout, and a fresh-process reload."""
import json
import unittest
from pathlib import Path
import pandas as pd
from txgnn_stage1.assets import file_sha256
from txgnn_stage1.bundle import prepare_bundle, verify_bundle


def _write_lock(raw):
    kg = pd.DataFrame([
        {"x_index": 0, "x_id": "DB1", "x_type": "drug", "x_name": "d", "x_source": "DrugBank",
         "relation": "indication", "y_index": 1, "y_id": "10", "y_type": "disease",
         "y_name": "flu", "y_source": "MONDO"},
        {"x_index": 1, "x_id": "10", "x_type": "disease", "x_name": "flu", "x_source": "MONDO",
         "relation": "indication", "y_index": 0, "y_id": "DB1", "y_type": "drug",
         "y_name": "d", "y_source": "DrugBank"},
        {"x_index": 2, "x_id": "A", "x_type": "disease", "x_name": "a", "x_source": "MONDO",
         "relation": "disease_disease", "y_index": 2, "y_id": "A", "y_type": "disease",
         "y_name": "a", "y_source": "MONDO"},
    ])
    nodes = pd.DataFrame([
        {"node_index": 0, "node_id": "DB1", "node_type": "drug", "node_name": "d", "node_source": "DrugBank"},
        {"node_index": 1, "node_id": "10", "node_type": "disease", "node_name": "flu", "node_source": "MONDO"},
        {"node_index": 2, "node_id": "A", "node_type": "disease", "node_name": "a", "node_source": "MONDO"},
    ])
    edges = pd.DataFrame([{"x_index": 0, "y_index": 1, "relation": "indication"}])
    kg.to_csv(raw / "kg.csv", index=False)
    nodes.to_csv(raw / "node.csv", sep="\t", index=False)
    edges.to_csv(raw / "edges.csv", index=False)
    lock = {}
    for name, delimiter in (("kg.csv", ","), ("node.csv", "\t"), ("edges.csv", ",")):
        path = raw / name
        lock[name] = {"datafile_id": 1, "sha256": file_sha256(path), "bytes": path.stat().st_size,
                      "delimiter": delimiter}
    (raw / "asset_lock.json").write_text(json.dumps(lock))


class BundleTests(unittest.TestCase):
    def test_repeat_and_fresh_process(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            _write_lock(raw)
            group = pd.DataFrame({"node_id": ["10"], "group_id_bert": ["10"]})
            group_path = tmp / "groups.csv"
            group.to_csv(group_path, index=False)
            first = prepare_bundle(raw, tmp / "one", group_map_path=group_path, fold_seed=42,
                                   preprocess_seed=1, input_seed=0, input_dim=4)
            second = prepare_bundle(raw, tmp / "two", group_map_path=group_path, fold_seed=42,
                                    preprocess_seed=99, input_seed=0, input_dim=4)
            report = verify_bundle(first)
            self.assertTrue(report["fresh_process_reload"])
            one = json.loads((first / "checks.json").read_text())
            two = json.loads((second / "checks.json").read_text())
            self.assertEqual(one["graph_ordered_hash"], two["graph_ordered_hash"])
            self.assertEqual(one["holdout"]["therapeutic_train_test_overlap"], 0)
            self.assertFalse(one["preprocess_seed_consumed"])
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertNotEqual(manifest["config"]["preprocess_seed"], 99)
            self.assertEqual(file_sha256(first / "inputs" / "input_000.npy"),
                             file_sha256(second / "inputs" / "input_000.npy"))
