"""Small autoimmune-area fixtures. These do not read the released KG."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from txgnn_stage1.area import (
    descendant_ids, obo_children, prepare_autoimmune, two_hop_induced_mask,
)
from txgnn_stage1.graph import build_graph


OBO = """
[Term]
id: DOID:417
name: autoimmune

[Term]
id: DOID:0000001
name: child
is_a: DOID:417 ! autoimmune

[Term]
id: DOID:2
name: other
is_a: DOID:9 ! elsewhere
"""


def write_obo(directory):
    path = Path(directory) / "tiny.obo"
    path.write_text(OBO)
    return path


class AreaTests(unittest.TestCase):
    def test_doid_closure_normalizes_leading_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            children = obo_children(write_obo(tmp))
        self.assertEqual(descendant_ids(children, "417"), {"417", "1"})

    def test_two_hop_is_directed_and_induced(self):
        x = np.array([1, 3, 4, 8], dtype=np.int64)
        y = np.array([3, 4, 6, 1], dtype=np.int64)
        induced, reached = two_hop_induced_mask(x, y, np.array([1], dtype=np.int64), 2)
        self.assertEqual(reached, 3)
        self.assertEqual(induced.tolist(), [True, True, False, False])

    def test_masked_therapeutic_fact_stays_out_of_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            obo = write_obo(tmp)
            kg = pd.DataFrame([
                {"x_index": 0, "x_id": "DB1", "x_type": "drug", "x_name": "d", "x_source": "DrugBank",
                 "relation": "indication", "y_index": 1, "y_id": "MONDO:1", "y_type": "disease",
                 "y_name": "a", "y_source": "MONDO"},
                {"x_index": 1, "x_id": "MONDO:1", "x_type": "disease", "x_name": "a", "x_source": "MONDO",
                 "relation": "indication", "y_index": 0, "y_id": "DB1", "y_type": "drug",
                 "y_name": "d", "y_source": "DrugBank"},
                {"x_index": 1, "x_id": "MONDO:1", "x_type": "disease", "x_name": "a", "x_source": "MONDO",
                 "relation": "disease_disease", "y_index": 3, "y_id": "MONDO:9", "y_type": "disease",
                 "y_name": "q", "y_source": "MONDO"},
                {"x_index": 3, "x_id": "MONDO:9", "x_type": "disease", "x_name": "q", "x_source": "MONDO",
                 "relation": "disease_disease", "y_index": 4, "y_id": "MONDO:8", "y_type": "disease",
                 "y_name": "r", "y_source": "MONDO"},
                {"x_index": 7, "x_id": "DB2", "x_type": "drug", "x_name": "e", "x_source": "DrugBank",
                 "relation": "indication", "y_index": 6, "y_id": "MONDO:6", "y_type": "disease",
                 "y_name": "z", "y_source": "MONDO"},
            ])
            edges = kg[["x_index", "y_index", "relation"]]
            nodes = pd.DataFrame([
                {"node_index": 1, "node_id": "MONDO:1", "node_type": "disease", "node_name": "a", "node_source": "MONDO"},
                {"node_index": 5, "node_id": "GRP", "node_type": "disease", "node_name": "g", "node_source": "MONDO_grouped"},
                {"node_index": 6, "node_id": "MONDO:6", "node_type": "disease", "node_name": "z", "node_source": "MONDO"},
            ])
            xref = pd.DataFrame({"ontology": ["DOID"], "ontology_id": ["1"], "mondo_id": ["MONDO:1"]})
            groups = pd.DataFrame({"node_id": ["MONDO:1"], "node_source": ["MONDO"], "group_id_bert": ["GRP"]})
            prepared = prepare_autoimmune(
                kg, edges, nodes, xref, groups, obo, ["MONDO:1"],
                fold_seed=42, preprocess_seed=1, hops=2, test_size=1.0)
            graph = build_graph(prepared["base"], prepared["splits"], prepared["schema"])
            self.assertIn("disease", graph.num_nodes)
            train = prepared["splits"]["train"]
            test = prepared["splits"]["test"]
            held = (test.relation == "indication") & (test.y_id == "MONDO:1")
            held_reverse = (test.relation == "rev_indication") & (test.x_id == "MONDO:1")
            self.assertTrue(bool(held.any() or held_reverse.any()))
            self.assertFalse(bool(((train.relation == "indication") & (train.y_id == "MONDO:1")).any()))
            self.assertFalse(bool(((train.relation == "rev_indication") & (train.x_id == "MONDO:1")).any()))
            self.assertGreater(prepared["checks"]["evaluation"]["kept"], 0)
            self.assertEqual(prepared["checks"]["seeds"]["grouped_nodes"], 1)
            # A one-row neighborhood cannot depend on an unseeded global RNG.
            again = prepare_autoimmune(
                kg, edges, nodes, xref, groups, obo, ["MONDO:1"],
                fold_seed=42, preprocess_seed=1, hops=2, test_size=1.0)
            self.assertTrue(np.array_equal(prepared["mask_pairs"], again["mask_pairs"]))
