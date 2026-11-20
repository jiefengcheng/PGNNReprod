"""Small structural tests; these are not tests against the released full KG."""
import copy
import unittest
import numpy as np
import pandas as pd
from txgnn_stage1.graph import COLS, build_graph


def table(rows):
    frame = pd.DataFrame(rows, columns=COLS)
    return frame.astype({"x_idx": "int64", "y_idx": "int64"})


def fixture():
    # Bare ID "shared" belongs to both a drug and a disease. Disease d2 is
    # present in the full KG but isolated in the training graph.
    forward = [
        ("drug", "shared", 0, "indication", "disease", "shared", 0),
        ("drug", "drug1", 1, "indication", "disease", "d1", 1),
        ("drug", "shared", 0, "contraindication", "disease", "shared", 0),
        ("disease", "shared", 0, "disease_disease", "disease", "shared", 0),
    ]
    heldout = ("drug", "drug1", 1, "indication", "disease", "d2", 2)
    rows = forward + [
        ("disease", "shared", 0, "rev_indication", "drug", "shared", 0),
        ("disease", "d1", 1, "rev_indication", "drug", "drug1", 1),
        ("disease", "shared", 0, "rev_contraindication", "drug", "shared", 0),
        forward[-1],  # Homogeneous self-loop duplicate after reversal.
    ]
    full = table(forward + [heldout])
    splits = {"train": table(rows), "valid": table([heldout]), "test": table([])}
    manifest = {
        "node_types": ["drug", "disease"],
        "canonical_etypes": [
            ["drug", "indication", "disease"],
            ["disease", "rev_indication", "drug"],
            ["drug", "contraindication", "disease"],
            ["disease", "rev_contraindication", "drug"],
            ["disease", "disease_disease", "disease"],
        ],
        "reverse_relations": {
            "rev_indication": "indication",
            "rev_contraindication": "contraindication",
        },
    }
    return full, splits, manifest


class GraphTests(unittest.TestCase):
    def test_inventory_identity_parallel_edges_and_multiplicity(self):
        full, splits, manifest = fixture()
        graph = build_graph(full, splits, manifest)
        self.assertEqual(graph.num_nodes, {"drug": 2, "disease": 3})
        self.assertEqual(graph.node_ids["drug"][0], graph.node_ids["disease"][0])
        self.assertEqual(graph.node_types, ("drug", "disease"))
        self.assertEqual(graph.canonical_etypes, tuple(map(tuple, manifest["canonical_etypes"])))
        indication = graph.edges[("drug", "indication", "disease")]
        np.testing.assert_array_equal(indication, [[0, 1], [0, 1]])
        self.assertEqual(graph.edges[("drug", "contraindication", "disease")].shape[1], 1)
        self.assertEqual(graph.edges[("disease", "disease_disease", "disease")].shape[1], 2)
        self.assertFalse(indication.flags.writeable)
        self.assertFalse(any(2 in edges for edges in graph.edges.values()))

    def test_reverse_heldout_leak_rejected(self):
        full, splits, manifest = fixture()
        leak = table([("disease", "shared", 0, "rev_indication", "drug", "shared", 0)])
        splits["test"] = leak
        with self.assertRaisesRegex(ValueError, "Leakage"):
            build_graph(full, splits, manifest)

    def test_homogeneous_reverse_leak_rejected(self):
        full, splits, manifest = fixture()
        first = table([("disease", "shared", 0, "disease_disease", "disease", "d1", 1)])
        reverse = table([("disease", "d1", 1, "disease_disease", "disease", "shared", 0)])
        full = pd.concat([full, first], ignore_index=True)
        splits["train"] = pd.concat([splits["train"], first], ignore_index=True)
        splits["test"] = reverse
        with self.assertRaisesRegex(ValueError, "Leakage"):
            build_graph(full, splits, manifest)

    def test_bad_local_index_rejected(self):
        for value in [-1.0, 0.5, float("nan"), 100.0]:
            full, splits, manifest = fixture()
            splits["train"]["x_idx"] = splits["train"].x_idx.astype(float)
            splits["train"].loc[0, "x_idx"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_graph(full, splits, manifest)

    def test_registry_bijection_and_contiguity(self):
        full, splits, manifest = fixture()
        broken = full.copy()
        broken.loc[1, "y_idx"] = 0
        with self.assertRaisesRegex(ValueError, "bijection"):
            build_graph(broken, splits, manifest)
        broken = full.copy()
        broken.loc[broken.y_id == "d2", "y_idx"] = 4
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_graph(broken, splits, manifest)

    def test_ordered_and_multiset_hashes(self):
        full, splits, manifest = fixture()
        a = build_graph(full, splits, manifest)
        moved = dict(splits)
        moved["train"] = splits["train"].iloc[::-1].reset_index(drop=True)
        b = build_graph(full, moved, manifest)
        self.assertNotEqual(a.logical_hash(), b.logical_hash())
        self.assertEqual(a.logical_hash(ordered=False), b.logical_hash(ordered=False))
        moved["train"] = splits["train"].iloc[:-1]
        c = build_graph(full, moved, manifest)
        self.assertNotEqual(a.logical_hash(ordered=False), c.logical_hash(ordered=False))

    def test_relation_name_ambiguity_rejected(self):
        full, splits, manifest = fixture()
        changed = copy.deepcopy(manifest)
        changed["canonical_etypes"].append(["disease", "indication", "drug"])
        with self.assertRaisesRegex(ValueError, "Ambiguous relation"):
            build_graph(full, splits, changed)

    def test_invented_source_fact_rejected(self):
        full, splits, manifest = fixture()
        # Both endpoints and the relation are valid, but the fact is absent.
        splits["test"] = table([
            ("drug", "shared", 0, "indication", "disease", "d2", 2)])
        with self.assertRaisesRegex(ValueError, "absent from full source"):
            build_graph(full, splits, manifest)

    def test_holdout_only_relation_does_not_expand_training_schema(self):
        full, splits, manifest = fixture()
        heldout = ("drug", "shared", 0, "off-label use", "disease", "d2", 2)
        reverse = ("disease", "d2", 2, "rev_off-label use", "drug", "shared", 0)
        full = pd.concat([full, table([heldout])], ignore_index=True)
        splits["test"] = table([heldout, reverse])
        manifest["validation_canonical_etypes"] = manifest["canonical_etypes"] + [
            ["drug", "off-label use", "disease"],
            ["disease", "rev_off-label use", "drug"],
        ]
        manifest["reverse_relations"]["rev_off-label use"] = "off-label use"
        graph = build_graph(full, splits, manifest)
        self.assertEqual(graph.canonical_etypes,
                         tuple(map(tuple, manifest["canonical_etypes"])))
        self.assertNotIn(("drug", "off-label use", "disease"), graph.edges)
        self.assertNotIn(("disease", "rev_off-label use", "drug"), graph.edges)
        # A complete validation schema must not admit these into training.
        splits["train"] = pd.concat([splits["train"], table([heldout])], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "train: unknown canonical"):
            build_graph(full, splits, manifest)
