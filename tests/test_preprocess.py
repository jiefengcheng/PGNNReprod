"""Hand-computed fixtures for the owned raw-to-graph protocol."""
import unittest
import numpy as np
import pandas as pd
from txgnn_stage1.graph import build_graph
from txgnn_stage1.inputs import make_numpy_inputs, xavier_uniform_bound
from txgnn_stage1.preprocess import (
    convert2str, prepare_tables, representative_edges,
)


def kg(rows):
    return pd.DataFrame(rows, columns=["x_index", "x_type", "x_id", "x_name", "x_source",
                                       "relation", "y_index", "y_type", "y_id", "y_name", "y_source"])


class PreprocessTests(unittest.TestCase):
    def test_convert2str_and_lexicographic_indices(self):
        self.assertEqual(convert2str("00123"), "123.0")
        self.assertEqual(convert2str("MONDO_0001"), "MONDO_0001")
        self.assertEqual(convert2str("DB0001"), "DB0001")
        frame = kg([
            (0, "drug", "10", "ten", "DrugBank", "drug_drug", 1, "drug", "2", "two", "DrugBank"),
            (2, "drug", "001", "one", "DrugBank", "drug_drug", 2, "drug", "001", "one", "DrugBank"),
        ])
        prepared = prepare_tables(frame, fold_seed=42)
        drugs = prepared["nodes"][prepared["nodes"].node_type == "drug"]
        self.assertEqual(list(drugs.reference_id), ["1.0", "10.0", "2.0"])
        self.assertEqual(list(drugs.local_idx), [0, 1, 2])

    def test_normalization_collision_is_audited_and_merged(self):
        frame = kg([
            (0, "gene/protein", "001", "a", "NCBI", "protein_protein", 1, "gene/protein", "1", "b", "NCBI"),
        ])
        prepared = prepare_tables(frame, fold_seed=1)
        self.assertEqual(prepared["checks"]["normalization_collision_groups"], 1)
        self.assertEqual(prepared["checks"]["node_counts"], {"gene/protein": 1})

    def test_first_source_type_orientation(self):
        frame = kg([
            (0, "drug", "DB1", "d", "DrugBank", "indication", 1, "disease", "10", "flu", "MONDO"),
            (1, "disease", "10", "flu", "MONDO", "indication", 0, "drug", "DB1", "d", "DrugBank"),
            (2, "disease", "11", "cold", "MONDO", "indication", 0, "drug", "DB1", "d", "DrugBank"),
        ])
        kept, rules = representative_edges(frame)
        self.assertEqual(rules[0]["anchor_type"], "drug")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept.iloc[0].x_type, "drug")
        self.assertEqual(str(kept.iloc[0].y_id), "10")

    def test_homogeneous_dedup_and_reverse_self_loop_multiplicity(self):
        frame = kg([
            (0, "disease", "A", "a", "MONDO", "disease_disease", 0, "disease", "A", "a", "MONDO"),
            (0, "disease", "A", "a", "MONDO", "disease_disease", 0, "disease", "A", "a", "MONDO"),
            (0, "disease", "A", "a", "MONDO", "disease_disease", 1, "disease", "B", "b", "MONDO"),
            (1, "disease", "B", "b", "MONDO", "disease_disease", 0, "disease", "A", "a", "MONDO"),
        ])
        prepared = prepare_tables(frame, fold_seed=3)
        graph = build_graph(prepared["base"], prepared["splits"], prepared["schema"])
        edges = graph.edges[("disease", "disease_disease", "disease")]
        # One undirected A-B pair becomes two directed rows, and the self-loop is duplicated.
        self.assertEqual(edges.shape[1], 4)
        self.assertEqual(graph.num_nodes["disease"], 2)
        loops = edges[:, edges[0] == edges[1]]
        self.assertEqual(loops.shape[1], 2)

    def test_therapeutic_holdout_and_reverse_stay_out_of_train(self):
        rows = []
        for index in range(4):
            rows.append((index, "drug", "DB1", "d", "DrugBank", "indication",
                         100 + index, "disease", str(index), "n", "MONDO"))
        rows.append((9, "gene/protein", "G", "g", "NCBI", "protein_protein",
                     9, "gene/protein", "G", "g", "NCBI"))
        prepared = prepare_tables(kg(rows), fold_seed=42)
        graph = build_graph(prepared["base"], prepared["splits"], prepared["schema"])
        train = prepared["splits"]["train"]
        test = prepared["splits"]["test"]
        test_diseases = set(test.loc[test.relation == "indication", "y_id"])
        self.assertTrue(test_diseases)
        self.assertTrue(set(train.loc[train.relation == "indication", "y_id"]).isdisjoint(test_diseases))
        self.assertTrue(set(train.loc[train.relation == "rev_indication", "x_id"]).isdisjoint(test_diseases))
        # The held-out disease still has a local index and no training indication edge.
        held = test.loc[test.relation == "indication", "y_idx"].iloc[0]
        indication = graph.edges.get(("drug", "indication", "disease"))
        if indication is not None and indication.shape[1]:
            self.assertNotIn(int(held), set(indication[1].tolist()))
        self.assertGreaterEqual(graph.num_nodes["disease"], 4)

    def test_duplicate_directed_facts_share_one_split(self):
        frame = kg([
            (0, "drug", "DB1", "d", "DrugBank", "drug_protein", 1, "gene/protein", "10", "g", "NCBI"),
            (0, "drug", "DB1", "d", "DrugBank", "drug_protein", 1, "gene/protein", "10", "g", "NCBI"),
        ])
        prepared = prepare_tables(frame, fold_seed=7)
        locations = [name for name, split in prepared["splits"].items()
                     if split.relation.isin(["drug_protein", "rev_drug_protein"]).any()]
        self.assertEqual(locations, ["train"])
        train = prepared["splits"]["train"]
        self.assertEqual(int((train.relation == "drug_protein").sum()), 2)
        self.assertEqual(int((train.relation == "rev_drug_protein").sum()), 2)

    def test_grouped_disease_map_resolves_local_index(self):
        frame = kg([
            (0, "drug", "DB1", "d", "DrugBank", "indication", 5, "disease", "20", "group", "MONDO_grouped"),
        ])
        group_map = pd.DataFrame({"node_id": ["10", "99"], "group_id_bert": ["20", "404"]})
        prepared = prepare_tables(frame, group_map=group_map, fold_seed=1)
        resolved = prepared["membership"].dropna(subset=["group_local_idx"])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(int(resolved.iloc[0].group_local_idx), 0)
        self.assertEqual(prepared["checks"]["group_map"]["unresolved_group_ids"], 1)

    def test_numpy_inputs_are_deterministic_float32(self):
        counts = {"disease": 3, "drug": 2}
        order = ("disease", "drug")
        first = make_numpy_inputs(counts, order, 4, 0)
        second = make_numpy_inputs(counts, order, 4, 0)
        bound = xavier_uniform_bound(3, 4)
        self.assertTrue(np.array_equal(first["disease"], second["disease"]))
        self.assertEqual(first["disease"].dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(first["disease"]))), bound + 1e-6)
        self.assertFalse(np.array_equal(first["disease"][:2], first["drug"]))
