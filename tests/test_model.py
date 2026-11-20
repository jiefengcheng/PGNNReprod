"""Hand-computed RGCN and DistMult checks. No parameters are learned here."""
import unittest
import numpy as np
from txgnn_stage1.model import (
    LEAKY_RELU_SLOPE, distmult, encode, leaky_relu, pretrain_probability, rgcn_layer,
    score_edges,
)


class ModelTests(unittest.TestCase):
    def test_mean_within_relation_and_sum_across_relations(self):
        features = {
            "drug": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "disease": np.array([[5.0, 6.0]]),
        }
        edges = {
            ("drug", "indication", "disease"): np.array([[0, 1], [0, 0]]),
            ("disease", "rev_indication", "drug"): np.array([[0], [0]]),
        }
        etypes = list(edges)
        weights = {
            "indication": np.array([[1.0, 0.0]]),
            "rev_indication": np.array([[0.0, 1.0]]),
        }
        biases = {
            "indication": np.array([0.5]),
            "rev_indication": np.array([0.0]),
        }
        hidden = rgcn_layer(features, edges, etypes, weights, biases)
        np.testing.assert_allclose(hidden["disease"], [[2.5]])
        np.testing.assert_allclose(hidden["drug"], [[6.0], [0.0]])

    def test_isolated_node_remains_zero_and_shape_is_preserved(self):
        features = {"drug": np.ones((2, 2)), "disease": np.ones((1, 2))}
        edges = {("drug", "indication", "disease"): np.zeros((2, 0), dtype=np.int64)}
        weights = {"indication": np.ones((3, 2))}
        biases = {"indication": np.zeros(3)}
        hidden = rgcn_layer(features, edges, list(edges), weights, biases)
        self.assertEqual(hidden["drug"].shape, (2, 3))
        self.assertEqual(hidden["disease"].shape, (1, 3))
        self.assertTrue(np.all(hidden["drug"] == 0.0))
        self.assertTrue(np.all(hidden["disease"] == 0.0))

    def test_two_layers_apply_leaky_relu_only_once(self):
        features = {"drug": np.array([[0.0]]), "disease": np.array([[0.0]])}
        edges = {
            ("drug", "indication", "disease"): np.array([[0], [0]]),
            ("disease", "rev_indication", "drug"): np.array([[0], [0]]),
        }
        layer1 = {
            "weights": {"indication": np.array([[0.0]]), "rev_indication": np.array([[0.0]])},
            "biases": {"indication": np.array([-2.0]), "rev_indication": np.array([0.0])},
        }
        layer2 = {
            "weights": {"indication": np.array([[0.0]]), "rev_indication": np.array([[3.0]])},
            "biases": {"indication": np.array([0.0]), "rev_indication": np.array([1.0])},
        }
        encoded = encode(features, edges, list(edges), (layer1, layer2))
        expected = -2.0 * LEAKY_RELU_SLOPE * 3.0 + 1.0
        np.testing.assert_allclose(encoded["drug"], [[expected]])
        np.testing.assert_allclose(encoded["disease"], [[0.0]])
        np.testing.assert_allclose(leaky_relu(np.array([-2.0])), [-0.02])

    def test_distmult_and_pretrain_sigmoid(self):
        score = distmult(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]))
        self.assertEqual(score, 63.0)
        embeddings = {
            "drug": np.array([[1.0, 2.0]]),
            "disease": np.array([[5.0, 6.0]]),
        }
        endpoints = np.array([[0], [0]])
        scored = score_edges(embeddings, np.array([[3.0, 4.0]]),
                             ("drug", "indication", "disease"), endpoints, 0)
        np.testing.assert_allclose(scored, [63.0])
        np.testing.assert_allclose(pretrain_probability(np.array([0.0])), [0.5])
