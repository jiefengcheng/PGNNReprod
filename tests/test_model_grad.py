"""FP32 forward, analytical gradients, and one SGD step on a two-node fixture."""
import unittest
import numpy as np
from txgnn_stage1.model import encode, rgcn_layer
from txgnn_stage1.model_grad import (
    distmult_dtype, encode_dtype, one_relation_loss, rgcn_layer_dtype, sgd_step,
)


def fixture():
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    endpoints = np.array([[0, 1], [1, 0]], dtype=np.int64)
    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    bias = np.array([0.1, -0.2], dtype=np.float64)
    relation = np.array([1.0, 1.0], dtype=np.float64)
    return features, endpoints, weight, bias, relation


class GradientProofTests(unittest.TestCase):
    def test_float32_forward_matches_float64_reference(self):
        features, endpoints, weight, bias, relation = fixture()
        edges = {("node", "rel", "node"): endpoints}
        etypes = [("node", "rel", "node")]
        reference = rgcn_layer(
            {"node": features}, edges, etypes, {"rel": weight}, {"rel": bias})
        narrow = rgcn_layer_dtype(
            {"node": features}, edges, etypes, {"rel": weight}, {"rel": bias}, np.float32)
        np.testing.assert_allclose(narrow["node"], reference["node"], rtol=1e-5, atol=1e-5)
        self.assertEqual(narrow["node"].dtype, np.float32)
        layer = {"weights": {"rel": weight}, "biases": {"rel": bias}}
        encoded = encode_dtype({"node": features}, edges, etypes, (layer, layer), np.float32)
        reference_encoded = encode({"node": features}, edges, etypes, (layer, layer))
        np.testing.assert_allclose(encoded["node"], reference_encoded["node"], rtol=1e-5, atol=1e-5)
        head = reference["node"][0]
        tail = reference["node"][1]
        score32 = distmult_dtype(head, relation, tail, np.float32)
        score64 = distmult_dtype(head, relation, tail, np.float64)
        np.testing.assert_allclose(score32, score64, rtol=1e-5, atol=1e-5)

    def test_analytical_gradients_match_finite_differences(self):
        values = fixture()
        base = one_relation_loss(*values, scored_edge=(0, 1), target=0.0)
        step = 1e-6
        names = ("source", "weight", "bias", "relation")
        positions = (0, 2, 3, 4)
        for name, position in zip(names, positions):
            parameter = values[position]
            gradient = base[name]
            numerical = np.zeros_like(parameter)
            it = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                index = it.multi_index
                plus = [array.copy() for array in values]
                minus = [array.copy() for array in values]
                plus[position][index] += step
                minus[position][index] -= step
                high = one_relation_loss(*plus, scored_edge=(0, 1), target=0.0)["loss"]
                low = one_relation_loss(*minus, scored_edge=(0, 1), target=0.0)["loss"]
                numerical[index] = (high - low) / (2.0 * step)
                it.iternext()
            np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6, err_msg=name)

    def test_one_sgd_step_decreases_loss(self):
        features, endpoints, weight, bias, relation = fixture()
        before = one_relation_loss(
            features, endpoints, weight, bias, relation, scored_edge=(0, 1), target=0.0)
        rate = 0.05
        after = one_relation_loss(
            sgd_step(features, before["source"], rate),
            endpoints,
            sgd_step(weight, before["weight"], rate),
            sgd_step(bias, before["bias"], rate),
            sgd_step(relation, before["relation"], rate),
            scored_edge=(0, 1),
            target=0.0,
        )
        self.assertGreater(before["loss"], 0.0)
        self.assertLess(after["loss"], before["loss"])
        self.assertTrue(np.isfinite(after["loss"]))
