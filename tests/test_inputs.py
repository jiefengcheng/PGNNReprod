import unittest
try:
    import torch
except ImportError:
    torch = None
from txgnn_stage1.inputs import make_owned_inputs

@unittest.skipIf(torch is None, "PyTorch not installed in minimal source-only CI")
class InputTests(unittest.TestCase):
    def test_reproducible_frozen_xavier_matrices(self):
        types = ("drug", "disease")
        counts = {"drug": 3, "disease": 5}
        a, before, after = make_owned_inputs(counts, types, 4, 0)
        b, before2, after2 = make_owned_inputs(counts, types, 4, 0)
        self.assertTrue(torch.equal(before, before2))
        self.assertTrue(torch.equal(after, after2))
        for t in types:
            self.assertTrue(torch.equal(a[t], b[t]))
            self.assertEqual(a[t].shape, (counts[t], 4))
            self.assertFalse(a[t].requires_grad)
