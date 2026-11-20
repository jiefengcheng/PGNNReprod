"""Acquisition client checks. These do not download the released KG."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from txgnn_stage1.assets import USER_AGENT, acquire_assets


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


PAYLOADS = {
    "7144484": b"x_index,y_index,x_type,x_id,relation,y_type,y_id\n",
    "7144482": b"node_index\tnode_id\tnode_source\n",
    "7144483": b"x_index,y_index,relation\n",
}


class AssetAcquisitionTests(unittest.TestCase):
    def test_dataverse_request_sets_non_default_user_agent(self):
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request, timeout))
            file_id = request.full_url.rstrip("/").rsplit("/", 1)[-1]
            return _Response(PAYLOADS[file_id])

        with mock.patch("txgnn_stage1.assets.urllib.request.urlopen", fake_urlopen):
            raw = tempfile.TemporaryDirectory()
            try:
                records = acquire_assets(raw.name)
            finally:
                raw.cleanup()
        self.assertEqual(len(seen), 3)
        for request, timeout in seen:
            self.assertEqual(timeout, 120)
            self.assertEqual(request.get_header("User-agent"), USER_AGENT)
            self.assertNotIn("Python-urllib", request.get_header("User-agent"))
        self.assertEqual(
            [records[name]["datafile_id"] for name in ("kg.csv", "node.csv", "edges.csv")],
            [7144484, 7144482, 7144483],
        )

    def test_invalid_header_does_not_keep_a_partial_asset(self):
        def fake_urlopen(request, timeout=None):
            return _Response(b"not,the,schema\n")

        with mock.patch("txgnn_stage1.assets.urllib.request.urlopen", fake_urlopen):
            raw = tempfile.TemporaryDirectory()
            try:
                with self.assertRaises(ValueError):
                    acquire_assets(raw.name)
                self.assertEqual(list(Path(raw.name).glob("*")), [])
            finally:
                raw.cleanup()
