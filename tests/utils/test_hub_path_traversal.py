import json
import os
import tempfile
import unittest

from transformers.utils.hub import get_checkpoint_shard_files


class TestPathTraversalProtection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_file = os.path.join(self.tmpdir, "model.safetensors.index.json")

    def _write_index(self, weight_map):
        with open(self.index_file, "w") as f:
            json.dump({"metadata": {"total_size": 1000}, "weight_map": weight_map}, f)

    def test_traversal_with_dotdot_raises(self):
        """Shard filenames containing '..' must be rejected."""
        self._write_index({"layer.weight": "../../etc/passwd"})
        with self.assertRaises(ValueError) as ctx:
            get_checkpoint_shard_files(self.tmpdir, self.index_file)
        self.assertIn("Invalid shard filename", str(ctx.exception))

    def test_absolute_path_raises(self):
        """Shard filenames starting with '/' must be rejected."""
        self._write_index({"layer.weight": "/etc/passwd"})
        with self.assertRaises(ValueError) as ctx:
            get_checkpoint_shard_files(self.tmpdir, self.index_file)
        self.assertIn("Invalid shard filename", str(ctx.exception))

    def test_symlink_escapes_directory(self):
        """Symlinks that escape the model directory must be caught by realpath check."""
        # Create a valid shard file
        shard_path = os.path.join(self.tmpdir, "model-00001-of-00001.safetensors")
        with open(shard_path, "w") as f:
            f.write("")

        # Create a symlink with a valid-looking name but pointing outside
        symlink_path = os.path.join(self.tmpdir, "model-00002-of-00002.safetensors")
        os.symlink("/etc/passwd", symlink_path)

        self._write_index({"layer.weight": "model-00002-of-00002.safetensors"})
        with self.assertRaises(ValueError) as ctx:
            get_checkpoint_shard_files(self.tmpdir, self.index_file)
        self.assertIn("resolves outside", str(ctx.exception))

    def test_valid_local_shards_pass(self):
        """Normal shard filenames in the same directory should work."""
        shard_path = os.path.join(self.tmpdir, "model-00001-of-00001.safetensors")
        with open(shard_path, "w") as f:
            f.write("")

        self._write_index({"layer.weight": "model-00001-of-00001.safetensors"})
        filenames, _ = get_checkpoint_shard_files(self.tmpdir, self.index_file)
        self.assertEqual(len(filenames), 1)
        self.assertTrue(os.path.samefile(filenames[0], shard_path))


if __name__ == "__main__":
    unittest.main()
