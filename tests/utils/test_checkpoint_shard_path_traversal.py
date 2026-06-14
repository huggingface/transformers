# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.util
import json
import os
import sys
import tempfile
import unittest


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

hub = importlib.import_module("transformers.utils.hub")
utils_init = importlib.import_module("transformers.utils")

SAFE_WEIGHTS_INDEX_NAME = utils_init.SAFE_WEIGHTS_INDEX_NAME
_join_checkpoint_shard_path = hub._join_checkpoint_shard_path
get_checkpoint_shard_files = hub.get_checkpoint_shard_files


class CheckpointShardPathTraversalTests(unittest.TestCase):
    def _write_malicious_index(self, directory: str, weight_map: dict[str, str]) -> str:
        index_path = os.path.join(directory, SAFE_WEIGHTS_INDEX_NAME)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
        return index_path

    def test_rejects_parent_directory_escape_in_weight_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = self._write_malicious_index(
                tmp_dir,
                {
                    "model.layer.weight": "../../etc/passwd",
                    "model.embed.weight": "model-00001-of-00001.safetensors",
                },
            )
            with self.assertRaises(ValueError):
                get_checkpoint_shard_files(tmp_dir, index_path)

    def test_rejects_absolute_shard_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = self._write_malicious_index(tmp_dir, {"model.layer.weight": "/etc/passwd"})
            with self.assertRaises(ValueError):
                get_checkpoint_shard_files(tmp_dir, index_path)

    def test_allows_nested_shard_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = os.path.join(tmp_dir, "shards")
            os.makedirs(nested_dir)
            shard_name = os.path.join("shards", "model-00001-of-00001.safetensors")
            open(os.path.join(tmp_dir, shard_name), "wb").close()
            index_path = self._write_malicious_index(tmp_dir, {"model.layer.weight": shard_name})
            shard_files, _ = get_checkpoint_shard_files(tmp_dir, index_path)
            self.assertEqual(len(shard_files), 1)
            self.assertTrue(shard_files[0].endswith(shard_name.replace("/", os.sep)))

    def test_join_helper_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                _join_checkpoint_shard_path(tmp_dir, "", "../../etc/passwd", "model.safetensors.index.json")


if __name__ == "__main__":
    unittest.main()
