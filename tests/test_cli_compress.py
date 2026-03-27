# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import json
import os
import shutil
import unittest

from typer.testing import CliRunner

from transformers.cli.transformers import app


runner = CliRunner()


class TestCLICompress(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_compressed_model"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_compress_command(self):
        # Use a tiny model for fast testing
        result = runner.invoke(
            app, ["compress", "hf-internal-testing/tiny-random-gpt2", "--output-dir", self.output_dir]
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("SUCCESS: Selective compression blueprint saved", result.stdout)

        # Verify the file exists and has content
        config_path = os.path.join(self.output_dir, "selective_compression_config.json")
        self.assertTrue(os.path.exists(config_path))

        with open(config_path, "r") as f:
            config = json.load(f)
            self.assertIn("protected_layers", config)
            self.assertIn("compression_metadata", config)
            self.assertEqual(config["compression_metadata"]["author"], "Osman Akkawi")


if __name__ == "__main__":
    unittest.main()
