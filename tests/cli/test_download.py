# Copyright 2025-present, the HuggingFace Inc. team.
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
import os
import tempfile

from transformers.testing_utils import require_torch


@require_torch
def test_cli_download(cli):
    with tempfile.TemporaryDirectory() as tmpdir:
        output = cli("download", "hf-internal-testing/tiny-random-gptj", "--cache-dir", tmpdir)
        assert output.exit_code == 0

        # check if the model files are downloaded correctly
        model_dir = os.path.join(tmpdir, "models--hf-internal-testing--tiny-random-gptj")
        assert os.path.exists(os.path.join(model_dir, "blobs"))
        assert os.path.exists(os.path.join(model_dir, "refs"))
        assert os.path.exists(os.path.join(model_dir, "snapshots"))


@require_torch
def test_cli_download_trust_remote(cli, caplog, capsys):
    caplog.set_level(100000)
    # ^ hack to avoid an issue happening only in CI. We don't check logs anyway so it's fine.
    #   Source: https://github.com/pallets/click/issues/824#issuecomment-562581313

    with capsys.disabled():
        with tempfile.TemporaryDirectory() as tmpdir:
            output = cli(
                "download",
                "hf-internal-testing/test_dynamic_model_with_tokenizer",
                "--trust-remote-code",
                "--cache-dir",
                tmpdir,
            )
            assert output.exit_code == 0

            # check if the model files are downloaded correctly
            model_dir = os.path.join(tmpdir, "models--hf-internal-testing--test_dynamic_model_with_tokenizer")
            assert os.path.exists(os.path.join(model_dir, "blobs"))
            assert os.path.exists(os.path.join(model_dir, "refs"))
            assert os.path.exists(os.path.join(model_dir, "snapshots"))
