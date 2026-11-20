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

from unittest.mock import patch

import pytest

from transformers import __version__
from transformers.testing_utils import require_torch


def test_cli_env(cli):
    output = cli("env")
    assert output.exit_code == 0
    assert "Python version" in output.output
    assert "Platform" in output.output
    assert "Using distributed or parallel set-up in script?" in output.output


@pytest.mark.parametrize(
    "accelerate_package_info, torch_version, expected_accelerate, expected_torch",
    [
        pytest.param(
            (True, "1.0.1"),
            "2.5.0",
            "1.0.1 (not in expected version range)",
            "2.5.0 (NA) (not in expected version range)",
            id="packages_out_of_expected_version",
        ),
        pytest.param(
            (True, "1.1.0"),
            "2.5.0",
            "1.1.0",
            "2.5.0 (NA) (not in expected version range)",
            id="torch_out_of_expected_version",
        ),
        pytest.param(
            (False, "N/A"),
            "N/A",
            "not installed",
            "not installed (NA)",
            id="packages_not_installed",
        ),
    ],
)
def test_cli_env_package_versions(cli, accelerate_package_info, torch_version, expected_accelerate, expected_torch):
    with (
        patch("transformers.cli.system._is_package_available", return_value=accelerate_package_info),
        patch("transformers.cli.system.is_accelerate_available", return_value=False),
        patch("transformers.cli.system.get_torch_version", return_value=torch_version),
        patch("transformers.cli.system.is_torch_available", return_value=False),
    ):
        output = cli("env")
    assert output.exit_code == 0
    assert f"Accelerate version: {expected_accelerate}\n" in output.output
    assert f"PyTorch version (accelerator?): {expected_torch}\n" in output.output


@require_torch
def test_cli_env_accelerate_out_of_expected_version(cli):
    import torch

    with (
        patch("transformers.cli.system._is_package_available", return_value=(True, "1.0.1")),
        patch("transformers.cli.system.is_accelerate_available", return_value=False),
    ):
        output = cli("env")
    assert output.exit_code == 0
    assert "Accelerate version: 1.0.1 (not in expected version range)" in output.output
    assert f"PyTorch version (accelerator?): {torch.__version__} (" in output.output
    assert output.output.count("(not in expected version range)") == 1


def test_cli_version(cli):
    output = cli("version")
    assert output.exit_code == 0
    assert output.output.strip() == __version__
