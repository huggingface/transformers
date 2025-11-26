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
"""Contains commands to print information about the environment and version.

Usage:
    transformers env
    transformers version
"""

import contextlib
import io
import os
import platform
from typing import Annotated

import huggingface_hub
import typer

from .. import __version__
from ..integrations.deepspeed import is_deepspeed_available
from ..utils import (
    is_accelerate_available,
    is_torch_available,
    is_torch_hpu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)


def env(
    accelerate_config_file: Annotated[
        str | None,
        typer.Argument(help="The accelerate config file to use for the default values in the launching script."),
    ] = None,
) -> None:
    """Print information about the environment."""
    import safetensors

    safetensors_version = safetensors.__version__

    accelerate_version = "not installed"
    accelerate_config = accelerate_config_str = "not found"

    if is_accelerate_available():
        import accelerate
        from accelerate.commands.config import default_config_file, load_config_from_file

        accelerate_version = accelerate.__version__
        # Get the default from the config file.
        if accelerate_config_file is not None or os.path.isfile(default_config_file):
            accelerate_config = load_config_from_file(accelerate_config_file).to_dict()

        accelerate_config_str = (
            "\n".join([f"\t- {prop}: {val}" for prop, val in accelerate_config.items()])
            if isinstance(accelerate_config, dict)
            else f"\t{accelerate_config}"
        )

    pt_version = "not installed"
    pt_cuda_available = "NA"
    pt_accelerator = "NA"
    if is_torch_available():
        import torch

        pt_version = torch.__version__
        pt_cuda_available = torch.cuda.is_available()
        pt_xpu_available = is_torch_xpu_available()
        pt_npu_available = is_torch_npu_available()
        pt_hpu_available = is_torch_hpu_available()

        if pt_cuda_available:
            pt_accelerator = "CUDA"
        elif pt_xpu_available:
            pt_accelerator = "XPU"
        elif pt_npu_available:
            pt_accelerator = "NPU"
        elif pt_hpu_available:
            pt_accelerator = "HPU"

    deepspeed_version = "not installed"
    if is_deepspeed_available():
        # Redirect command line output to silence deepspeed import output.
        with contextlib.redirect_stdout(io.StringIO()):
            import deepspeed
        deepspeed_version = deepspeed.__version__

    info = {
        "`transformers` version": __version__,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Huggingface_hub version": huggingface_hub.__version__,
        "Safetensors version": f"{safetensors_version}",
        "Accelerate version": f"{accelerate_version}",
        "Accelerate config": f"{accelerate_config_str}",
        "DeepSpeed version": f"{deepspeed_version}",
        "PyTorch version (accelerator?)": f"{pt_version} ({pt_accelerator})",
        "Using distributed or parallel set-up in script?": "<fill in>",
    }
    if is_torch_available():
        if pt_cuda_available:
            info["Using GPU in script?"] = "<fill in>"
            info["GPU type"] = torch.cuda.get_device_name()
        elif pt_xpu_available:
            info["Using XPU in script?"] = "<fill in>"
            info["XPU type"] = torch.xpu.get_device_name()
        elif pt_hpu_available:
            info["Using HPU in script?"] = "<fill in>"
            info["HPU type"] = torch.hpu.get_device_name()
        elif pt_npu_available:
            info["Using NPU in script?"] = "<fill in>"
            info["NPU type"] = torch.npu.get_device_name()
            info["CANN version"] = torch.version.cann

    print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
    print(_format_dict(info))

    return info


def version() -> None:
    """Print CLI version."""
    print(__version__)


def _format_dict(d: dict) -> str:
    return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
