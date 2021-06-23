# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import platform
from argparse import ArgumentParser

from .. import __version__ as version
from ..file_utils import is_flax_available, is_tf_available, is_torch_available
from . import BaseTransformersCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        tf_version = "not installed"
        tf_cuda_available = "NA"
        if is_tf_available():
            import tensorflow as tf

            tf_version = tf.__version__
            try:
                # deprecated in v2.1
                tf_cuda_available = tf.test.is_gpu_available()
            except AttributeError:
                # returns list of devices, convert to bool
                tf_cuda_available = bool(tf.config.list_physical_devices("GPU"))

        flax_version = "not installed"
        jax_version = "not installed"
        jaxlib_version = "not installed"
        jax_backend = "NA"
        if is_flax_available():
            import flax
            import jax
            import jaxlib

            flax_version = flax.__version__
            jax_version = jax.__version__
            jaxlib_version = jaxlib.__version__
            jax_backend = jax.lib.xla_bridge.get_backend().platform

        info = {
            "`transformers` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
            "Tensorflow version (GPU?)": f"{tf_version} ({tf_cuda_available})",
            "Flax version (CPU?/GPU?/TPU?)": f"{flax_version} ({jax_backend})",
            "Jax version": f"{jax_version}",
            "JaxLib version": f"{jaxlib_version}",
            "Using GPU in script?": "<fill in>",
            "Using distributed or parallel set-up in script?": "<fill in>",
        }

        print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
