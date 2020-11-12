# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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
from abc import ABC, abstractmethod
from pickle import UnpicklingError
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes
from flax.traverse_util import unflatten_dict
from jax.random import PRNGKey

from .configuration_utils import PretrainedConfig
from .file_utils import WEIGHTS_NAME, cached_path, hf_bucket_url, is_remote_url
from .utils import logging


logger = logging.get_logger(__name__)


@jax.jit
def gelu(x):
    r"""
    Gaussian error linear unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
        \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    We explicitly use the approximation rather than the exact formulation for speed. For more information, see
    `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_, section 2.
    """
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": gelu,
}


class FlaxPreTrainedModel(ABC):
    config_class = None
    pretrained_model_archive_map = {}
    base_model_prefix = ""
    model_class = None

    def __init__(self, config: PretrainedConfig, module: nn.Module, params: Dict, seed: int = 0):
        if config is None:
            raise ValueError("config cannot be None")

        if module is None:
            raise ValueError("module cannot be None")

        if params is None:
            raise ValueError("state cannot be None")

        # Those are private to be exposed as typed property on derived classes.
        self._config = config
        self._module = module

        # Those are public as their type is generic to every derived classes.
        self.key = PRNGKey(seed)
        self.params = params
        self.model = module

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @staticmethod
    @abstractmethod
    def convert_from_pytorch(pt_state: Dict, config: PretrainedConfig) -> Dict:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiate a pretrained Flax model from a pre-trained model configuration.
        """
        config = kwargs.pop("config", None)
        # state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        # from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        # output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            else:
                archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=WEIGHTS_NAME, revision=revision)

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # Instantiate model.
        with open(resolved_archive_file, "rb") as state_f:
            try:
                from flax.serialization import from_bytes

                state = from_bytes(cls.model_class, state_f)
            except TypeError:
                try:
                    import torch

                    state = torch.load(state_f)
                    state = {k: v.numpy() for k, v in state.items()}
                    state = cls.convert_from_pytorch(state, config)
                    state = unflatten_dict({tuple(k.split(".")[1:]): v for k, v in state.items()})
                except UnpicklingError:
                    raise EnvironmentError(
                        f"Unable to convert model {archive_file} to Flax deserializable object. "
                        "Supported format are PyTorch archive or Flax msgpack"
                    )

        return cls(config, state, *model_args, **model_kwargs)

    def save_pretrained(self, folder):
        folder_abs = os.path.abspath(folder)

        if not os.path.exists(folder_abs):
            os.mkdir(folder_abs)

        with open(os.path.join(folder_abs, f"{self._config.model_type}.flax", "wb")) as f:
            model_bytes = to_bytes(self.params)
            f.write(model_bytes)
