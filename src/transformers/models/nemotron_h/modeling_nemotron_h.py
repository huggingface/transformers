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
"""Backward-compat model dispatchers for ``nemotron_h``.

The original Nemotron-H model has been split into ``nemotron_h_dense`` and
``nemotron_h_sparse``. These classes are thin dispatchers that route
instantiation and ``from_pretrained`` calls to the correct implementation based
on the config.
"""

from ...utils import logging

# These imports are intentional: `nemotron_h` is a backward-compat shim that routes
# to the dense / sparse implementations via `__new__` / `from_pretrained`. The lint
# rules below catch cross-model imports that usually indicate poor modular hygiene,
# but here the cross-reference is the whole point of this file.
# trf-ignore: TRF009
from ..nemotron_h_dense.modeling_nemotron_h_dense import (  # trf-ignore: TRF009
    NemotronHDenseForCausalLM,
    NemotronHDenseModel,
    NemotronHDensePreTrainedModel,
)

# trf-ignore: TRF009
from ..nemotron_h_sparse.configuration_nemotron_h_sparse import NemotronHSparseConfig  # trf-ignore: TRF009

# trf-ignore: TRF009
from ..nemotron_h_sparse.modeling_nemotron_h_sparse import (  # trf-ignore: TRF009
    NemotronHSparseForCausalLM,
    NemotronHSparseModel,
    NemotronHSparsePreTrainedModel,
)


logger = logging.get_logger(__name__)


def _pick_target(config, dense_cls, sparse_cls):
    return sparse_cls if isinstance(config, NemotronHSparseConfig) else dense_cls


def _dispatch_from_pretrained(pretrained_model_name_or_path, dense_cls, sparse_cls, *args, **kwargs):
    from ..auto import AutoConfig

    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    target_cls = _pick_target(config, dense_cls, sparse_cls)
    return target_cls.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)


class NemotronHPreTrainedModel:
    """Dispatcher shim. Returns a :class:`NemotronHDensePreTrainedModel` or
    :class:`NemotronHSparsePreTrainedModel` based on the config."""

    def __new__(cls, config, *args, **kwargs):
        if cls is NemotronHPreTrainedModel:
            target = _pick_target(config, NemotronHDensePreTrainedModel, NemotronHSparsePreTrainedModel)
            return target(config, *args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _dispatch_from_pretrained(
            pretrained_model_name_or_path,
            NemotronHDensePreTrainedModel,
            NemotronHSparsePreTrainedModel,
            *args,
            **kwargs,
        )


class NemotronHModel:
    """Dispatcher shim. Returns a :class:`NemotronHDenseModel` or
    :class:`NemotronHSparseModel` based on the config."""

    def __new__(cls, config, *args, **kwargs):
        if cls is NemotronHModel:
            target = _pick_target(config, NemotronHDenseModel, NemotronHSparseModel)
            return target(config, *args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _dispatch_from_pretrained(
            pretrained_model_name_or_path,
            NemotronHDenseModel,
            NemotronHSparseModel,
            *args,
            **kwargs,
        )


class NemotronHForCausalLM:
    """Dispatcher shim. Returns a :class:`NemotronHDenseForCausalLM` or
    :class:`NemotronHSparseForCausalLM` based on the config."""

    def __new__(cls, config, *args, **kwargs):
        if cls is NemotronHForCausalLM:
            target = _pick_target(config, NemotronHDenseForCausalLM, NemotronHSparseForCausalLM)
            return target(config, *args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _dispatch_from_pretrained(
            pretrained_model_name_or_path,
            NemotronHDenseForCausalLM,
            NemotronHSparseForCausalLM,
            *args,
            **kwargs,
        )


__all__ = [
    "NemotronHPreTrainedModel",
    "NemotronHModel",
    "NemotronHForCausalLM",
]
