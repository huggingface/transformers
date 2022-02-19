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
"""
Integration with OSLO
"""

import importlib.util
import io
import json
import weakref
from copy import deepcopy

from .dependency_versions_check import dep_version_check
from .file_utils import is_torch_available
from .utils import logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def is_oslo_available():
    return importlib.util.find_spec("oslo") is not None


# keep the config object global to be able to access it anywhere during TrainingArguments life-cycle
_hf_deepspeed_config_weak_ref = None


class HfOsloConfig:
    def __init__(self, config_file_or_dict):
        # set global weakref object
        set_hf_oslo_config(self)

        # dep_version_check("oslo")

        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            config = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with io.open(config_file_or_dict, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError("expecting either a path to a Oslo config file or a pre-populated dict")
        self.config = config
        self.mpu = None


class HfTrainerOsloConfig(HfOsloConfig):
    """
    The `HfTrainerOsloConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = torch.float16

    def dtype(self):
        return self._dtype


def set_hf_oslo_config(hf_oslo_config_obj):
    global _hf_oslo_config_weak_ref
    _hf_oslo_config_weak_ref = weakref.ref(hf_oslo_config_obj)


def oslo_config():
    if _hf_oslo_config_weak_ref is not None and _hf_oslo_config_weak_ref() is not None:
        return _hf_oslo_config_weak_ref().config
    return None


def oslo_init(trainer):
    # NOTE: maybe return mpu and undo stashing altogther?
    import oslo

    config = trainer.args.hf_oslo_config.config

    trainer.model = oslo.initialize(model=trainer.model, config=config)

    # stash mpu for later use, e.g. deepspeed
    config.mpu = trainer.model.mpu

    # stash kwargs to enabled a later oslo_reinit
    trainer.oslo_initialize_kwargs = config

    return trainer.model
