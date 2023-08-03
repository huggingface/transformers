# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from contextlib import contextmanager
import importlib

from packaging import version

from typing import Optional

from .hub import cached_file


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"


def find_adapter_config_file(
    model_id: str,
    revision: str = None,
    subfolder: str = None,
    use_auth_token: Optional[str] = None,
    commit_hash: Optional[str] = None,
) -> Optional[str]:
    r"""
    Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path the the adapter
    config file if it is, None otherwise.

    Args:
        model_id (`str`):
            The identifier of the model to look for, can be either a local path or an id to the repository on the Hub.
        revision (`str`, `optional`):
            revision argument to be passed to `hf_hub_download` method from `huggingface_hub`.
        use_auth_token (`str`, `optional`):
            use_auth_token argument to be passed to `hf_hub_download` method from `huggingface_hub`.
        commit_hash (`str`, `optional`):
            commit_hash argument to be passed to `hf_hub_download` method from `huggingface_hub`.
    """
    adapter_cached_filename = None
    if os.path.isdir(model_id):
        list_remote_files = os.listdir(model_id)
        if ADAPTER_CONFIG_NAME in list_remote_files:
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    else:
        adapter_cached_filename = cached_file(
            model_id,
            ADAPTER_CONFIG_NAME,
            revision=revision,
            use_auth_token=use_auth_token,
            _commit_hash=commit_hash,
            subfolder=subfolder,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

    return adapter_cached_filename


@contextmanager
def check_peft_version(min_version: str) -> None:
    r"""
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) <= version.parse(
        min_version
    )

    if not is_peft_version_compatible:
        raise ValueError(
            f"The version of PEFT you are using is not compatible, please use a version that is greater"
            f" than {min_version}"
        )

    try:
        yield 
    finally:
        # Do nothing
        pass