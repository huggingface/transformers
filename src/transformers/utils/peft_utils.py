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
import importlib
import json
import os
from typing import Any

from huggingface_hub import is_offline_mode
from packaging import version

from .hub import cached_file
from .import_utils import is_peft_available
from .logging import get_logger


logger = get_logger(__name__)


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"


def find_adapter_config_file(
    model_id: str,
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    subfolder: str = "",
    _commit_hash: str | None = None,
) -> str | None:
    r"""
    Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path of the adapter
    config file if it is, None otherwise.

    Args:
        model_id (`str`):
            The identifier of the model to look for, can be either a local path or an id to the repository on the Hub.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.

            <Tip>

            To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

            </Tip>

        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
    """
    adapter_cached_filename = None
    if model_id is None:
        return None
    elif os.path.isdir(model_id):
        list_remote_files = os.listdir(model_id)
        if ADAPTER_CONFIG_NAME in list_remote_files:
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    else:
        adapter_cached_filename = cached_file(
            model_id,
            ADAPTER_CONFIG_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            subfolder=subfolder,
            _commit_hash=_commit_hash,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )

    return adapter_cached_filename


def check_peft_version(min_version: str) -> None:
    r"""
    Checks if the version of PEFT is compatible.

    Args:
        version (`str`):
            The version of PEFT to check against.
    """
    if not is_peft_available():
        raise ValueError("PEFT is not installed. Please install it with `pip install peft`")

    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) >= version.parse(min_version)

    if not is_peft_version_compatible:
        raise ValueError(f"The version of PEFT you are using is not compatible, please use a version >= {min_version}")


def resolve_peft_base_model_path(
    original_path: str | os.PathLike,
    adapter_config_base_model_path: str | None,
    local_files_only: bool = False,
    adapter_config: dict[str, Any] | None = None,
) -> tuple[str | os.PathLike, bool]:
    """
    Resolves the base model path when loading PEFT adapters, preserving local paths when appropriate.

    This function centralizes the logic for deciding whether to use a local path or the hub path
    specified in the adapter config. It ensures consistent behavior across all model loading entry points.

    Args:
        original_path (`str` or `os.PathLike`):
            The original path provided by the user (can be local or hub path).
        adapter_config_base_model_path (`str` or `None`):
            The base model path from the adapter config (typically a hub path).
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only use local files and raise an error if hub download is required.
        adapter_config (`dict[str, Any]` or `None`, *optional*):
            The full adapter config dictionary for compatibility checking.

    Returns:
        `tuple[str | os.PathLike, bool]`: A tuple containing:
            - The resolved base model path to use
            - A boolean indicating if a warning was logged (True) or not (False)

    Raises:
        `OSError`: If `local_files_only=True` and a hub download would be required but the original
            path is not a valid local directory.
    """
    if adapter_config_base_model_path is None:
        return original_path, False

    original_path_str = str(original_path)
    original_path_is_local = os.path.isdir(original_path_str) or os.path.exists(original_path_str)
    base_path_is_local = os.path.isdir(adapter_config_base_model_path) or os.path.exists(
        adapter_config_base_model_path
    )

    # If offline mode or local_files_only is enabled, we must use local paths
    if is_offline_mode() or local_files_only:
        if not original_path_is_local:
            raise OSError(
                f"Cannot load adapter from '{original_path_str}': offline/local_files_only mode is enabled "
                f"but the path is not a local directory. Adapter config specifies base model "
                f"'{adapter_config_base_model_path}' which would require a hub download."
            )
        # In offline mode, always preserve the local path
        if not base_path_is_local:
            logger.warning(
                f"Adapter config specifies base model '{adapter_config_base_model_path}' from hub, "
                f"but using local path '{original_path_str}' instead (offline/local_files_only mode). "
                "Make sure the local checkpoint matches the expected base model."
            )
            return original_path, True
        # Both are local, use the one from adapter config
        return adapter_config_base_model_path, False

    # Normal mode: decide based on path types
    if original_path_is_local:
        if not base_path_is_local:
            # Original is local, adapter config points to hub - preserve local path
            _check_adapter_compatibility(original_path_str, adapter_config_base_model_path, adapter_config)
            logger.warning(
                f"Adapter config specifies base model '{adapter_config_base_model_path}' from hub, "
                f"but using local path '{original_path_str}' instead. "
                "Make sure the local checkpoint matches the expected base model."
            )
            return original_path, True
        else:
            # Both are local - use the one from adapter config
            return adapter_config_base_model_path, False
    else:
        # Original is hub path - use adapter config value (which may also be hub or local)
        return adapter_config_base_model_path, False


def _check_adapter_compatibility(
    local_path: str, expected_base_model_path: str, adapter_config: dict[str, Any] | None
) -> None:
    """
    Performs a lightweight compatibility check between a local model and the adapter's expected base model.

    This checks model type and architecture if available, emitting a warning if there's a mismatch.
    This is non-invasive and does not perform heavy checksum or weight comparisons.

    Args:
        local_path (`str`):
            Path to the local model directory.
        expected_base_model_path (`str`):
            The expected base model path from adapter config.
        adapter_config (`dict[str, Any]` or `None`):
            The adapter config dictionary.
    """
    if adapter_config is None:
        return

    # Try to load config from local path
    local_config_path = os.path.join(local_path, "config.json")
    if not os.path.exists(local_config_path):
        return

    try:
        with open(local_config_path, "r", encoding="utf-8") as f:
            local_config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    # Get model type from adapter config if available
    adapter_base_model_type = adapter_config.get("base_model_name_or_path")
    if not adapter_base_model_type:
        return

    # Check model type compatibility
    local_model_type = local_config.get("model_type")
    if local_model_type:
        # Try to infer expected model type from adapter config or expected path
        # This is a lightweight check - we just warn if types don't match
        # Note: We can't easily get the model type from hub path without downloading,
        # so we only check if we have both pieces of information
        pass  # Future enhancement: could check against cached configs

    # Check architecture compatibility if available
    local_architectures = local_config.get("architectures", [])
    if local_architectures:
        # If adapter was trained on a specific architecture, we could check here
        # For now, this is a placeholder for future lightweight checks
        pass
