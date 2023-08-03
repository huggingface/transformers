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
import inspect
from typing import Optional

from ..utils import (
    find_adapter_config_file,
    is_accelerate_available,
    is_peft_available,
    logging,
    requires_backends,
)


if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_balanced_memory, infer_auto_device_map


logger = logging.get_logger(__name__)


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library.
    Currently supported PEFT methods are all non-prefix tuning methods
    """

    _hf_peft_config_loaded = False

    def load_adapter(
        self,
        peft_model_id: str,
        adapter_name: Optional[str] = "default",
        revision: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        commit_hash: Optional[str] = None,
        device_map: Optional[str] = "auto",
        max_memory: Optional[int] = None,
        offload_dir: Optional[str] = None,
        offload_index: Optional[int] = None,
    ) -> None:
        """
        Load adapter weights from file. Requires peft as a backend to load the adapter weights
        """
        requires_backends(self.load_adapter, "peft")

        from peft import PeftConfig, inject_adapter_in_model, load_peft_weights
        from peft.utils import set_peft_model_state_dict

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        adapter_config_file = find_adapter_config_file(
            peft_model_id,
            revision=revision,
            use_auth_token=use_auth_token,
            commit_hash=commit_hash,
        )

        if adapter_config_file is None:
            raise ValueError(
                f"adapter model file not found in {peft_model_id}. Make sure you are passing the correct path to the "
                "adapter model."
            )

        loaded_peft_config = PeftConfig.from_pretrained(
            peft_model_id,
            revision=revision,
            use_auth_token=use_auth_token,
            commit_hash=commit_hash,
        )

        # Replace the adapter with the loaded adapter
        inject_adapter_in_model(loaded_peft_config, self, adapter_name)

        adapter_state_dict = load_peft_weights(
            peft_model_id,
            revision=revision,
            use_auth_token=use_auth_token,
        )

        # We need to pre-process the state dict to remove unneeded prefixes - for backward compatibility
        processed_adapter_state_dict = {}
        for key, value in adapter_state_dict.items():
            if "base_model.model" in key:
                new_key = key.replace("base_model.model.", "")
            else:
                new_key = key
            processed_adapter_state_dict[new_key] = value

        # Load state dict
        incompatible_keys = set_peft_model_state_dict(self, processed_adapter_state_dict, adapter_name)

        if incompatible_keys is not None:
            # check only for unexpected keys
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                logger.warning(
                    f"Loading adapter weights from {peft_model_id} led to unexpected keys not found in the model: "
                    f" {incompatible_keys.unexpected_keys}. "
                )

        # @pacman100 why this was needed?
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            self._dispatch_accelerate_model(
                device_map=device_map, max_memory=max_memory, offload_dir=offload_dir, offload_index=offload_index
            )

    def add_adapter(
        self,
        adapter_config,
        adapter_name: Optional[str] = "default",
    ) -> None:
        r"""
        Adds a fresh new adapter to the current model for training purpose.
        """
        requires_backends(self.add_adapter, "peft")

        from peft import PeftConfig, inject_adapter_in_model

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        inject_adapter_in_model(adapter_config, self, adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        r"""
        Sets an adapter to switch easily between multiple adapters.
        """
        requires_backends(self.set_adapter, "peft")
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    # TODO: change it to a property but torch.jit fails. Maybe we should return None is PEFT is not available
    def active_adapter(self) -> str:
        r"""
        Gets the current active adapter of the model.
        """
        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                return module.active_adapter

    def get_adapter_state_dict(
        self,
        adapter_name: Optional[str] = None,
    ) -> dict:
        r"""
        Gets the adapter state dict.
        """
        requires_backends(self.save_adapter, "peft")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft import get_peft_model_state_dict

        if adapter_name is None:
            adapter_name = self.current_active_adapter

        adapter_state_dict = get_peft_model_state_dict(self, adapter_name=adapter_name)
        return adapter_state_dict

    def _dispatch_accelerate_model(
        self,
        device_map: str,
        max_memory: Optional[int] = None,
        offload_dir: Optional[str] = None,
        offload_index: Optional[int] = None,
    ) -> None:
        r"""
        Optionnal re-dispatch the model and attach new hooks to the model in case the model has been loaded with
        accelerate (i.e. with `device_map=xxx`)

        Args:
            device_map (`str`):
                The device map used to load the model with accelerate.
            max_memory (`int`, `optional`):
                The maximum memory argument to be passed to `accelerate.get_balanced_memory` method.
            offload_dir (`str`, `optional`):
                The offload_dir argument to be passed to `accelerate.dispatch_model` method.
            offload_index (`int`, `optional`):
                The offload_index argument to be passed to `accelerate.dispatch_model` method.
        """
        dispatch_model_kwargs = {}
        # Safety checker for previous `accelerate` versions
        # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = self._no_split_modules

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                self,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )
        dispatch_model(
            self,
            device_map=device_map,
            offload_dir=offload_dir,
            **dispatch_model_kwargs,
        )
