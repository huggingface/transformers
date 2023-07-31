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
from typing import Optional

from ..utils import ADAPTER_CONFIG_NAME, cached_file, is_peft_available, logging, requires_backends


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
    ):
        """
        Load adapter weights from file. Requires peft as a backend to load the adapter weights
        """
        requires_backends(self.load_adapter, "peft")

        from peft import PeftConfig, create_and_replace, load_peft_weights
        from peft.utils import set_peft_model_state_dict
        from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

        if not self._hf_peft_config_loaded:
            self.peft_config = {}
            self._hf_peft_config_loaded = True

        adapter_config_file = self._find_adapter_config_file(
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

        if not hasattr(loaded_peft_config, "target_modules"):
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[self.config.model_type]
            loaded_peft_config.target_modules = target_modules

        if adapter_name not in self.peft_config:
            self.peft_config[adapter_name] = loaded_peft_config
        else:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        # Replace the adapter with the loaded adapter
        create_and_replace(loaded_peft_config, self, adapter_name)

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

    def set_adapter(self, adapter_name: str):
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

    @property
    def current_active_adapter(self):
        r"""
        Gets the current active adapter of the model
        """
        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                return module.active_adapter

    def _find_adapter_config_file(
        self,
        model_id: str,
        revision: str = None,
        use_auth_token: Optional[str] = None,
        commit_hash: Optional[str] = None,
    ) -> Optional[str]:
        r"""
        Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path the the
        adapter config file if it is, None otherwise.
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
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )

        return adapter_cached_filename
