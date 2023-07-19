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

from ..utils import requires_backends, is_peft_available, ADAPTER_CONFIG_NAME, cached_file

if is_peft_available():
    from peft import PeftModel


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library.
    Currently supported PEFT methods are all non-prefix tuning methods
    """
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