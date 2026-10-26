# Copyright 2026 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Backward-compatible configuration dispatch for legacy MiniCPM4 checkpoints."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import CONFIG_MAPPING


def _resolve_minicpm_config_class(config_dict: dict) -> type[PreTrainedConfig]:
    config_class = CONFIG_MAPPING["minicpm4"]
    if not config_dict or config_dict.get("model_type") == "minicpm4":
        return config_class

    remote_config_class = config_dict.get("auto_map", {}).get("AutoConfig", "").rsplit("--", 1)[-1]
    architecture_signature = tuple(
        config_dict.get(key)
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        )
    )
    if (
        remote_config_class == "configuration_minicpm.MiniCPMConfig"
        and (config_dict.get("rope_scaling") or {}).get("rope_type") == "longrope"
        and all(key in config_dict for key in ("scale_emb", "scale_depth", "dim_model_base"))
        and architecture_signature in {(1024, 4096, 24, 16, 2), (4096, 16384, 32, 32, 2)}
    ):
        return config_class

    raise ValueError(
        "This legacy MiniCPM configuration is not supported by MiniCPM4. "
        "Use the checkpoint's custom code with trust_remote_code=True if you trust its authors."
    )


# TODO: Remove this dispatcher once the official MiniCPM4 Hub configs use model_type="minicpm4".
@auto_docstring(checkpoint="openbmb/MiniCPM4-8B")
@strict
class MiniCPMConfig(PreTrainedConfig):
    r"""Compatibility dispatcher for MiniCPM4 checkpoints saved with `model_type="minicpm"`.

    Loading a supported legacy configuration returns a [`MiniCPM4Config`]. Use [`MiniCPM4Config`] directly for
    new configurations. Other MiniCPM architectures are not supported by this dispatcher.

    transformers_version (`str`, *optional*):
        Transformers version recorded in the legacy configuration.
    architectures (`list[str]`, *optional*):
        Model architecture names recorded in the legacy configuration.
    """

    model_type = "minicpm"

    def __new__(cls, *args, **kwargs):
        if cls is MiniCPMConfig:
            target_cls = _resolve_minicpm_config_class(kwargs)
            kwargs.pop("model_type", None)
            return target_cls(*args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if cls is MiniCPMConfig:
            target_cls = _resolve_minicpm_config_class(config_dict)
            return target_cls.from_dict({**config_dict, "model_type": target_cls.model_type}, **kwargs)
        return super().from_dict(config_dict, **kwargs)


__all__ = ["MiniCPMConfig"]
