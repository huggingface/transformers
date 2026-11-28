# Copyright 2026 The OpenBMB Team and the HuggingFace Inc. team. All rights reserved.
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

import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import CONFIG_MAPPING
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaAttention
from ..minicpm3.modeling_minicpm3 import (
    MiniCPM3DecoderLayer,
    MiniCPM3ForCausalLM,
    MiniCPM3ForSequenceClassification,
    MiniCPM3Model,
    MiniCPM3PreTrainedModel,
)
from ..olmo.modeling_olmo import OlmoRotaryEmbedding, apply_rotary_pos_emb  # noqa: F401


@auto_docstring(checkpoint="openbmb/MiniCPM4-8B")
@strict
class MiniCPM4Config(LlamaConfig):
    r"""
    scale_emb (`int` or `float`, *optional*, defaults to 12):
        Multiplier applied to input embeddings.
    scale_depth (`int` or `float`, *optional*, defaults to 1.4):
        Multiplier for residual connections; the effective scaling is `scale_depth / sqrt(num_hidden_layers)`.
        If `None`, defaults to `sqrt(num_hidden_layers)` (no-op scaling).
    dim_model_base (`int`, *optional*, defaults to 256):
        Base model dimension used to scale logits before the language model head. If `None`,
        defaults to `hidden_size` (no-op scaling).

    Example:

    ```python
    >>> from transformers import MiniCPM4Config, MiniCPM4Model

    >>> configuration = MiniCPM4Config()
    >>> model = MiniCPM4Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "minicpm4"

    vocab_size: int = 73448
    intermediate_size: int = 16384
    num_key_value_heads: int | None = 2
    max_position_embeddings: int = 32768
    initializer_range: float = 0.1
    tie_word_embeddings: bool = True
    scale_emb: int | float = 12
    scale_depth: int | float | None = 1.4
    dim_model_base: int | None = 256

    def __post_init__(self, **kwargs):
        if self.scale_depth is None:
            self.scale_depth = math.sqrt(self.num_hidden_layers)
        if self.dim_model_base is None:
            self.dim_model_base = self.hidden_size
        super().__post_init__(**kwargs)
        # Recompute after the parent restores saved attributes so a serialized logits_scaling cannot override this ratio.
        self.logits_scaling = self.hidden_size / self.dim_model_base


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


# TODO: Revisit this dispatcher after the official MiniCPM4 Hub configs use model_type="minicpm4":
# https://huggingface.co/openbmb/MiniCPM4-0.5B/discussions/12
# https://huggingface.co/openbmb/MiniCPM4-8B/discussions/19
# https://huggingface.co/openbmb/MiniCPM4.1-8B/discussions/5
@auto_docstring(
    checkpoint="openbmb/MiniCPM4-8B",
    custom_args=r"""
    transformers_version (`str`, *optional*):
        Transformers version recorded in the legacy configuration.
    architectures (`list[str]`, *optional*):
        Model architecture names recorded in the legacy configuration.
    """,
)
@strict
class MiniCPMConfig(PreTrainedConfig):
    r"""
    Compatibility dispatcher for MiniCPM4 checkpoints saved with `model_type="minicpm"`.

    Loading a supported legacy configuration returns a [`MiniCPM4Config`]. Use [`MiniCPM4Config`] directly for
    new configurations. Other MiniCPM architectures are not supported by this dispatcher.
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


class MiniCPM4RotaryEmbedding(OlmoRotaryEmbedding):
    pass


class MiniCPM4Attention(LlamaAttention):
    pass


class MiniCPM4DecoderLayer(MiniCPM3DecoderLayer):
    pass


class MiniCPM4PreTrainedModel(MiniCPM3PreTrainedModel):
    pass


class MiniCPM4Model(MiniCPM3Model):
    pass


class MiniCPM4ForCausalLM(MiniCPM3ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPM4ForCausalLM

        >>> model = MiniCPM4ForCausalLM.from_pretrained("openbmb/MiniCPM4-0.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4-0.5B")

        >>> prompt = "The capital of France is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> generate_ids = model.generate(**inputs, max_new_tokens=10)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        return super().forward(**super_kwargs)


class MiniCPM4ForSequenceClassification(MiniCPM3ForSequenceClassification):
    pass


__all__ = [
    "MiniCPM4Config",
    "MiniCPMConfig",
    "MiniCPM4PreTrainedModel",
    "MiniCPM4Model",
    "MiniCPM4ForCausalLM",
    "MiniCPM4ForSequenceClassification",
]
