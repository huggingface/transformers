# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...models.qwen2.configuration_qwen2 import Qwen2Config
from ...models.qwen3.configuration_qwen3 import Qwen3Config
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/LocateAnything-3B")
@strict
class MoonViTConfig(PreTrainedConfig):
    r"""
    Args:
        patch_size (`int`, *optional*, defaults to 14):
            Patch size used by the MoonViT vision encoder.
        init_pos_emb_height (`int`, *optional*, defaults to 64):
            Initial height of the learned positional embedding grid.
        init_pos_emb_width (`int`, *optional*, defaults to 64):
            Initial width of the learned positional embedding grid.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the vision encoder.
        num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of hidden layers in the vision encoder.
        hidden_size (`int`, *optional*, defaults to 1152):
            Hidden size of the vision encoder.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Intermediate size of the vision encoder MLP layers.
        merge_kernel_size (`list[int]`, *optional*, defaults to `[2, 2]`):
            Spatial merge kernel size used to merge vision patches before projection.
    """

    model_type = "moonvit"

    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    merge_kernel_size: list[int] | tuple[int, int] = (2, 2)


@auto_docstring(checkpoint="nvidia/LocateAnything-3B")
@strict
class LocateAnythingConfig(PreTrainedConfig):
    r"""
    Args:
        vision_config (`dict` or [`MoonViTConfig`], *optional*):
            Vision encoder configuration.
        text_config (`dict` or [`~Qwen2Config`], *optional*):
            Text decoder configuration.
        use_backbone_lora (`int`, *optional*, defaults to 0):
            LoRA rank for optional vision backbone adapters. A value of 0 disables adapters.
        use_llm_lora (`int`, *optional*, defaults to 0):
            LoRA rank for optional language model adapters. A value of 0 disables adapters.
        downsample_ratio (`float`, *optional*, defaults to 0.5):
            Downsampling ratio kept for compatibility with the original configuration.
        template (`str`, *optional*):
            Prompt template name kept for compatibility with original checkpoints.
        loss_version (`str`, *optional*, defaults to `"v1"`):
            Training loss version kept for checkpoint compatibility.
        mlp_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether the projector MLP used checkpointing during training.
        image_token_index (`int`, *optional*, defaults to 151667):
            Token id used as the image placeholder in the text sequence.
        box_start_token_id (`int`, *optional*, defaults to 151668):
            Token id marking the start of a generated box.
        box_end_token_id (`int`, *optional*, defaults to 151669):
            Token id marking the end of a generated box.
        coord_start_token_id (`int`, *optional*, defaults to 151677):
            First token id in the coordinate-token range.
        coord_end_token_id (`int`, *optional*, defaults to 152677):
            Last token id in the coordinate-token range.
        ref_start_token_id (`int`, *optional*, defaults to 151672):
            Token id marking the start of a generated referring expression.
        ref_end_token_id (`int`, *optional*, defaults to 151673):
            Token id marking the end of a generated referring expression.
        none_token_id (`int`, *optional*, defaults to 4064):
            Token id used by the original generation helper for empty coordinates.
    """

    model_type = "locateanything"
    is_composition = True
    sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    use_backbone_lora: int = 0
    use_llm_lora: int = 0
    downsample_ratio: float = 0.5
    template: str | None = None
    loss_version: str = "v1"
    mlp_checkpoint: bool = False
    image_token_index: int = 151667
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    none_token_id: int = 4064

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = MoonViTConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "moonvit")
            if self.vision_config["model_type"] != "moonvit":
                raise ValueError(
                    f"Unsupported vision model type: {self.vision_config['model_type']}. Only moonvit is supported."
                )
            self.vision_config = MoonViTConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = Qwen2Config(architectures=["Qwen2ForCausalLM"])
        elif isinstance(self.text_config, dict):
            architectures = self.text_config.get("architectures") or ["Qwen2ForCausalLM"]
            if architectures[0] == "Qwen2ForCausalLM":
                self.text_config = Qwen2Config(**self.text_config)
            elif architectures[0] == "Qwen3ForCausalLM":
                self.text_config = Qwen3Config(**self.text_config)
            else:
                raise ValueError(
                    f"Unsupported language model architecture: {architectures[0]}. Only Qwen2ForCausalLM and "
                    "Qwen3ForCausalLM are supported."
                )

        if not hasattr(self.text_config, "rope_theta"):
            self.text_config.rope_theta = 10000.0

        # `magi` is the original block-diffusion training attention, not an inference backend.
        # Reset it so Transformers selects a real implementation (sdpa / flash) at load time.
        if getattr(self, "_attn_implementation", None) == "magi":
            self._attn_implementation = None

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        super().__post_init__(**kwargs)


__all__ = ["LocateAnythingConfig", "MoonViTConfig"]
