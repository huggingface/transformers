# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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


from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import AttentionInterface, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..idefics3.modeling_idefics3 import Idefics3VisionEmbeddings
from ..lfm2_vl.modeling_lfm2_vl import Lfm2VlModel
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoder, SiglipEncoderLayer, SiglipMLP


def minicpmv4_6_vision_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """SDPA with per-segment attention via cu_seq_lens for packed NaViT sequences."""
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", False)

    if cu_seq_lens_q is not None:
        attn_output = torch.empty_like(query)
        for i in range(len(cu_seq_lens_q) - 1):
            start, end = cu_seq_lens_q[i].item(), cu_seq_lens_q[i + 1].item()
            attn_output[:, :, start:end, :] = F.scaled_dot_product_attention(
                query[:, :, start:end, :],
                key[:, :, start:end, :],
                value[:, :, start:end, :],
                dropout_p=dropout,
                is_causal=False,
            )
    else:
        is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# FIXME @vasqu: needs a varlen path for SDPA similar to qwen-vision-attention
MINICPMV4_6_ATTENTION_FUNCTIONS = AttentionInterface()
MINICPMV4_6_ATTENTION_FUNCTIONS["sdpa"] = minicpmv4_6_vision_sdpa_attention_forward


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.6")
@strict
class MiniCPMV4_6VisionConfig(SiglipVisionConfig):
    r"""
    insert_layer_id (`int`, *optional*, defaults to 6):
        Vision encoder layer index after which the window-attention merger is applied.

    Example:

    ```python
    >>> from transformers import MiniCPMV4_6VisionConfig

    >>> configuration = MiniCPMV4_6VisionConfig()
    >>> print(configuration.hidden_size)
    768
    ```"""

    model_type = "minicpmv4_6_vision"
    insert_layer_id: int = 6

    def __post_init__(self, **kwargs):
        # FIXME: why not adjust configs or do we want this to be dynamically applied?
        # Needs a comment in code
        if self.drop_vision_last_layer:
            self.num_hidden_layers -= 1
        PreTrainedConfig.__post_init__(**kwargs)


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.6")
@strict
class MiniCPMV4_6Config(PreTrainedConfig):
    r"""
    insert_layer_id (`int`, *optional*, defaults to 6):
        Vision encoder layer index after which the window-attention merger is applied.
    image_size (`int`, *optional*, defaults to 448):
        Base resolution for image preprocessing.
    drop_vision_last_layer (`bool`, *optional*, defaults to `False`):
        Whether to drop the last layer of the vision encoder.
    image_token_id (`int`, *optional*):
        Token id used as the image placeholder.
    downsample_mode (`str`, *optional*, defaults to `"16x"`):
        Visual token downsampling ratio. `"4x"` keeps 4× more tokens.
    merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        Kernel size `(h, w)` for merging adjacent visual patches in the Merger.
    merger_times (`int`, *optional*, defaults to 1):
        Number of iterative merge rounds in the Merger.
    """

    model_type = "minicpmv4_6"
    sub_configs = {"text_config": AutoConfig, "vision_config": MiniCPMV4_6VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    insert_layer_id: int = 6
    image_size: int = 448
    drop_vision_last_layer: bool = False
    image_token_id: int = 32000  # cannot be `None` or we get error no? Needs default
    tie_word_embeddings: bool = False
    downsample_mode: str = "16x"
    merge_kernel_size: tuple[int, int] | list[int] = (2, 2)
    merger_times: int = 1

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config.pop("model_type", None)
            self.vision_config = MiniCPMV4_6VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = MiniCPMV4_6VisionConfig()

        self.vision_config.insert_layer_id = self.insert_layer_id
        self.patch_size = self.vision_config.patch_size

        if isinstance(self.text_config, dict):
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3_5_text"]()

        super().__post_init__(**kwargs)


class MiniCPMV4_6VisionEmbeddings(Idefics3VisionEmbeddings):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        target_sizes: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)

        position_embeddings = []
        for target_size in target_sizes:
            nb_patches_h = target_size[0]
            nb_patches_w = target_size[1]

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (
                (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w)
                .flatten()
                .to(self.position_embedding.weight.device)
            )

            position_embeddings.append(self.position_embedding(pos_ids))

        position_embeddings = torch.concat(position_embeddings, dim=0).unsqueeze(0)
        embeddings = embeddings + position_embeddings
        return embeddings


class MiniCPMV4_6VisionMLP(SiglipMLP):
    pass


class MiniCPMV4_6VisionAttention(SiglipAttention):
    """NaViT variant: supports packed variable-length attention via cu_seqlens/max_seqlens."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = MINICPMV4_6_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MiniCPMV4_6VisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: MiniCPMV4_6VisionConfig):
        super().__init__(config)
        self.self_attn = MiniCPMV4_6VisionAttention(config)
        self.mlp = MiniCPMV4_6VisionMLP(config)


class MiniCPMV4_6VisionEncoder(SiglipEncoder):
    """Transformer encoder consisting of `config.num_hidden_layers` [`MiniCPMV4_6VisionEncoderLayer`] layers."""

    def __init__(self, config: MiniCPMV4_6VisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MiniCPMV4_6VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class MiniCPMV4_6ViTWindowAttentionMerger(nn.Module):
    def __init__(self, config: MiniCPMV4_6VisionConfig):
        super().__init__()
        self.window_kernel_size = (2, 2)
        self.embed_dim = config.hidden_size

        self.self_attn = MiniCPMV4_6VisionAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        hidden_4x = self.embed_dim * self.window_kernel_size[0] * self.window_kernel_size[1]
        inter_4x = config.intermediate_size * self.window_kernel_size[0] * self.window_kernel_size[1]

        self.pre_norm = nn.LayerNorm(hidden_4x, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(hidden_4x, inter_4x, bias=True)
        self.act = ACT2FN["gelu_pytorch_tanh"]
        self.linear_2 = nn.Linear(inter_4x, self.embed_dim, bias=True)

    def _init_weights(self):
        """Block-diagonal normal init: preserves the structural prior that each
        2x2 window patch is processed independently at initialization."""
        for proj in (self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj, self.self_attn.out_proj):
            proj.weight.data.normal_()
            proj.bias.data.zero_()

        for ln in (self.layer_norm1, self.pre_norm):
            ln.weight.data.fill_(1.0)
            ln.bias.data.zero_()

        hidden_size = self.embed_dim
        intermediate_size = self.linear_1.weight.shape[0] // 4
        self.linear_1.weight.data.zero_()
        for i in range(4):
            self.linear_1.weight.data[
                i * intermediate_size : (i + 1) * intermediate_size,
                i * hidden_size : (i + 1) * hidden_size,
            ].normal_()
        self.linear_1.bias.data.normal_(std=1e-6)

        self.linear_2.weight.data.normal_(std=0.25)
        self.linear_2.bias.data.normal_(std=1e-6)

    def get_window_index(self, target_sizes):
        window_h, window_w = self.window_kernel_size
        max_seqlens = window_h * window_w

        window_index_list = []
        cu_seqlens = [0]
        token_offset = 0

        for height, width in target_sizes:
            if height % window_h != 0 or width % window_w != 0:
                raise ValueError(
                    f"height={height}, width={width} must be divisible by window size ({window_h}, {window_w})"
                )
            index = torch.arange(height * width).reshape(height, width)
            num_windows_h = height // window_h
            num_windows_w = width // window_w
            num_windows = num_windows_h * num_windows_w

            index = index.reshape(num_windows_h, window_h, num_windows_w, window_w)
            index = index.permute(0, 2, 1, 3).reshape(num_windows, window_h * window_w)

            window_index_list.append(index.reshape(-1) + token_offset)

            cu_this = torch.arange(1, num_windows + 1) * (window_h * window_w) + cu_seqlens[-1]
            cu_seqlens.extend(cu_this.tolist())

            token_offset += height * width

        window_index = torch.cat(window_index_list)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        return window_index, cu_seqlens, max_seqlens

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sizes: torch.IntTensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlens: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        device = hidden_states.device

        window_index, window_cu_seqlens, window_max_seqlens = self.get_window_index(target_sizes)
        window_index = window_index.to(device)

        hidden_states = hidden_states[:, window_index, :]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            cu_seq_lens_q=window_cu_seqlens.to(device),
            cu_seq_lens_k=window_cu_seqlens.to(device),
            max_length_q=window_max_seqlens,
            max_length_k=window_max_seqlens,
        )
        hidden_states = hidden_states[:, torch.argsort(window_index), :]
        hidden_states = residual + hidden_states

        batch_size, _ = target_sizes.shape
        if (target_sizes % 2 != 0).any():
            raise ValueError(f"All target_sizes must be divisible by 2, got {target_sizes}")
        new_target_sizes = target_sizes // 2

        window_h, window_w = self.window_kernel_size
        all_pixel_values = []
        for batch_idx in range(batch_size):
            height, width = target_sizes[batch_idx]
            patch = hidden_states[0, cu_seqlens[batch_idx] : cu_seqlens[batch_idx + 1], :].squeeze(0)

            embed_dim = patch.shape[-1]
            merged_h, merged_w = height // window_h, width // window_w
            patch_5d = patch.view(merged_h, window_h, merged_w, window_w, embed_dim).permute(0, 2, 1, 3, 4)
            hidden_state = patch_5d.reshape(merged_h * merged_w, window_h * window_w * embed_dim)
            residual = patch_5d.reshape(merged_h * merged_w, window_h * window_w, embed_dim).mean(dim=1)

            hidden_state = self.pre_norm(hidden_state)
            hidden_state = self.linear_1(hidden_state)
            hidden_state = self.act(hidden_state)
            hidden_state = self.linear_2(hidden_state)

            all_pixel_values.append(hidden_state + residual)

        new_hidden_states = torch.concat(all_pixel_values, dim=0).unsqueeze(0)
        new_cu_seqlens = F.pad(
            torch.cumsum(new_target_sizes[:, 0] * new_target_sizes[:, 1], dim=0, dtype=torch.int32).to(device), (1, 0)
        )
        if max_seqlens % 4 != 0:
            raise ValueError(f"max_seqlens ({max_seqlens}) must be divisible by 4")
        new_max_seqlens = max_seqlens // 4

        return (
            new_hidden_states,
            new_target_sizes,
            new_cu_seqlens,
            new_max_seqlens,
        )


class MiniCPMV4_6VisionPreTrainedModel(PreTrainedModel):
    config_class = MiniCPMV4_6VisionConfig
    main_input_name = "pixel_values"
    _input_embed_layer = "patch_embedding"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True

    _can_record_outputs = {
        "hidden_states": MiniCPMV4_6VisionEncoderLayer,
        "attentions": MiniCPMV4_6VisionAttention,
    }


class MiniCPMV4_6VisionTransformer(MiniCPMV4_6VisionPreTrainedModel):
    def __init__(self, config: MiniCPMV4_6VisionConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.embeddings = MiniCPMV4_6VisionEmbeddings(config)
        self.encoder = MiniCPMV4_6VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values,
        target_sizes: torch.IntTensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlens: torch.Tensor | None = None,
        use_vit_merger: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        target_sizes (`torch.IntTensor` of shape `(batch_size, 2)`, *optional*):
            Patch grid sizes `(h, w)` for computing position embeddings.
        cu_seqlens (`torch.Tensor`, *optional*):
            Cumulative sequence lengths for packed (flash) attention.
        max_seqlens (`torch.Tensor`, *optional*):
            Maximum sequence length for packed (flash) attention.
        use_vit_merger (`bool`, *optional*, defaults to `True`):
            Whether to apply the ViT window-attention merger after the encoder.
        """

        hidden_states = self.embeddings(pixel_values, target_sizes=target_sizes)
        attn_kwargs = {
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seqlens,
            "max_length_k": max_seqlens,
            **kwargs,
        }

        insert_layer_id = self.config.insert_layer_id if use_vit_merger else -1
        if use_vit_merger and insert_layer_id >= 0:
            for layer_index, encoder_layer in enumerate(self.encoder.layers):
                hidden_states = encoder_layer(hidden_states, **attn_kwargs)
                # NOTE: Downsample the hidden states and therefore cu-seqlens are new values!
                if layer_index == insert_layer_id:
                    (hidden_states, target_sizes, cu_seqlens, max_seqlens) = self.vit_merger(
                        hidden_states,
                        target_sizes,
                        cu_seqlens,
                        max_seqlens,
                    )
                    attn_kwargs = {
                        "cu_seq_lens_q": cu_seqlens,
                        "cu_seq_lens_k": cu_seqlens,
                        "max_length_q": max_seqlens,
                        "max_length_k": max_seqlens,
                        **kwargs,
                    }
        else:
            encoder_outputs = self.encoder(inputs_embeds=hidden_states, **attn_kwargs)
            hidden_states = encoder_outputs.last_hidden_state

        last_hidden_state = self.post_layernorm(hidden_states)

        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)


class MiniCPMV4_6DownsampleMLP(nn.Module):
    def __init__(self, hidden_size: int, llm_embed_dim: int):
        super().__init__()
        # factor 4 = two successive 2×2 spatial merges (ViT insert merger + downsample MLP)
        merged_hidden_size = hidden_size * 4

        self.pre_norm = nn.LayerNorm(merged_hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(merged_hidden_size, merged_hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(merged_hidden_size, llm_embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states).view(-1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MiniCPMV4_6Merger(nn.Module):
    def __init__(self, config: MiniCPMV4_6Config):
        super().__init__()

        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.merger_times = config.merger_times
        hidden_size = config.vision_config.hidden_size
        llm_embed_dim = config.text_config.hidden_size
        # Downsample `self.merger_times - 1` times and finally apply projection into LLM space
        self.mlp = nn.ModuleList(
            [
                MiniCPMV4_6DownsampleMLP(
                    hidden_size, llm_embed_dim if i == self.merger_times - 1 else hidden_size, self.merge_kernel_size
                )
                for i in range(self.merger_times)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sizes: torch.IntTensor,
    ) -> list[torch.Tensor]:
        merge_h, merge_w = self.merge_kernel_size

        start = 0
        processed_features = []
        for batch_idx in range(len(target_sizes)):
            height, width = target_sizes[batch_idx]
            num_patches = height * width

            embed_dim = hidden_states.shape[-1]
            merged_h, merged_w = height // merge_h, width // merge_w
            hidden_state = (
                hidden_states[0, start : start + num_patches, :]
                .view(merged_h, merge_h, merged_w, merge_w, embed_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(merged_h * merged_w, merge_h * merge_w * embed_dim)
            )
            hidden_state = self.mlp[0](hidden_state)

            for i in range(1, self.merger_times):
                if height % merge_h != 0 or width % merge_w != 0:
                    raise ValueError(
                        f"Patch grid ({height}, {width}) must be divisible by merge kernel size "
                        f"{self.merge_kernel_size} at merge round {i}"
                    )
                height = height // merge_h
                width = width // merge_w

                inner_dim = hidden_state.shape[-1]
                merged_h, merged_w = height // merge_h, width // merge_w
                hidden_state = (
                    hidden_state.view(merged_h, merge_h, merged_w, merge_w, inner_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(merged_h * merged_w, merge_h * merge_w * inner_dim)
                )
                hidden_state = self.mlp[i](hidden_state)

            start += num_patches
            processed_features.append(hidden_state)

        return processed_features


class MiniCPMV4_6PreTrainedModel(PreTrainedModel):
    config_class = MiniCPMV4_6Config
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _no_split_modules = [
        "MiniCPMV4_6VisionEmbeddings",
        "MiniCPMV4_6VisionEncoderLayer",
        "MiniCPMV4_6ViTWindowAttentionMerger",
    ]


class MiniCPMV4_6Model(Lfm2VlModel):
    def __init__(self, config: MiniCPMV4_6Config):
        super().__init__(config)
        del self.multi_modal_projector

        self.vision_tower = MiniCPMV4_6VisionTransformer._from_config(config.vision_config)
        self.merger = MiniCPMV4_6Merger(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring(custom_intro="Extract image features: vision encoder, insert merger, then MLP merger.")
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        target_sizes: torch.IntTensor,
        downsample_mode: str | None = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        target_sizes (`torch.IntTensor` of shape `(num_images, 2)`):
            Height and width (in patches) of each image.
        downsample_mode (`str`, *optional*):
            When set to `"4x"` the intermediate `vit_merger` is skipped so that each image keeps
            `4×` more visual tokens. Default `"16x"` mode applies the full merge pipeline.
        """
        downsample_mode = downsample_mode if downsample_mode else self.config.downsample_mode
        use_vit_merger = downsample_mode != "4x"

        cu_seqlens = F.pad(torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32), (1, 0))
        max_seqlens = int(torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item())

        vision_output = self.vision_tower(
            pixel_values,
            target_sizes=target_sizes,
            cu_seqlens=cu_seqlens,
            max_seqlens=max_seqlens,
            use_vit_merger=use_vit_merger,
        )

        if use_vit_merger:
            target_sizes = target_sizes // 2
        vision_output.pooler_output = self.merger(vision_output.last_hidden_state, target_sizes)
        return vision_output

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: list[list[torch.Tensor]] | None = None,
        target_sizes: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        downsample_mode: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values (`list[list[torch.Tensor]]`, *optional*):
            Pixel value patches per sample, grouped as `[sample][image_slices]`.
        target_sizes (`list[torch.Tensor]` or `list[list]`, *optional*):
            Height and width (in patches) for each image per sample.
        downsample_mode (`str`, *optional*):
            `"4x"` keeps 4x more visual tokens; default `"16x"` applies full merge.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            vision_output = self.get_image_features(pixel_values, target_sizes, downsample_mode=downsample_mode)
            image_features = torch.cat(vision_output.pooler_output, dim=0).to(device=inputs_embeds.device)
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        output = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return output


class MiniCPMV4_6ForConditionalGeneration(MiniCPMV4_6PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: MiniCPMV4_6Config):
        super().__init__(config)
        self.model = MiniCPMV4_6Model(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: list[list[torch.Tensor]] | None = None,
        target_sizes: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        downsample_mode: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        pixel_values (`list[list[torch.Tensor]]`, *optional*):
            Pixel value patches per sample, grouped as `[sample][image_slices]`.
        target_sizes (`list[torch.Tensor]` or `list[list]`, *optional*):
            Height and width (in patches) for each image per sample.
        downsample_mode (`str`, *optional*):
            `"4x"` keeps 4x more visual tokens; default `"16x"` applies full merge.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            target_sizes=target_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            downsample_mode=downsample_mode,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @auto_docstring(custom_intro="Extract image features: vision encoder, insert merger, then MLP merger.")
    def get_image_features(self, *args, **kwargs) -> BaseModelOutputWithPooling:
        return self.model.get_image_features(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        target_sizes=None,
        downsample_mode=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            downsample_mode=downsample_mode,
            **kwargs,
        )
        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["target_sizes"] = target_sizes
        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        if expand_size > 1:
            for key in ("pixel_values", "target_sizes"):
                values = model_kwargs.get(key)
                if values is not None and isinstance(values, list):
                    model_kwargs[key] = [v for v in values for _ in range(expand_size)]
        return input_ids, model_kwargs


__all__ = [
    "MiniCPMV4_6Config",
    "MiniCPMV4_6PreTrainedModel",
    "MiniCPMV4_6Model",
    "MiniCPMV4_6ForConditionalGeneration",
]
