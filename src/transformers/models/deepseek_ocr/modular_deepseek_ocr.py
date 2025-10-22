# Copyright 2025 Deepseek-AI and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..llava.modeling_llava import LlavaForConditionalGeneration
from ..clip.modeling_clip import CLIPEncoder, CLIPVisionModel, CLIPVisionEmbeddings, CLIPVisionTransformer
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2PreTrainedModel
from ..sam.modeling_sam import SamVisionEncoder
from .configuration_deepseek_ocr import DeepseekOcrConfig
logger = logging.get_logger(__name__)


class DeepseekOcrSAMVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_sam_vision"
    base_config_key = "sam_vision_config"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=None,
        mlp_ratio=4.0,
        output_channels=256,
        downsample_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes if global_attn_indexes is not None else [2, 5, 8, 11]
        self.mlp_ratio = mlp_ratio
        self.output_channels = output_channels
        self.downsample_channels = downsample_channels if downsample_channels is not None else [512, 1024]
        self.mlp_dim = int(hidden_size * mlp_ratio)
        self.out_channels = output_channels


class DeepseekOcrCLIPVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_clip_vision"
    base_config_key = "clip_vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor


class DeepseekOcrProjectorConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_projector"
    base_config_key = "projector_config"

    def __init__(
        self,
        input_dim=2048,
        n_embed=1280,
        projector_type="linear",
        depth=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.projector_type = projector_type
        self.depth = depth


class DeepseekOcrConfig(PreTrainedConfig):
    model_type = "deepseek_ocr"
    sub_configs = {
        "text_config": AutoConfig,
        "sam_vision_config": DeepseekOcrSAMVisionConfig,
        "clip_vision_config": DeepseekOcrCLIPVisionConfig,
        "projector_config": DeepseekOcrProjectorConfig,
    }

    def __init__(
        self,
        text_config=None,
        sam_vision_config=None,
        clip_vision_config=None,
        projector_config=None,
        candidate_resolutions=None,
        global_view_pos="head",
        tile_tag="2D",
        image_token_index=100015,
        **kwargs,
    ):
        if candidate_resolutions is None:
            candidate_resolutions = [[1024, 1024]]

        self.candidate_resolutions = candidate_resolutions
        self.global_view_pos = global_view_pos
        self.tile_tag = tile_tag
        self.image_token_index = image_token_index

        if sam_vision_config is None:
            self.sam_vision_config = DeepseekOcrSAMVisionConfig()
        elif isinstance(sam_vision_config, dict):
            self.sam_vision_config = DeepseekOcrSAMVisionConfig(**sam_vision_config)
        else:
            self.sam_vision_config = sam_vision_config

        if clip_vision_config is None:
            self.clip_vision_config = DeepseekOcrCLIPVisionConfig()
        elif isinstance(clip_vision_config, dict):
            self.clip_vision_config = DeepseekOcrCLIPVisionConfig(**clip_vision_config)
        else:
            self.clip_vision_config = clip_vision_config

        if projector_config is None:
            self.projector_config = DeepseekOcrProjectorConfig()
        elif isinstance(projector_config, dict):
            self.projector_config = DeepseekOcrProjectorConfig(**projector_config)
        else:
            self.projector_config = projector_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "deepseek_v2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["deepseek_v2"](
                hidden_size=1280,
                intermediate_size=6848,
                num_hidden_layers=12,
                num_attention_heads=10,
                num_key_value_heads=10,
                moe_intermediate_size=896,
                n_routed_experts=64,
                n_shared_experts=2,
                num_experts_per_tok=6,
                first_k_dense_replace=1,
                vocab_size=129280,
                max_position_embeddings=8192,
                use_mla=False,
            )

        self.text_config = text_config
        self.hidden_size = text_config.hidden_size
        self.vocab_size = text_config.vocab_size

        super().__init__(**kwargs)


class DeepseekOcrProjector(nn.Module):
    """
    Projector that maps concatenated SAM + CLIP features to language model space.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.projector_type == "identity":
            self.layers = nn.Identity()
        elif config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            mlp_depth = config.get("depth", 1)
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x):
        return self.layers(x)


class DeepseekOcrSAMVisionEncoder(SamVisionEncoder):
    """
    SAM ViT-B vision encoder with additional neck layers for Deepseek OCR.
    Wraps the SAM vision encoder and adds downsampling convolutions.
    """

    def __init__(self, config):
        super().__init__()
        out_channels = config.out_channels
        downsample_channels = config.downsample_channels
        
        # TODO move hardcoded values to config
        self.net_2 = nn.Conv2d(out_channels, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values):
        encoder_output = self.encoder(pixel_values)
        hidden_states = encoder_output.last_hidden_state

        hidden_states = self.net_2(hidden_states)
        hidden_states = self.net_3(hidden_states)

        return hidden_states


class DeepseekOcrVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values, patch_embeds, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape

        #if patch_embeds is not None:
        #    patch_embeds = patch_embeds
        #else:
        patch_embeds = self.patch_embedding(pixel_values) # Deepseek OCR CLIP embedder always uses SAM features
        
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class DeepseekOcrCLIPEncoder(CLIPEncoder):
    pass

class DeepseekOcrCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__()
        self.embeddings = DeepseekOcrVisionEmbeddings(config)
        self.encoder = DeepseekOcrCLIPEncoder(config)


class DeepseekOcrVisionModel(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()
        self.vision_model = DeepseekOcrCLIPVisionTransformer(config)


class DeepseekOcrPreTrainedModel(PreTrainedModel):
    config_class = DeepseekOcrConfig
    base_model_prefix = "model"


class DeepseekOcrModel(DeepseekOcrPreTrainedModel):
    """
    Deepseek OCR model with dual vision encoders (SAM + CLIP) and a projector.
    """

    def __init__(self, config: DeepseekOcrConfig):
        super().__init__(config)
        self.config = config

        self.language_model = AutoModel.from_config(config.deepseek_config)

        self.sam_model = DeepseekOcrSAMVisionEncoder(config.sam_vision_config)
        self.clip_model = AutoModel.from_config(config.clip_vision_config)

        self.projector = DeepseekOcrProjector(config.projector_config)

        embed_std = 1 / math.sqrt(config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * embed_std)
        self.view_separator = nn.Parameter(torch.randn(config.hidden_size) * embed_std)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def _merge_image_features(self, local_features, global_features, spatial_crop):
        """
        Merge local and global image features with newlines and separators.

        Args:
            local_features: (batch, num_patches, height*width, hidden_size)
            global_features: (batch, height*width, hidden_size)
            spatial_crop: (batch, 2) - [width_crop_num, height_crop_num]
        """
        batch_size = local_features.size(0) if local_features is not None else global_features.size(0)
        all_image_features = []

        for idx in range(batch_size):
            global_feat = global_features[idx]
            hw, n_dim = global_feat.shape
            h = w = int(hw**0.5)

            global_feat = global_feat.view(h, w, n_dim)
            global_feat = torch.cat([global_feat, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
            global_feat = global_feat.view(-1, n_dim)

            if local_features is not None and spatial_crop[idx, 0] > 1 or spatial_crop[idx, 1] > 1:
                local_feat = local_features[idx]
                width_crop_num, height_crop_num = int(spatial_crop[idx, 0]), int(spatial_crop[idx, 1])

                hw2, n_dim2 = local_feat.shape
                h2 = w2 = int(hw2**0.5)

                local_feat = (
                    local_feat.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                )
                local_feat = torch.cat(
                    [local_feat, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                )
                local_feat = local_feat.view(-1, n_dim2)

                image_features = torch.cat([local_feat, global_feat, self.view_separator[None, :]], dim=0)
            else:
                image_features = torch.cat([global_feat, self.view_separator[None, :]], dim=0)

            all_image_features.append(image_features)

        return torch.cat(all_image_features, dim=0)

    def get_image_features(self, pixel_values, image_spatial_crop):
        batch_size = pixel_values.size(0)
        patches = pixel_values[:, 0]
        global_view = pixel_values[:, 1]

        all_features = []

        for idx in range(batch_size):
            patch_images = patches[idx]
            global_image = global_view[idx].unsqueeze(0)

            has_patches = torch.sum(patch_images).item() != 0

            if has_patches:
                sam_local = self.sam_model(patch_images)
                clip_local = self.clip_model(patch_images)
                local_features = torch.cat([clip_local[:, 1:], sam_local.flatten(2).permute(0, 2, 1)], dim=-1)
                local_features = self.projector(local_features)
            else:
                local_features = None

            sam_global = self.sam_model(global_image)
            clip_global = self.clip_model(global_image)
            global_features = torch.cat([clip_global[:, 1:], sam_global.flatten(2).permute(0, 2, 1)], dim=-1)
            global_features = self.projector(global_features)

            merged_features = self._merge_image_features(
                local_features, global_features, image_spatial_crop[idx : idx + 1]
            )
            all_features.append(merged_features)

        return torch.cat(all_features, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.BoolTensor] = None,
        image_spatial_crop: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and torch.sum(pixel_values[0, 1]).item() != 0:
            
            vision_features = self.get_image_features(pixel_values, image_spatial_crop)

            inputs_embeds = inputs_embeds.masked_scatter(
                image_attention_mask.unsqueeze(-1).to(inputs_embeds.device), vision_features.to(inputs_embeds.dtype)
            )

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

@auto_docstring(
    custom_intro="""
    The Deepseek-OCR model which consists of two vision backbones and a deepseek language model.
    """
)
class DeepseekOcrForConditionalGeneratin(LlavaForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekOcrModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.BoolTensor] = None,
        image_spatial_crop: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_attention_mask=image_attention_mask, # TODO this is just the special image mask
            image_spatial_crop=image_spatial_crop,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        image_attention_mask=None,
        image_spatial_crop=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.seen_tokens
            else:
                past_length = past_key_values[0][0].shape[2]

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_attention_mask": image_attention_mask,
                "image_spatial_crop": image_spatial_crop,
            }
        )
        return model_inputs


__all__ = [
    "DeepseekOcrModel",
    "DeepseekOcrForCausalLM",
    "DeepseekOcrPreTrainedModel",
    "DeepseekOcrProjector",
    "DeepseekOcrSAMVisionEncoder",
    "DeepseekOcrCLIPVisionModel",
]
