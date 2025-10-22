# Copyright 2025 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...utils import logging
from ..auto import AutoModel
from ..clip.modeling_clip import CLIPVisionModel
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2PreTrainedModel
from ..sam.modeling_sam import SamVisionEncoder
from .configuration_deepseek_ocr import DeepSeekOCRConfig


logger = logging.get_logger(__name__)


class DeepSeekOCRProjector(nn.Module):
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


class DeepSeekOCRSAMVisionModel(SamVisionEncoder):
    """
    SAM ViT-B vision encoder with additional neck layers for DeepSeek OCR.
    Wraps the SAM vision encoder and adds downsampling convolutions.
    """

    def __init__(self, config):
        super().__init__()
        out_channels = config.out_channels
        downsample_channels = config.downsample_channels

        self.net_2 = nn.Conv2d(out_channels, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values):
        encoder_output = self.encoder(pixel_values)
        hidden_states = encoder_output.last_hidden_state

        x2 = self.net_2(hidden_states)
        x3 = self.net_3(x2)

        return x3


class DeepSeekOCRCLIPVisionModel(CLIPVisionModel):
    pass


class DeepSeekOCRPreTrainedModel(DeepseekV2PreTrainedModel):
    config_class = DeepSeekOCRConfig
    base_model_prefix = "model"


class DeepSeekOCRModel(DeepSeekOCRPreTrainedModel):
    """
    DeepSeek OCR model with dual vision encoders (SAM + CLIP) and a projector.
    """

    def __init__(self, config: DeepSeekOCRConfig):
        super().__init__(config)
        self.config = config

        self.language_model = AutoModel.from_config(config.deepseek_config)

        self.sam_model = DeepSeekOCRSAMVisionModel(config.sam_vision_config)
        self.clip_model = DeepSeekOCRCLIPVisionModel(config.clip_vision_config)

        self.projector = DeepSeekOCRProjector(config.projector_config)

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

    def _encode_images(self, pixel_values, image_spatial_crop):
        """
        Encode images with dual encoders (SAM + CLIP).

        Args:
            pixel_values: (batch, 2, max_crops, 3, H, W) - [patches, global_view]
            image_spatial_crop: (batch, 2) - [width_crop_num, height_crop_num]
        """
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
            vision_features = self._encode_images(pixel_values, image_spatial_crop)

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


class DeepSeekOCRForCausalLM(DeepSeekOCRPreTrainedModel):
    """
    DeepSeek OCR model with a language modeling head.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepSeekOCRModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.BoolTensor] = None,
        image_spatial_crop: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_attention_mask=image_attention_mask,
            image_spatial_crop=image_spatial_crop,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
    "DeepSeekOCRModel",
    "DeepSeekOCRForCausalLM",
    "DeepSeekOCRPreTrainedModel",
    "DeepSeekOCRProjector",
    "DeepSeekOCRSAMVisionModel",
    "DeepSeekOCRCLIPVisionModel",
]
