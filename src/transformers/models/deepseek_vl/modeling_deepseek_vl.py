

from _utils import show_tensor

from typing import List, Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...modeling_utils import PreTrainedModel
from ...generation import GenerationMixin
from ..siglip.modeling_siglip import SiglipVisionModel
from ..sam.configuration_sam import SamVisionConfig
from ..sam.modeling_sam import SamVisionEncoder
from .configuration_deepseek_vl import DeepseekVLAlignerConfig, DeepseekVLVisionConfig, DeepseekVLConfig


class DeepseekVLSamVisionNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.conv1 = nn.Conv2d(
            config.output_channels,
            config.output_channels*2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            config.output_channels*2,
            config.output_channels*4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(
            features,
            size=(96, 96),
            mode="bilinear",
            align_corners=False,
        )
        features = self.conv1(features)
        features = self.conv2(features)
        return features


class DeepseekVLVisionModel(nn.Module):
    def __init__(self, config: DeepseekVLVisionConfig):
        super().__init__()
        self.concat_type = config.concat_type
        self.use_high_res = config.use_high_res
        self.low_res_config = config.low_res_config
        self.high_res_config = config.high_res_config

        self.resize = torchvision.transforms.Resize(self.low_res_config.image_size, antialias=True)
        self.low_res_model = SiglipVisionModel(self.low_res_config)
        self.low_res_norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if self.use_high_res:
            self.high_res_model = SamVisionEncoder(self.high_res_config)
            self.high_res_model_global_neck = deepcopy(self.high_res_model.neck)
            self.high_res_neck = DeepseekVLSamVisionNeck(self.high_res_config)
            self.high_res_alpha = nn.Parameter(torch.zeros(1))
            self.high_res_norm = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, n_images, n_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * n_images, n_channels, height, width)

        if self.use_high_res:
            high_res_pixel_values = pixel_values
            high_res_pixel_values = self.high_res_norm(pixel_values)
            high_res_output = self.high_res_model(
                pixel_values=high_res_pixel_values,
                output_hidden_states=True,
            )
            high_res_last_hidden_state = high_res_output[0]
            global_attn_index = self.high_res_config.global_attn_indexes[0]
            high_res_global_hidden_state = high_res_output[1][global_attn_index+1] # +1 for embedding layer
            high_res_global_hidden_state = self.high_res_model_global_neck(high_res_global_hidden_state)

            high_res_last_hidden_state = self.high_res_neck(high_res_last_hidden_state)
            high_res_global_hidden_state = self.high_res_neck(high_res_global_hidden_state)
            high_res_last_hidden_state = high_res_last_hidden_state + high_res_global_hidden_state * self.high_res_alpha

            high_res_last_hidden_state = high_res_last_hidden_state.permute(0, 2, 3, 1)
            batch_size_x_n_images, height, width, hidden_size = high_res_last_hidden_state.shape
            high_res_last_hidden_state = high_res_last_hidden_state.reshape(batch_size_x_n_images, height * width, hidden_size)
        else:
            high_res_last_hidden_state = None

        low_res_pixel_values = self.resize(pixel_values)
        low_res_pixel_values = self.low_res_norm(low_res_pixel_values)
        low_res_output = self.low_res_model(
            pixel_values=low_res_pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
        low_res_last_hidden_state = low_res_output[0]

        return low_res_last_hidden_state, high_res_last_hidden_state


class DeepseekVLAlignerModel(nn.Module):
    def __init__(self, config: DeepseekVLAlignerConfig):
        super().__init__()
        self.config = config

    def forward(self, vision_encodings: Tuple[torch.Tensor, Optional[torch.Tensor]]):
        low_res_encodings, high_res_encodings = vision_encodings

        show_tensor(high_res_encodings, name="high_res_encodings")
        # show_tensor(low_res_encodings, name="low_res_encodings")

        """
        low_res_encodings.shape: [1, 576, 1024] (batch_size, seq_len, config.low_res_config.output_channels)
        high_res_encodings.shape: [1, 576, 1024] (batch_size, seq_len, config.high_res_config.output_channels)

        low_res_encodings
        - Siglip
        - always provided
        - used in 1.3b model and 7b model
        high_res_encodings
        - Sam
        - None if config.use_high_res is False
        - used in 7b model
        """


class DeepseekVLPreTrainedModel(PreTrainedModel):
    config_class = DeepseekVLConfig
    base_model_prefix = "deepseek_vl"


class DeepseekVLModel(DeepseekVLPreTrainedModel):

    def __init__(self, config: DeepseekVLConfig):
        super().__init__(config)
        self.config = config

        self.vision_model = DeepseekVLVisionModel(config.vision_config)
        self.aligner_model = DeepseekVLAlignerModel(config.aligner_config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        vision_encodings = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        self.aligner_model(vision_encodings)


class DeepseekVLForVisionText2Text(DeepseekVLPreTrainedModel, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.model = DeepseekVLModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            cache_position=cache_position,
        )




