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
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..sam.configuration_sam import SamVisionConfig
from ..sam.modeling_sam import SamVisionEncoder
from ..llama.modeling_llama import LlamaForCausalLM
from .configuration_deepseek_vl import DeepseekVLVisionConfig, DeepseekVLConfig


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


class DeepseekVLSamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.global_attn_index = config.global_attn_indexes[0]

        self.model = SamVisionEncoder(config)
        self.global_neck = deepcopy(self.model.neck)
        self.neck = DeepseekVLSamVisionNeck(config)
        self.alpha = nn.Parameter(torch.zeros(1))
        # TODO: convert to python functions
        self.norm = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values
        pixel_values = self.norm(pixel_values)
        output = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        last_hidden_state = output[0]
        last_hidden_state = self.neck(last_hidden_state)

        global_hidden_state = output[1][self.global_attn_index+1] # +1 for embedding layer
        global_hidden_state = self.global_neck(global_hidden_state)
        global_hidden_state = self.neck(global_hidden_state)

        output = last_hidden_state + global_hidden_state * self.alpha

        # batch_size, hidden_size, height, width -> batch_size, seq_len, hidden_size
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.shape[0], -1, output.shape[-1])

        return output


class DeepseekVLSiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config

        self.model = SiglipVisionModel(config)
        # TODO: convert to torch funtions
        self.resize = torchvision.transforms.Resize(config.image_size, antialias=True)
        self.norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = self.resize(pixel_values)
        pixel_values = self.norm(pixel_values)
        output = self.model(pixel_values=pixel_values)
        output = output[0] # last_hidden_state
        return output


class DeepseekVLVisionEncoder(nn.Module):
    def __init__(self, config: DeepseekVLVisionConfig):
        super().__init__()
        self.config = config
        self.use_high_res = config.use_high_res

        # Siglip is used for low resolution encoding
        self.low_res_encoder = DeepseekVLSiglipVisionEncoder(config.low_res_config)
        # Sam is used for high resolution encoding
        if self.use_high_res:
            self.high_res_encoder = DeepseekVLSamVisionEncoder(config.high_res_config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, n_images, n_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * n_images, n_channels, height, width)

        low_res_encodings = self.low_res_encoder(pixel_values)
        high_res_encodings = self.high_res_encoder(pixel_values) if self.use_high_res else None

        return low_res_encodings, high_res_encodings


class DeepseekVLAligner(nn.Module):
    def __init__(self, config: DeepseekVLConfig):
        super().__init__()
        self.config = config
        self.use_high_res = config.vision_config.use_high_res

        low_res_in_channels = config.vision_config.low_res_config.hidden_size
        high_res_in_channels = config.vision_config.high_res_config.output_channels * 4
        out_channels = config.text_config.hidden_size
        if self.use_high_res:
            self.low_res_proj = nn.Linear(low_res_in_channels, out_channels // 2)
            self.high_res_proj = nn.Linear(high_res_in_channels, out_channels // 2)
        else:
            self.low_res_proj = nn.Linear(low_res_in_channels, out_channels)

        self.act = nn.GELU()
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(self, low_res_encodings: torch.Tensor, high_res_encodings: Optional[torch.Tensor]) -> torch.Tensor:
        encodings = self.low_res_proj(low_res_encodings)
        if self.use_high_res:
            high_res_encodings = self.high_res_proj(high_res_encodings)
            encodings = torch.concat([high_res_encodings, encodings], dim=-1)

        encodings = self.act(encodings)
        encodings = self.proj(encodings)

        return encodings


class DeepseekVLPreTrainedModel(PreTrainedModel):
    config_class = DeepseekVLConfig
    base_model_prefix = "deepseek_vl"


class DeepseekVLModel(DeepseekVLPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = DeepseekVLVisionEncoder(config.vision_config)
        self.language_model = LlamaForCausalLM(config.text_config)
        self.aligner = DeepseekVLAligner(config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_embeddings(self, pixel_values):
        low_res_embeds, high_res_embeds = self.vision_model(pixel_values=pixel_values)
        images_embeds = self.aligner(low_res_embeds, high_res_embeds)
        # (batch_size * n_images, seq_len, hidden_size) -> (batch_size, n_images * seq_len, hidden_size)
        images_embeds = images_embeds.reshape(pixel_values.shape[0], -1, images_embeds.shape[-1])
        return images_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.Tensor] = None,
        images_emb_mask: Optional[torch.Tensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            images_embeds = self.get_image_embeddings(pixel_values)
            images_embeds = images_embeds.to(inputs_embeds)
            inputs_embeds[images_seq_mask] = images_embeds

        show_tensor(inputs_embeds, name="inputs_embeds")

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


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
        images_seq_mask: Optional[torch.Tensor] = None,
        images_emb_mask: Optional[torch.Tensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )




