# coding=utf-8
# Copyright 2025 Apple Inc. and The HuggingFace Team. All rights reserved.
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

"""Pytorch implementation of AIMv2 Model"""

import math
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from ...activations import ACT2FN
from ...utils import (
    logging,
)
from ..clip.modeling_clip import CLIPModel, CLIPOutput, CLIPTextEmbeddings, _get_vector_norm
from ..llama.modeling_llama import LlamaRMSNorm
from ..siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipEncoder


logger = logging.get_logger(__name__)


class AIMv2VisionConfig(SiglipVisionConfig):
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        hidden_act="silu",
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            **kwargs,
        )

        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.qkv_bias = qkv_bias
        self.rms_norm_eps = rms_norm_eps
        self.projection_dropout = projection_dropout

        del self.layer_norm_eps


class AIMv2TextConfig(SiglipTextConfig):
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id: int = 49407,
        max_position_embeddings: int = 77,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            qkv_bias=qkv_bias,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.projection_dropout = projection_dropout
        self.use_bias = use_bias

        del self.bos_token_id
        del self.pad_token_id
        del self.projection_size
        del self.hidden_act
        del self.layer_norm_eps


class AIMv2Config(SiglipConfig):
    # Modify default logit scale value accordingly with aimv2 configs
    def __init__(self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(text_config, vision_config, **kwargs)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    pass


class AIMv2Output(CLIPOutput):
    pass


class AIMv2RMSNorm(LlamaRMSNorm):
    pass


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__()
        in_features = config.hidden_size
        out_features = config.intermediate_size
        self.act_fn = config.hidden_act

        self.fc1 = nn.Linear(in_features, out_features, bias=config.use_bias)
        self.fc2 = nn.Linear(out_features, in_features, bias=config.use_bias)
        self.fc3 = nn.Linear(in_features, out_features, bias=config.use_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fc3_out = self.fc3(hidden_states)
        fc1_out = self.fc1(hidden_states)
        hidden_states = ACT2FN[self.act_fn](fc1_out) * fc3_out
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AIMv2VisionEmbeddings(nn.Module):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.rms_norm = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ):
        grid_w = torch.arange(int(width), dtype=dtype, device=device)
        grid_h = torch.arange(int(height), dtype=dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="xy")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_h = grid_h.flatten()[..., None] @ omega[None, :]
        out_w = grid_w.flatten()[..., None] @ omega[None, :]

        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.size()
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = self.rms_norm(hidden_states)

        if self.config.image_size != height or self.config.image_size != width:
            pos_embed = self.build_2d_sincos_position_embedding(
                height // self.patch_size, width // self.patch_size, embed_dim=self.config.hidden_size
            )
        else:
            pos_embed = self.position_embeddings(self.position_ids)

        hidden_states = hidden_states + pos_embed
        return hidden_states


class AIMv2TextEmbeddings(CLIPTextEmbeddings):
    pass


def eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    if head_mask is not None:
        attn_weights = attn_weights + head_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Only apply attention dropout during training.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class AIMv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AIMv2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.proj_drop = nn.Dropout(config.projection_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            head_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.proj_out(attn_output)
        attn_output = self.proj_drop(attn_output)

        output = (attn_output, attn_weights) if output_attentions else (attn_output, None)

        return output


class AIMv2EncoderLayer(nn.Module):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__()
        self.attention = AIMv2Attention(config)
        self.ffn = AIMv2SwiGLUFFN(config)
        self.rms_norm1 = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states = self.rms_norm1(hidden_states)
        attn_output, attn_wights = self.attention(
            hidden_states=norm_hidden_states, head_mask=head_mask, output_attentions=output_attentions
        )

        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.rms_norm2(hidden_states)
        mlp_output = self.ffn(norm_hidden_states)

        hidden_states = hidden_states + mlp_output
        return (hidden_states, attn_wights) if output_attentions else (hidden_states, None)


class AIMv2Encoder(SiglipEncoder):
    pass


class AIMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AIMv2Config
    base_model_prefix = "aimv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AIMv2SwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class AIMv2VisionModel(AIMv2PreTrainedModel):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = AIMv2VisionEmbeddings(config)
        self.encoder = AIMv2Encoder(config)
        self.rms_norm = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AIMv2TextModel(AIMv2PreTrainedModel):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = AIMv2TextEmbeddings(config)
        self.encoder = AIMv2Encoder(config)
        # Here comes the eos extract class
        self.head = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AIMv2Model(CLIPModel, nn.Module):
    def __init__(self, config: AIMv2Config):
        nn.Module().__init__()

        if not isinstance(config.text_config, AIMv2VisionConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, AIMv2VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = AIMv2TextModel._from_config(text_config)

        self.vision_model = AIMv2VisionModel._from_config(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # Verify whether it's working right or not.
        logit_scale_tensor = torch.tensor(self.config.logit_scale_init_value)
        self.log_logit_scale = nn.Parameter(torch.log(logit_scale_tensor))

        self.max_log_logit_scale = math.log(config.max_logit_scale)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AIMv2Output]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        logit_scale = self.log_logit_scale.clamp(0.0, self.max_log_logit_scale).exp()
        logits_per_text = text_embeds * logit_scale.exp().to(text_embeds.device)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logits_per_image = logits_per_text.t()

        loss = None
        # if return_loss:
        # Use the loss used in aimv2
        # loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return AIMv2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = ["AIMv2Config", "AIMv2VisionConfig", "AIMv2TextConfig", "AIMv2VisionModel", "AIMv2Model"]
