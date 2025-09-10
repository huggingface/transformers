# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ..aimv2.modeling_aimv2 import Aimv2Attention, Aimv2EncoderLayer
from ..auto import AutoModel
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel
from ..llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, LlavaNextModelOutputWithPast
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionEmbeddings
from .configuration_ovis2 import Ovis2Config, Ovis2VisionConfig


def hard_softmax(logits: torch.Tensor, dim: int):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


class Ovis2ModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class Ovis2CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class Ovis2RMSNorm(LlamaRMSNorm):
    pass


class Ovis2VisionMLP(LlamaMLP):
    pass


class Ovis2VisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def interpolate_pos_encoding(self):
        raise NotImplementedError("Not needed for Ovis2")

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = self.rms_norm(embeddings)

        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class Ovis2VisionAttention(Aimv2Attention):
    pass


class Ovis2VisionEncoderLayer(Aimv2EncoderLayer):
    pass


class Ovis2VisionEncoder(SiglipEncoder):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Ovis2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class Ovis2VisionTransformer(nn.Module):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Ovis2VisionEmbeddings(config)
        self.encoder = Ovis2VisionEncoder(config)
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Ovis2VisualEmbeddingTable(nn.Embedding):
    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)


class Ovis2PreTrainedModel(PreTrainedModel):
    config: Ovis2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ovis2VisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class Ovis2VisionModel(Ovis2PreTrainedModel):
    config: Ovis2VisionConfig

    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.config = config
        self.transformer = Ovis2VisionTransformer(config)
        self.num_visual_indicator_tokens = config.num_visual_indicator_tokens
        self.vocab_size = config.vocab_size
        self.head_linear = nn.Linear(
            config.hidden_size * config.hidden_stride * config.hidden_stride,
            self.vocab_size - self.num_visual_indicator_tokens,
            bias=False,
        )
        self.head_norm = nn.LayerNorm(self.vocab_size - self.num_visual_indicator_tokens)

    def forward(self, pixel_values: torch.FloatTensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(pixel_values)
        last_hidden_state = outputs.last_hidden_state

        if self.config.hidden_stride > 1:
            num_images, seq_len, hidden_dim = last_hidden_state.shape
            hidden_stride = self.config.hidden_stride

            sqrt_l = int(math.sqrt(seq_len))
            if sqrt_l * sqrt_l != seq_len:
                raise ValueError("Token sequence length must be a perfect square")

            pad_size = (hidden_stride - (sqrt_l % hidden_stride)) % hidden_stride
            last_hidden_state = nn.functional.pad(last_hidden_state, (0, 0, 0, pad_size, 0, pad_size), "constant", 0)
            sqrt_l += pad_size

            last_hidden_state = last_hidden_state.reshape(
                num_images, sqrt_l // hidden_stride, hidden_stride, sqrt_l // hidden_stride, hidden_stride, hidden_dim
            )
            last_hidden_state = last_hidden_state.permute(0, 1, 3, 2, 4, 5)
            last_hidden_state = last_hidden_state.reshape(
                num_images, -1, hidden_stride * hidden_stride * hidden_dim
            )  # (n, (sqrt_l//hs)^2, hs^2*d)

        logits = self.head_linear(last_hidden_state)
        logits = self.head_norm(logits)

        if self.config.tokenize_function == "gumbel_argmax":
            prob_token = nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
        elif self.config.tokenize_function == "st_argmax":
            prob_token = hard_softmax(logits, dim=-1)
        elif self.config.tokenize_function == "softmax":
            prob_token = nn.functional.softmax(logits, dim=-1)

        return prob_token


class Ovis2Model(LlavaModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: Ovis2Config):
        super().__init__(config)
        self.vision_tower = Ovis2VisionModel(config.vision_config)
        self.visual_embeddings_table = Ovis2VisualEmbeddingTable(config.vision_config.vocab_size, config.hidden_size)

        self.visual_vocab_size = config.vision_config.vocab_size
        self.vocab_size = config.vocab_size
        self.visual_indicator_token_ids = config.visual_indicator_token_ids
        self.language_model = AutoModel.from_config(config.text_config)
        del self.multi_modal_projector

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        image_features = self.vision_tower(pixel_values)
        batch_size, img_seq_len, _ = image_features.shape
        padding_tensor = torch.zeros(
            (batch_size, img_seq_len, self.vision_tower.num_visual_indicator_tokens),
            dtype=image_features.dtype,
            device=image_features.device,
            requires_grad=False,
            layout=image_features.layout,
        )
        image_features = torch.cat([image_features, padding_tensor], dim=2)
        image_features = self.visual_embeddings_table(image_features)

        visual_indicator = torch.arange(
            self.visual_vocab_size - self.vision_tower.num_visual_indicator_tokens,
            self.visual_vocab_size,
            dtype=torch.long,
        ).to(image_features.device)
        visual_indicator_features = self.visual_embeddings_table(visual_indicator)

        return image_features, visual_indicator_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Ovis2ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features, visual_indicator_features = self.get_image_features(pixel_values=pixel_values)

            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            for i, visual_indicator_id in enumerate(self.visual_indicator_token_ids):
                if input_ids is None:
                    mask = inputs_embeds == self.get_input_embeddings()(
                        torch.tensor(visual_indicator_id, dtype=torch.long, device=inputs_embeds.device)
                    )
                    mask = mask.all(-1)
                else:
                    mask = (input_ids == visual_indicator_id).to(inputs_embeds.device)

                if mask.any():
                    inputs_embeds[mask] = (
                        visual_indicator_features[i]
                        .expand_as(inputs_embeds[mask])
                        .to(inputs_embeds.device, inputs_embeds.dtype)
                    )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return Ovis2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@auto_docstring
class Ovis2ForConditionalGeneration(LlavaForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: Ovis2Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def multi_modal_projector(self):
        raise AttributeError("Not needed for Ovis2")

    def get_image_features(self, pixel_values: torch.FloatTensor):
        return self.model.get_image_features(pixel_values=pixel_values)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Ovis2CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Ovis2ForConditionalGeneration

        >>> model = Ovis2ForConditionalGeneration.from_pretrained("thisisiron/Ovis2-2B-hf")
        >>> processor = AutoProcessor.from_pretrained("thisisiron/Ovis2-2B-hf")

        >>> prompt = "<|im_start|>user\n<image>\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n"
        >>> url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        "user\n\nDescribe the image.\nassistant\nThe image features a brown dog standing on a wooden floor, looking up with"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
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

        return Ovis2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = ["Ovis2PreTrainedModel", "Ovis2Model", "Ovis2ForConditionalGeneration"]
