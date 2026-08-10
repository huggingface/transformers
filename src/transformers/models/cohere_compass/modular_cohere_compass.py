# Copyright 2026 Cohere Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch CohereCompass model."""

from collections.abc import Callable

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GenericForSequenceClassification
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import maybe_autocast
from ..cohere2.configuration_cohere2 import Cohere2Config
from ..cohere2.modeling_cohere2 import (
    Cohere2Attention,
    Cohere2DecoderLayer,
    Cohere2ForCausalLM,
    Cohere2LayerNorm,
    Cohere2MLP,
    eager_attention_forward,
)
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ..qwen3.modeling_qwen3 import apply_rotary_pos_emb
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from ..qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


@auto_docstring(checkpoint="CohereLabs/North-Micro-Vision-Instruct")
@strict
class CohereCompassVisionConfig(Qwen3VLVisionConfig):
    model_type = "cohere_compass_vision"
    base_config_key = "vision_config"


@auto_docstring(checkpoint="CohereLabs/North-Micro-Vision-Instruct")
@strict
class CohereCompassTextConfig(Cohere2Config):
    r"""
    logit_scale (`float`, *optional*):
        Scale applied to language-model logits.
    pooling (`str`, *optional*):
        The pooling strategy (`bos` | `eos` | `mean`); `None` defaults to `eos`.
    """

    model_type = "cohere_compass_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    rope_parameters: dict[str, RopeParameters | dict | float | str | int | None] | None = None
    logit_scale: float | None = None
    pooling: str | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="CohereLabs/North-Micro-Vision-Instruct")
@strict
class CohereCompassConfig(Qwen3VLConfig):
    model_type = "cohere_compass"
    sub_configs = {
        "text_config": CohereCompassTextConfig,
        "vision_config": CohereCompassVisionConfig,
    }

    image_token_id: int = 255031
    video_token_id: int = 255032
    vision_start_token_id: int = 255028
    vision_end_token_id: int = 255029


# Overwritten to show type as cohere_compass for internal compatibility
class CohereCompassImageProcessor(Qwen2VLImageProcessor):
    pass


# Overwritten to show type as cohere_compass for internal compatibility
class CohereCompassImageProcessorPil(Qwen2VLImageProcessorPil):
    pass


# Overwritten to show type as cohere_compass for internal compatibility
class CohereCompassVideoProcessor(Qwen3VLVideoProcessor):
    pass


# Overwritten to show type as cohere_compass for internal compatibility
@auto_docstring
class CohereCompassProcessor(Qwen3VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.image_token = "<|IMAGE_PAD|>"
        self.video_token = "<|VIDEO_PAD|>"
        self.vision_start_token = "<|VISION_START|>"
        self.vision_end_token = "<|VISION_END|>"


class CohereCompassRotaryEmbedding(Gemma3RotaryEmbedding):
    """Gemma3 rotary embedding adapted to Compass's multi-axis position IDs."""

    # Copied from: Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope, added supported for layer_types
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        # In contrast to other models, Cohere has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            mrope_section = self.config.rope_parameters[layer_type].get("mrope_section", [24, 20, 20])
            freqs = self.apply_interleaved_mrope(freqs, mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    # Copied from: Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope
    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class CohereCompassMLP(Cohere2MLP):
    pass


class CohereCompassLayerNorm(Cohere2LayerNorm):
    pass


class CohereCompassAttention(Cohere2Attention):
    def __init__(self, config: CohereCompassTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CohereCompassDecoderLayer(Cohere2DecoderLayer):
    def __init__(self, config: CohereCompassTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)


@auto_docstring
class CohereCompassPreTrainedModel(Qwen3VLPreTrainedModel):
    _no_split_modules = [
        "CohereCompassDecoderLayer",
    ]
    _can_record_outputs = {
        "hidden_states": CohereCompassDecoderLayer,
        "attentions": CohereCompassAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CohereCompassRotaryEmbedding):
            for layer_type in module.rope_type:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class CohereCompassTextModel(Qwen3VLTextModel, CohereCompassPreTrainedModel):
    """Unified text decoder. Plain (2D) for text-only checkpoints; 3D mrope + DeepStack for VL."""

    config: CohereCompassTextConfig

    def __init__(self, config: CohereCompassTextConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [CohereCompassDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CohereCompassLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)

        self.rotary_emb = CohereCompassRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: (
                self.rotary_emb(hidden_states, position_ids, layer_type)
                if self.config.rope_parameters[layer_type] is not None
                else None
            )
            for layer_type in dict.fromkeys(self.config.layer_types)
        }

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.config.layer_types[layer_idx]],
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class CohereCompassForCausalLM(Cohere2ForCausalLM, CohereCompassPreTrainedModel):
    config: CohereCompassTextConfig

    def __init__(self, config: CohereCompassTextConfig):
        super().__init__(config)
        self.model = CohereCompassTextModel(config)
        self.logit_scale = config.logit_scale if config.logit_scale is not None else 1.0


# Overwritten to show type as cohere_compass_vision for internal compatibility
class CohereCompassVisionModel(Qwen3VLVisionModel):
    config: CohereCompassVisionConfig
    input_modalities = ("image",)


@auto_docstring
class CohereCompassModel(Qwen3VLModel):
    """Vision-language base model that fuses the vision tower into the decoder and applies DeepStack residuals."""

    def __init__(self, config: CohereCompassConfig):
        super().__init__(config)


@auto_docstring(
    custom_intro="""
    The CohereCompass model with a language modeling head, for image-text-to-text generation.
    """
)
class CohereCompassForConditionalGeneration(Qwen3VLForConditionalGeneration, CohereCompassPreTrainedModel):
    def __init__(self, config: CohereCompassConfig):
        super().__init__(config)
        self.logit_scale = config.text_config.logit_scale

    @can_return_tuple
    @auto_docstring(checkpoint="CohereLabs/North-Micro-Vision-Instruct")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""

        Example:

        ```python
        >>> import torch
        >>> from transformers import CohereCompassForConditionalGeneration, CohereCompassProcessor

        >>> model_id = "CohereLabs/North-Micro-Vision-Instruct"

        >>> processor = CohereCompassProcessor.from_pretrained(model_id)
        >>> model = CohereCompassForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
        )

        >>> image_url = "https://cdn-uploads.huggingface.co/production/uploads/66d732effe6684fc16b12c28/Io_5OCmftsmH-n158ZtPs.png"
        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        >>> outputs = model.generate(
            **inputs,
            max_new_tokens=128,
        )

        >>> input_length = inputs["input_ids"].shape[-1]
        >>> response = processor.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        >>> print(response)
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.logit_scale is not None:
            logits = logits * self.logit_scale

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
        )


@auto_docstring(
    custom_intro="""
    The CohereCompass model with a sequence classification head on top.
    """
)
class CohereCompassTextForSequenceClassification(GenericForSequenceClassification, CohereCompassPreTrainedModel):
    r"""Sequence classification head.

    Projects every position with ``score`` then pools according to ``config.text_config.pooling``:

    - ``eos`` (default when ``pooling`` is ``None``): the rightmost non-pad token.
    - ``bos``: the first token. Meaningful only when the backbone is bidirectional.
    - ``mean``: masked mean over non-pad tokens.
    """

    config: CohereCompassTextConfig

    def __init__(self, config: CohereCompassTextConfig):
        super().__init__(config)
        self.pooling = config.pooling or "eos"
        if self.pooling not in {"bos", "eos", "mean"}:
            raise ValueError(f"Unsupported pooling {self.pooling!r}; expected one of bos|eos|mean.")

    def _non_pad_mask(self, input_ids, attention_mask, ref):
        """Boolean non-pad mask ``[batch, seq]`` from ``attention_mask`` (preferred) or ``pad_token_id``."""
        if attention_mask is not None:
            return attention_mask.to(ref.device, torch.bool)
        if input_ids is not None and self.config.pad_token_id is not None:
            return (input_ids != self.config.pad_token_id).to(ref.device, torch.bool)
        return None

    # Overwritten: We want to support multiple pooling strategies
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        if self.pooling == "eos":
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.score(hidden_states)

        if self.pooling == "bos":
            pooled_logits = logits[:, 0, :]
        else:
            non_pad_mask = self._non_pad_mask(input_ids, attention_mask, logits)
            if non_pad_mask is None:
                pooled_logits = logits.mean(dim=1)
            else:
                m = non_pad_mask.to(logits.dtype).unsqueeze(-1)
                pooled_logits = (logits * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "CohereCompassConfig",
    "CohereCompassTextConfig",
    "CohereCompassVisionConfig",
    "CohereCompassImageProcessor",
    "CohereCompassImageProcessorPil",
    "CohereCompassVideoProcessor",
    "CohereCompassProcessor",
    "CohereCompassPreTrainedModel",
    "CohereCompassTextModel",
    "CohereCompassForCausalLM",
    "CohereCompassVisionModel",
    "CohereCompassModel",
    "CohereCompassForConditionalGeneration",
    "CohereCompassTextForSequenceClassification",
    "CohereCompassRotaryEmbedding",
]
