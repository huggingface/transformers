# Copyright 2025 the HuggingFace Team. All rights reserved.
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

import numpy as np

from ...activations import ACT2FN
from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, is_torch_available, logging
from ...utils.generic import can_return_tuple, check_model_inputs
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward, rotate_half
from .configuration_glmasr import GlmAsrConfig, GlmAsrEncoderConfig


if is_torch_available():
    import torch
    from torch import nn


logger = logging.get_logger(__name__)


class GlmAsrProcessorKwargs(AudioFlamingo3ProcessorKwargs): ...


class GlmAsrProcessor(AudioFlamingo3Processor):
    r"""
    Constructs an GlmAsr processor which wraps an GlmAsr feature extractor and an GlmAsr
    tokenizer into a single processor.

    [`GlmAsrProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`]. See the [`~GlmAsrProcessor.__call__`] for more information.

    Args:
            feature_extractor ([`WhisperFeatureExtractor`]):
                The feature extractor is a required input.
            tokenizer ([`Qwen2TokenizerFast`]):
                The tokenizer is a required input.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's default chat
                template will be used.
            audio_token (`Optional[str]`, *optional*, defaults to `"<|pad|>`"):
                Special token used to represent audio inputs in the chat template.
            default_transcription_prompt (`str`, *optional*, defaults to `"Please transcribe this audio into text"`):
                Default prompt to use for transcription tasks when applying transcription requests.
            max_audio_len (`int`, *optional*, defaults to 655):
                Maximum length of audio sequences in seconds. Audio longer than this will be truncated.
                655 gives approximately 8192 tokens, corresponding to the maximum sequence length of the text model.
    """

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|pad|>",
        default_transcription_prompt="Please transcribe this audio into text",
        max_audio_len=655,
    ):
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            default_transcription_prompt=default_transcription_prompt,
            max_audio_len=max_audio_len,
        )

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        merge_factor = 4
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        num_tokens = (audio_lengths - merge_factor) // merge_factor + 1
        return num_tokens

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[GlmAsrProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses `"Transcribe the input speech."`.
            **kwargs:
                Additional keyword arguments forwarded to [`~AudioFlamingo3Processor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to [`AudioFlamingo3ForConditionalGeneration.generate`].

        """

        if isinstance(audio, str):
            audio_items: list[str | np.ndarray] = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(el, str) for el in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))
            if is_torch_available():
                audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [self.default_transcription_prompt] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(self.default_transcription_prompt)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": audio_item}
                        if isinstance(audio_item, str)
                        else {"type": "audio", "audio": audio_item},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            for prompt_text, audio_item in zip(prompts, audio_items)
        ]

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )


class GlmAsrRotaryEmbedding(GlmRotaryEmbedding): ...


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class GlmAsrAttention(LlamaAttention):
    def __init__(self, config: GlmAsrConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GlmAsrMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GlmAsrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmAsrConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmAsrAttention(config=config, layer_idx=layer_idx)

        self.mlp = GlmAsrMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GlmAsrPreTrainedModel(AudioFlamingo3PreTrainedModel): ...


# TODO: @eustlb, this is what WhisperEncoder should look like
class GlmAsrEncoder(GlmAsrPreTrainedModel):
    config: GlmAsrEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["GlmAsrEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": GlmAsrEncoderLayer,
        "attentions": GlmAsrAttention,
    }

    def __init__(self, config: GlmAsrEncoderConfig):
        super().__init__(config)
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList(
            [GlmAsrEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = GlmAsrRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(self, input_features, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutputWithPooling:
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.transpose(1, 2)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(
            hidden_states, position_ids=torch.arange(hidden_states.shape[1], device=hidden_states.device)[None, :]
        )

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, position_embeddings=position_embeddings, **kwargs)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class GlmAsrMultiModalProjector(AudioFlamingo3MultiModalProjector):
    def __init__(self, config: GlmAsrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.intermediate_size, config.text_config.hidden_size * 2)
        self.linear_2 = nn.Linear(config.text_config.hidden_size * 2, config.text_config.hidden_size)


@auto_docstring(
    custom_intro="""
    The GlmAsr model which consists of a fine-tuned Whisper encoder, a multi-modal projector and a Llama language model.
    """
)
class GlmAsrForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    @can_return_tuple
    @auto_docstring(
        custom_intro="Compute audio embeddings from log-mel input features using the audio encoder and multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        audio_outputs = self.audio_tower(input_features, return_dict=True, **kwargs)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(
            input_features.shape[0], -1, self.config.audio_config.intermediate_size
        )
        audio_embeds = self.multi_modal_projector(audio_hidden_states)

        audio_lengths = input_features_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        merge_factor = 4
        post_lengths = (audio_lengths - merge_factor) // merge_factor + 1

        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_outputs.pooler_output = audio_embeds[valid_mask.to(audio_embeds.device)]

        return audio_outputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import GlmAsrForConditionalGeneration, AutoProcessor

        >>> model_id = "zai-org/GLM-ASR-Nano-2512"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = GlmAsrForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
        >>> inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")

        >>> inputs = inputs.to(model.device, dtype=model.dtype)

        >>> outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        >>> decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        >>> print(decoded_outputs)
        ```"""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = ["GlmAsrEncoder", "GlmAsrForConditionalGeneration", "GlmAsrProcessor", "GlmAsrPreTrainedModel"]
