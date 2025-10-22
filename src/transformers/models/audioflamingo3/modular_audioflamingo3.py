# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

"""PyTorch AudioFlamingo3 model."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...masking_utils import eager_mask, padding_mask_function
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging, can_return_tuple
from ..auto import AutoModelForCausalLM
from ..whisper.modeling_whisper import WhisperAttention, WhisperEncoderLayer
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
from .configuration_audioflamingo3 import AudioFlamingo3Config, AudioFlamingo3EncoderConfig


logger = logging.get_logger(__name__)



class AudioFlamingo3Attention(WhisperAttention):
    pass


class AudioFlamingo3EncoderLayer(WhisperEncoderLayer):
    pass


@auto_docstring
class AudioFlamingo3PreTrainedModel(PreTrainedModel):
    """
    Base class with common functionality for AudioFlamingo3 models.
    """

    config_class = AudioFlamingo3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AudioFlamingo3Attention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        # Initialize modules following config.init_std; used for fine-tuning/inference scaffolding.
        std = getattr(self.config, "init_std", None)
        if std is None and hasattr(self.config, "audio_config"):
            std = getattr(self.config.audio_config, "init_std", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@auto_docstring(
    custom_intro="""
    The audio model from AudioFlamingo3 without any head or projection on top.
    """
)
class AudioFlamingo3Encoder(Qwen2AudioEncoder):
    """
    Audio encoder: Whisper conv front-end, Transformer encoder, average pool (time/2), then LayerNorm.

    Expects `attention_mask` to be `None` or a 4D mask `(B, 1, S, S)` on the *pre-pool* time axis with `-inf` on pads.
    """

    config: AudioFlamingo3EncoderConfig
    _no_split_modules = ["AudioFlamingo3EncoderLayer"]


    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[BaseModelOutput, tuple]:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Log-Mel features extracted from raw audio. Use the processor/feature extractor to compute and pad
                these features from waveform input.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, S, S)`, *optional*):
                Pre-pool encoder attention mask on the time axis. Provide `0` on valid positions and `-inf` on
                padded positions (added to attention logits). If `None`, full attention is used. Here `S` is the
                sequence length after the conv front-end (typically `ceil(T_mel/2)`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """

        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )

        # Conv front-end
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (B, C, T')

        # Add positions, dropout
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (B, S_in, C)
        # TODO (ebezzam) can `self.embed_positions.weight` be used?` 
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        pos = self.embed_positions(positions).squeeze(0)
        if pos.shape[0] < inputs_embeds.shape[1]:
            raise ValueError(f"embed_positions shorter than sequence length: {pos.shape[0]} < {inputs_embeds.shape[1]}")
        hidden_states = nn.functional.dropout(inputs_embeds + pos[: inputs_embeds.shape[1]], p=self.dropout, training=self.training)

        # Transformer stack
        hs_list = [] if output_hidden_states else None
        attn_list = [] if output_attentions else None
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                hs_list.append(hidden_states)
            to_drop = self.training and (torch.rand([]) < self.layerdrop)
            if to_drop:
                out = (hidden_states, None)
            else:
                out = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = out[0]
            if output_attentions:
                attn_list.append(out[1])

        # AvgPool (time/2) + LayerNorm
        prepool = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)  # (B, S_out, C)
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            hs_list.append(prepool)
            hs_list.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(hs_list) if hs_list is not None else None,
            attentions=tuple(attn_list) if attn_list is not None else None,
        )
    

class AudioFlamingo3MultiModalProjector(nn.Module):
    """
    Audio adaptor (a small MLP) that projects AudioFlamingo3Encoder (AF-Whisper)
    features to the LLM embedding space so they can replace `<sound>` tokens.
    """
    def __init__(self, config: AudioFlamingo3Config):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.intermediate_size, config.text_config.hidden_size, bias=config.projector_bias)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=config.projector_bias)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The AudioFlamingo3 model which consists of a audio backbone and a language model.
    """
)
class AudioFlamingo3ForConditionalGeneration(AudioFlamingo3PreTrainedModel, GenerationMixin):
    """
    AudioFlamingo3 model composed of an audio encoder, a projection to the LM hidden size, and a causal LM.

    The audio-text fusion is performed by *replacing* occurrences of the `<sound>` token with per-frame audio embeddings,
    without changing sequence length. The number of `<sound>` tokens is expected to match the *post-pool* frame count
    computed by the processor.
    """

    config_class = AudioFlamingo3Config

    def __init__(self, config: AudioFlamingo3Config):
        super().__init__(config)
        # Language model
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        # Audio encoder (explicitly instantiate our class to guarantee helper availability)
        self.audio_tower = AudioFlamingo3Encoder(config.audio_config)
        # Projection to LM hidden size
        self.multi_modal_projector = AudioFlamingo3MultiModalProjector(config)

        self.post_init()

    # --- Embedding plumbing (forward to LM) ---
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        # TODO (ebezzam) why unused? normally passed to self.language_model
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        r"""
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        >>> MODEL_ID = "nvidia/audio-flamingo-3"
        >>> processor = AutoProcessor.from_pretrained(MODEL_ID)
        >>> model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto").eval()

        >>> conversations = [
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {"type": "text", "text": "Transcribe the input speech."},
        >>>                 {"type": "audio", "path": "audio_1.wav"},
        >>>             ],
        >>>         }
        >>>     ],
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {"type": "text", "text": "Describe the song."},
        >>>                 {"type": "audio", "path": "audio_2.wav"},
        >>>             ],
        >>>         }
        >>>     ]
        >>> ]

        >>> batch = processor.apply_chat_template(
        >>>     conversations,
        >>>     tokenize=True,
        >>>     add_generation_prompt=True,
        >>>     sampling_rate=getattr(processor.feature_extractor, "sampling_rate", 16000),
        >>> ).to(model.device)

        >>> gen_ids = model.generate(**batch, max_new_tokens=512)

        >>> inp_len = batch["input_ids"].shape[1]
        >>> new_tokens = gen_ids[:, inp_len:]
        >>> texts = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(texts)
        ["Transcription of the input speech: Good morning everyone...", "The song is an orchestral piece..."]
        ```"""

        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )

        # Text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Replace <sound> token slots with audio features (no length change)
        if input_features is not None and input_ids is not None and input_ids.shape[1] != 1:
            if feature_attention_mask is None:
                raise ValueError("`feature_attention_mask` is required when `input_features` is provided.")

            # Compute pre/post lengths (mel -> conv -> pool)
            Lmel = feature_attention_mask.sum(-1).to(dtype=torch.long)  # (#windows,)

            pre_lengths = (Lmel - 1) // 2 + 1
            post_lengths = (pre_lengths - 2) // 2 + 1

            # Build 4D encoder mask on pre-pool axis with -inf on pads using masking_utils
            time_dim = input_features.shape[-1]
            # Construct a (B, T_mel_max) boolean validity mask from measured mel lengths
            mask_1d = torch.arange(time_dim, device=input_features.device).unsqueeze(0) < Lmel.unsqueeze(1)
            
            # TODO (ebezzam) move `_build_square_attn_mask` here since only used once, 
            # -- but can probably still be simplified
            # Convert (B, T_mel) frame-validity mask to Whisper's 4D square mask (B, 1, S, S) with -inf on pads
            audio_feat_lengths = ((mask_1d.sum(-1).to(torch.long) - 1) // 2) + 1
            batch_size = mask_1d.shape[0]
            # Sequence length after conv2 (stride=2, kernel=3, pad=1)
            seq_len = (time_dim - 1) // 2 + 1
            
            # 2D padding mask on the downsampled timeline: True => keep, False => pad
            seq = torch.arange(seq_len, device=mask_1d.device).unsqueeze(0).expand(batch_size, seq_len)
            padding_mask = seq < audio_feat_lengths.unsqueeze(1)
            
            # Build 4D float mask (B, 1, S, S) with 0 on valid, -inf on pads
            mask_fn = padding_mask_function(padding_mask)
            cache_position = torch.arange(seq_len, device=mask_1d.device)
            enc_mask = eager_mask(
                batch_size=batch_size,
                cache_position=cache_position,
                kv_length=seq_len,
                mask_function=mask_fn,
                dtype=self.audio_tower.conv1.weight.dtype,
            )
            # TODO (ebezzam) end of `_build_square_attn_mask`

            # Encode audio -> project -> flatten valid frames
            enc_out = self.audio_tower(input_features, attention_mask=enc_mask)
            post = enc_out.last_hidden_state  # (#windows, seq_len_max, C)
            audio_feats = self.multi_modal_projector(post)  # (#windows, seq_len_max, hidden_size)

            _, seq_len_max, hidden_size = audio_feats.shape
            valid_mask = torch.arange(seq_len_max, device=post_lengths.device)[None, :] < post_lengths[:, None]
            flat_audio = audio_feats[valid_mask]  # (sum(post_lengths), hidden_size)

            # --- Scatter into <sound> slots ---
            # Build a boolean mask over token positions where we should inject audio frames
            special_ids_mask = input_ids == self.config.audio_token_id  # (B, L)
            # Never treat padding as content.
            if attention_mask is not None:
                special_ids_mask = special_ids_mask & attention_mask.to(torch.bool)
            n_audio_tokens = int(special_ids_mask.sum().item())
            n_audio_frames = int(flat_audio.shape[0])
            if n_audio_tokens != n_audio_frames:
                raise ValueError(
                    f"Audio tokens and features mismatch: tokens={n_audio_tokens}, frames={n_audio_frames}. "
                    "Ensure the processor expands <sound> by the post-pool frame count."
                )

            # Expand mask to embedding dimension and scatter the flattened audio features
            special_mask = special_ids_mask.unsqueeze(-1).expand(-1, -1, hidden_size)  # (B, L, D)
            src = flat_audio.to(inputs_embeds.device, dtype=inputs_embeds.dtype).reshape(-1)  # (n_audio_tokens * D,)
            inputs_embeds = inputs_embeds.masked_scatter(special_mask, src)

        # Language model forward
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Pass `input_features`/`feature_attention_mask` only on the first step of generation.
        """
        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        is_first = model_inputs.get("past_key_values", None) is None or (
            isinstance(model_inputs.get("cache_position", None), torch.Tensor)
            and model_inputs["cache_position"].numel() > 0
            and int(model_inputs["cache_position"][0].item()) == 0
        )
        if is_first:
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if feature_attention_mask is not None:
                model_inputs["feature_attention_mask"] = feature_attention_mask
        return model_inputs


__all__ = ["AudioFlamingo3ForConditionalGeneration", "AudioFlamingo3PreTrainedModel", "AudioFlamingo3Encoder"]
