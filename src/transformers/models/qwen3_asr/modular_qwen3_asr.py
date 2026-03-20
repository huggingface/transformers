# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import re

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, can_return_tuple
from ..qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTextConfig,
)
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextMLP,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    SinusoidsPositionEmbedding,
    _get_feat_extract_output_lengths,
)


@auto_docstring(checkpoint="bezzam/Qwen3-ASR-1.7B")
@strict(accept_kwargs=True)
class Qwen3ASRAudioEncoderConfig(Qwen3OmniMoeAudioEncoderConfig):
    r"""
    downsample_hidden_size ( `int`, *optional*, defaults to `480`): Hidden size in donwsampling layer
    conv_chunksize ( `int`, *optional*, defaults to `500`): Chunk size of each input to convolutional layer
    n_window_infer ( `int`, *optional*, defaults to `800`): Number of windows during inference
    max_source_positions (`int`, *optional*, defaults to 1500): Maximum sequence length for the inputs
    n_window (`int`, *optional*, defaults to 50):  Number of windwos
    output_dim (`int`, *optional*, defaults to 2048):  Dimensionality of the output
    """

    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    n_window: int = 50
    output_dim: int = 2048
    n_window_infer: int = 800


@auto_docstring(checkpoint="bezzam/Qwen3-ASR-1.7B")
@strict(accept_kwargs=True)
class Qwen3ASRTextConfig(Qwen3OmniMoeTextConfig):
    """
    Example:

    ```python
    >>> from transformers import Qwen3ASRTextModel, Qwen3ASRTextConfig

    >>> # Initializing a Qwen3ASRText style configuration
    >>> configuration = Qwen3ASRTextConfig()

    >>> # Initializing a model
    >>> model = Qwen3ASRTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = 151936
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = True

    # Remove MoE-specific attributes from parent
    decoder_sparse_step = AttributeError()
    moe_intermediate_size = AttributeError()
    num_experts_per_tok = AttributeError()
    num_experts = AttributeError()
    norm_topk_prob = AttributeError()
    output_router_logits = AttributeError()
    router_aux_loss_coef = AttributeError()
    sliding_window = AttributeError()


@auto_docstring(checkpoint="bezzam/Qwen3-ASR-1.7B")
@strict(accept_kwargs=True)
class Qwen3ASRConfig(PreTrainedConfig):
    r"""
    audio_token_id (`int`, *optional*, defaults to 151676):
        The audio token id to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRConfig

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr"
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151676
    pad_token_id: int = 151645
    eos_token_id: list[int] | tuple[int, ...] | int = (151643, 151645)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.audio_config is None:
            self.audio_config = Qwen3ASRAudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = Qwen3ASRAudioEncoderConfig(**self.audio_config)

        if self.text_config is None:
            self.text_config = Qwen3ASRTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Qwen3ASRTextConfig(**self.text_config)

        super().__post_init__(**kwargs)


class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class Qwen3ASRProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3ASR processor.
    [`Qwen3ASRProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3ASRProcessor.__call__`] and [`~Qwen3ASRProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_token = self.tokenizer.audio_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

    # TODO (ebezzam) could use modular from VibeVoice ASR, if we define a method `_get_feat_extract_output_lengths` for it
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | list[TextInput],
        output_labels: bool | None = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.
        """
        call_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        audio = make_list_of_audio(audio)
        if not isinstance(text, list):
            text = [text]
        if len(text) != len(audio):
            raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

        # Prepare audio
        data = self.feature_extractor(audio, **audio_kwargs)
        data["input_features_mask"] = data.pop("attention_mask")

        # Replace audio tokens in text
        audio_lengths = _get_feat_extract_output_lengths(data["input_features_mask"].sum(-1)).cpu().numpy()
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for i, num_tokens in enumerate(audio_lengths):
            text[i] = audio_token_pattern.sub(self.audio_token * int(num_tokens), text[i])

        # Prepare text
        texts_inputs = self.tokenizer(text, **text_kwargs)
        data.update(texts_inputs)

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["input_features_mask"]))


class Qwen3ASRRMSNorm(Qwen3OmniMoeThinkerTextRMSNorm):
    pass


class Qwen3ASRAttention(Qwen3OmniMoeThinkerTextAttention):
    pass


class Qwen3ASRMLP(Qwen3OmniMoeThinkerTextMLP):
    pass


class Qwen3ASRThinkerTextDecoderLayer(Qwen3OmniMoeThinkerTextDecoderLayer):
    def __init__(self, config: Qwen3ASRTextConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3ASRAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRMLP(config)
        self.input_layernorm = Qwen3ASRRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@auto_docstring
class Qwen3ASRPreTrainedModel(PreTrainedModel):
    config: Qwen3ASRConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer", "Qwen3ASRThinkerTextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"attentions": Qwen3ASRAttention}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, SinusoidsPositionEmbedding):
            log_timescale_increment = np.log(module.max_timescale) / (module.channels // 2 - 1)
            inv_timescales = torch.exp(-log_timescale_increment * torch.arange(module.channels // 2).float())
            scaled_time = torch.arange(module.length)[:, None] * inv_timescales[None, :]

            init.copy_(
                module.positional_embedding,
                torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            )


class Qwen3ASRAudioEncoder(Qwen3OmniMoeAudioEncoder):
    pass


class Qwen3ASRThinkerTextRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding):
    def __init__(self, config: Qwen3ASRTextConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_parameters["rope_type"]
        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])


class Qwen3ASRThinkerTextMLP(Qwen3OmniMoeThinkerTextMLP):
    pass


class Qwen3ASRThinkerTextRMSNorm(Qwen3OmniMoeThinkerTextRMSNorm):
    pass


class Qwen3ASRThinkerTextAttention(Qwen3OmniMoeThinkerTextAttention):
    pass


@auto_docstring(custom_intro=("Text part of Qwen3ASRThinker, "))
class Qwen3ASRThinkerTextModel(Qwen3OmniMoeThinkerTextModel):
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRThinkerTextAttention,
    }

    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__(config)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(self, *args, **kwargs):
        raise NotImplementedError("Not needed")


@auto_docstring(
    custom_intro="""
    The Qwen3ASR model which consists of an audio backbone and a language model.
    """
)
class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    config_class = Qwen3ASRConfig
    _no_split_modules = ["Qwen3ASRAudioEncoder", "Qwen3ASRThinkerTextDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRThinkerTextAttention,
    }

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        # TODO use AutoModel? at least for audio encoder
        self.audio_tower = Qwen3ASRAudioEncoder(config.audio_config)
        # TODO possible to use Qwen3ForCausalLM via AutoModelForCausalLM? for both text model and LM head
        self.model = Qwen3ASRThinkerTextModel(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = (
            self.config.text_config.pad_token_id if self.config.text_config.pad_token_id is not None else -1
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_rope_index(
        self,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the rope index in LLM.

        Args:
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            input_features_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if input_features_mask is not None:
            audio_feature_lengths = torch.sum(input_features_mask, dim=1)
        else:
            audio_feature_lengths = None
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else input_features_mask.sum(-1)

        # audio encoder do not support batch inference to keep precision
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            audio_feature = audio_output.last_hidden_state
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=0)

        return audio_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text, audios
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                input_features_mask=input_features_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(self, *args, is_first_iteration=False, **kwargs):
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        model_inputs["position_ids"] = None

        if is_first_iteration:
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask

        return model_inputs


__all__ = [
    "Qwen3ASRAudioEncoderConfig",
    "Qwen3ASRTextConfig",
    "Qwen3ASRConfig",
    "Qwen3ASRProcessor",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ASRAudioEncoder",
]
