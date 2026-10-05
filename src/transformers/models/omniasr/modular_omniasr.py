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

from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import CompileConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    CausalLMOutput,
    CausalLMOutputWithPast,
)
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..parakeet.modeling_parakeet import (
    ParakeetCTCGenerateOutput,
    ParakeetEncoderModelOutput,
    ParakeetForCTC,
    ParakeetPreTrainedModel,
)
from ..vibevoice.modeling_vibevoice import VibeVoiceModel
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2EncoderLayer,
    Wav2Vec2LayerNormConvLayer,
)


@auto_docstring(checkpoint="bezzam/omniasr-ctc-300m-v2")
@strict
class OmniASRAudioConfig(PreTrainedConfig):
    r"""
    conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
        A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
        feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
    conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
        length of *conv_kernel* defines the number of convolutional layers and has to match the length of
        *conv_dim*.
    conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
        A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
        of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
    conv_bias (`bool`, *optional*, defaults to `True`):
        Whether the 1D convolutional layers have a bias.
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Number of convolutional positional embeddings. Defines the kernel size of the 1D convolutional positional
        embeddings layer.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of the 1D convolutional positional embeddings layer.

    Example:

    ```python
    >>> from transformers import OmniASRAudioConfig, OmniASRAudioModel

    >>> # Initializing an OmniASR encoder configuration
    >>> configuration = OmniASRAudioConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_audio"
    base_config_key = "audio_config"

    hidden_size: int = 1024
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    intermediate_size: int = 4096
    attention_dropout: float | int = 0.0
    hidden_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        num_conv_layers = len(self.conv_dim)
        if (len(self.conv_stride) != num_conv_layers) or (len(self.conv_kernel) != num_conv_layers):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )


@auto_docstring(checkpoint="bezzam/omniasr-ctc-300m-v2")
@strict
class OmniASRCTCConfig(PreTrainedConfig):
    r"""
    audio_config (`Union[dict, OmniASRAudioConfig]`, *optional*):
        The config object or dictionary of the audio encoder.
    ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
        instance of [`OmniASRForCTC`].
    ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`OmniASRForCTC`].

    Example:

    ```python
    >>> from transformers import OmniASRForCTC, OmniASRCTCConfig

    >>> # Initializing an OmniASR-CTC configuration
    >>> configuration = OmniASRCTCConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRForCTC(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_ctc"
    sub_configs = {"audio_config": OmniASRAudioConfig}

    vocab_size: int = 10288
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = False
    audio_config: dict | PreTrainedConfig | None = None
    bos_token_id: int | None = 0
    pad_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config = OmniASRAudioConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = OmniASRAudioConfig()
        self.initializer_range = self.audio_config.initializer_range
        super().__post_init__(**kwargs)

    @classmethod
    def from_audio_config(cls, audio_config: OmniASRAudioConfig, **kwargs):
        r"""
        Instantiate a [`OmniASRCTCConfig`] (or a derived class) from omniASR audio model configuration.

        Returns:
            [`OmniASRCTCConfig`]: An instance of a configuration object
        """

        return cls(audio_config=audio_config.to_dict(), **kwargs)

    @property
    def hidden_size(self):
        return self.audio_config.hidden_size


@auto_docstring(checkpoint="bezzam/omniasr-llm-300m-v2")
@strict
class OmniASRConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import OmniASRForConditionalGeneration, OmniASRConfig

    >>> # Initializing an OmniASR-LLM configuration
    >>> configuration = OmniASRConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr"
    sub_configs = {"audio_config": OmniASRAudioConfig, "text_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 10289
    bos_token_id: int | None = 0
    pad_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config = OmniASRAudioConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = OmniASRAudioConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](
                vocab_size=11984,
                hidden_size=4096,
                intermediate_size=2816,
                num_hidden_layers=12,
                num_key_value_heads=8,
                rope_theta=10000.0,
                rms_norm_eps=1e-05,
            )

        self.initializer_range = self.audio_config.initializer_range
        super().__post_init__(**kwargs)


# NOTE: Simplified version of Wav2Vec2PositionalConvEmbedding
class OmniASRPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        position_embeddings = hidden_states.transpose(1, 2)
        position_embeddings = self.conv(position_embeddings)
        # Instead of `Wav2Vec2SamePadLayer`, in-line removal of padding
        position_embeddings = position_embeddings[:, :, :-1]
        position_embeddings = self.activation(position_embeddings)
        return position_embeddings.transpose(1, 2)


class OmniASRAttention(Wav2Vec2Attention):
    pass


# NOTE: original: https://github.com/facebookresearch/fairseq2/blob/a1f0c565a99d3cd3e3157678b5c48653e3d439f4/src/fairseq2/models/transformer/encoder_layer.py#L141
class OmniASREncoderLayer(Wav2Vec2EncoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # Move layer norm before attention
        hidden_states, _ = self.attention(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        ffn_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)  # Move layer norm before feed forward
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = ffn_residual + hidden_states  # Add residual

        return hidden_states


class OmniASRLayerNormConvLayer(Wav2Vec2LayerNormConvLayer):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.activation = ACT2FN[config.hidden_act]


# NOTE: similar to `ParakeetEncoderSubsamplingConv2D` but for 1D directly on audio, and a replacement for `Wav2Vec2FeatureEncoder` and `Wav2Vec2FeatureProjection`
class OmniASREncoderSubsamplingConv1D(nn.Module):
    def __init__(self, config: OmniASRAudioConfig):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [OmniASRLayerNormConvLayer(config, layer_id=i) for i in range(len(config.conv_dim))]
        )
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing (like Wav2Vec2FeatureEncoder)
        if self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.layer_norm(hidden_states)
        return self.projection(hidden_states)


@auto_docstring
class OmniASRPreTrainedModel(ParakeetPreTrainedModel):
    main_input_name = "input_values"
    _supports_flash_attn = True
    _no_split_modules = None
    _can_record_outputs = None

    def _init_weights(self, module):
        raise AttributeError("Normal super call")

    def _get_subsampling_output_length(self, input_lengths: torch.LongTensor | int) -> torch.LongTensor | int:
        audio_config = getattr(self.config, "audio_config", self.config)
        lengths = input_lengths
        for kernel_size, stride in zip(audio_config.conv_kernel, audio_config.conv_stride):
            lengths = torch.div(lengths - kernel_size, stride, rounding_mode="floor") + 1

        return lengths


class OmniASREncoderModelOutput(ParakeetEncoderModelOutput):
    pass


@auto_docstring(custom_intro="""Outputs of OmniASR CTC model generation.""")
@dataclass
class OmniASRCTCGenerateOutput(ParakeetCTCGenerateOutput):
    pass


@auto_docstring(
    custom_intro="""
    The OmniASR speech encoder, which is a Wav2Vec2-style encoder.
    """
)
class OmniASRAudioModel(OmniASRPreTrainedModel):
    config: OmniASRAudioConfig
    base_model_prefix = "audio_tower"
    _no_split_modules = ["OmniASREncoderLayer"]
    _can_record_outputs = {
        "attentions": OmniASRAttention,
        "hidden_states": OmniASREncoderLayer,
    }

    def __init__(self, config: OmniASRAudioConfig):
        super().__init__(config)

        self.gradient_checkpointing = False

        self.hidden_dropout = config.hidden_dropout
        self.layerdrop = config.layerdrop

        self.subsampling = OmniASREncoderSubsamplingConv1D(config)
        self.encode_positions = OmniASRPositionalConvEmbedding(config)
        self.layers = nn.ModuleList([OmniASREncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_values: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
        output_attention_mask: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OmniASREncoderModelOutput:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `input_values`.
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the subsampled attention mask. Only effective when `padding_mask` is provided.
        """
        hidden_states = self.subsampling(input_values)

        output_mask = None
        if padding_mask is not None:
            output_mask = self._get_output_attention_mask(padding_mask, target_length=hidden_states.shape[1])
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~output_mask[..., None], 0.0)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=output_mask,
        )

        position_embeddings = self.encode_positions(hidden_states)
        hidden_states = hidden_states + position_embeddings

        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask, **kwargs)

        # Layer norm applied after the layers (Wav2Vec2Encoder applies it before)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)

        return OmniASREncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_attention_mask and output_mask is not None else None,
        )


class OmniASRForCTC(ParakeetForCTC):
    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)
        del self.encoder

    # same as ParakeetForCTC but with `input_values` instead of `input_features` as we use audio values directly
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `input_values`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-ctc-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> outputs = model(**inputs)

        >>> print(outputs.loss)
        ```"""

        if labels is not None:
            kwargs.setdefault("output_attention_mask", True)
        encoder_outputs = self.audio_tower(
            input_values=input_values,
            padding_mask=padding_mask,
            **kwargs,
        )

        hidden_states = encoder_outputs.last_hidden_state
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            encoder_lengths = encoder_outputs.attention_mask.sum(-1)

            # assuming that padded tokens are filled with -100 when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    encoder_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # same as ParakeetForCTC but with `input_values` instead of `input_features` as we use audio values directly
    def generate(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict_in_generate: bool = False,
        compile_config: CompileConfig | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OmniASRCTCGenerateOutput | torch.LongTensor:
        r"""
        compile_config ([`~generation.CompileConfig`], *optional*):
            If provided, `torch.compile` will be applied to the forward calls in the decoding loop.

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-ctc-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        model_forward = self.get_compiled_call(compile_config) if compile_config is not None else self.__call__

        kwargs["return_dict"] = True
        outputs: CausalLMOutput = model_forward(
            input_values=input_values,
            padding_mask=padding_mask,
            **kwargs,
        )

        # greedy decoding
        sequences = outputs.logits.argmax(dim=-1)

        # mask out padded tokens
        if padding_mask is not None:
            output_mask = self._get_output_attention_mask(padding_mask, target_length=sequences.shape[1])
            sequences[~output_mask] = self.config.pad_token_id

        if return_dict_in_generate:
            return OmniASRCTCGenerateOutput(
                sequences=sequences,
                logits=outputs.logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )

        return sequences


@auto_docstring(
    custom_intro="""
    The OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model,
    without a language modeling head.
    """
)
class OmniASRModel(VibeVoiceModel):
    config: OmniASRConfig
    main_input_name = "input_ids"
    input_modalities = ("audio", "text")

    def __init__(self, config):
        super().__init__(config)
        self.multi_modal_projector = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size)
        del self.semantic_tokenizer_encoder
        del self.semantic_connector
        del self.diffusion_head
        del self.latent_scaling_factor
        del self.latent_bias_factor

    @can_return_tuple
    @auto_docstring(custom_intro="Encode audio into embeddings that can be used by the language model.")
    def get_audio_features(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            Float values of the raw audio waveform, as produced by [`OmniASRFeatureExtractor`].
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask to avoid running the speech encoder over padding samples of `input_values`. Values selected in
            `[0, 1]`: 1 for samples that are **not** padding, 0 for padding.
        """
        audio_output = self.audio_tower(input_values, padding_mask=padding_mask, **kwargs)
        audio_embeds = self.multi_modal_projector(audio_output.last_hidden_state)
        frames_mask = audio_output.attention_mask
        if frames_mask is None:
            audio_embeds = audio_embeds.flatten(0, 1)
        else:
            audio_embeds = audio_embeds[frames_mask.to(audio_embeds.device).bool()]
        audio_output.pooler_output = audio_embeds
        return audio_output


@auto_docstring(
    custom_intro="""
    OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model.
    """
)
class OmniASRForConditionalGeneration(VoxtralForConditionalGeneration):
    config: OmniASRConfig
    main_input_name = "input_ids"
    input_modalities = ("audio", "text")
    _keep_in_fp32_modules_strict = AttributeError()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Float values of the raw audio waveform, scattered over the audio placeholders of `input_ids`. See
            [`OmniASRModel.forward`].
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask to avoid running the speech encoder over padding samples of `input_values`, with 1 for samples that
            are **not** padding. Distinct from `attention_mask`, which covers the decoder sequence.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. They are shifted internally, so they must be
            aligned with the decoder sequence -- i.e. as long as `input_ids` / `inputs_embeds`, with `-100` on the
            positions that should not contribute to the loss (the audio context, and padding).

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-llm-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForConditionalGeneration.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor.apply_transcription_request(ds[0]["audio"]["array"], language="eng_Latn")
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> transcription = processor.decode(generated_ids, skip_special_tokens=True)
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            input_values=input_values,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
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
        )

    def prepare_inputs_for_generation(self, *args, is_first_iteration=False, **kwargs):
        input_values = kwargs.pop("input_values", None)
        padding_mask = kwargs.pop("padding_mask", None)

        model_inputs = super().prepare_inputs_for_generation(*args, is_first_iteration=is_first_iteration, **kwargs)

        if is_first_iteration or not kwargs.get("use_cache", True):
            if input_values is not None:
                model_inputs["input_values"] = input_values
            if padding_mask is not None:
                model_inputs["padding_mask"] = padding_mask

        return model_inputs


__all__ = [
    "OmniASRConfig",
    "OmniASRCTCConfig",
    "OmniASRAudioConfig",
    "OmniASRForCTC",
    "OmniASRForConditionalGeneration",
    "OmniASRModel",
    "OmniASRAudioModel",
    "OmniASRPreTrainedModel",
]
