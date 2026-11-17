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

from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_layers import GenericForTokenClassification
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import accepts_precomputed_kwargs, get_max_seqlen
from ...utils.output_capturing import capture_outputs
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Model
from ..auto import AutoConfig
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioPreTrainedModel
from ..qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeAudioEncoderLayer,
    SinusoidsPositionEmbedding,
    get_audio_cu_seqlens,
)
from ..voxtral.modeling_voxtral import VoxtralMultiModalProjector


@auto_docstring(checkpoint="Qwen/Qwen3-ASR-1.7B-hf")
@strict
class Qwen3ASREncoderConfig(Qwen3OmniMoeAudioEncoderConfig):
    r"""
    n_window (`int`, *optional*, defaults to 50):
        Half the number of mel frames in one encoder chunk. Each chunk processed by the conv stack has
        ``2 * n_window`` mel frames (1 second of audio at 16 kHz with a 10 ms hop).
    output_dim (`int`, *optional*, defaults to 3584):
        Dimensionality of the output.
    n_window_infer (`int`, *optional*, defaults to 800):
        Number of mel frames worth of audio over which each attention window spans. Must be a multiple
        of ``n_window * 2`` so attention windows align with encoder chunks.
    downsample_hidden_size (`int`, *optional*, defaults to 480):
        Hidden size of the convolutional downsampling stack.
    """

    model_type = "qwen3_asr_encoder"
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    max_position_embeddings: int = 13
    conv_chunksize = AttributeError()
    max_source_positions = AttributeError()


@auto_docstring(checkpoint="Qwen/Qwen3-ASR-1.7B-hf")
@strict
class Qwen3ASRConfig(PreTrainedConfig):
    r"""
    audio_token_id (`int`, *optional*, defaults to 151676):
        The audio token id to encode the audio prompt.
    timestamp_token_id (`int`, *optional*, defaults to 151705):
        Token ID of the ``<timestamp>`` marker in the tokenizer vocabulary. These markers
        delimit word boundaries in the forced-alignment input sequence.
    token_classification_bias (`bool`, *optional*, defaults to False):
        Whether the token classification head for forced alignment should have a bias term.

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
    sub_configs_defaults = {
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3_asr_encoder"),
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen3",
            init_kwargs={
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 65536,
                "tie_word_embeddings": True,
            },
        ),
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151676
    timestamp_token_id: int = 151705
    pad_token_id: int = 151645
    eos_token_id: list[int] | tuple[int, ...] | int = (151643, 151645)
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    token_classification_bias: bool = False


@auto_docstring
class Qwen3ASRPreTrainedModel(Qwen2AudioPreTrainedModel):
    _no_split_modules = AttributeError()
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, SinusoidsPositionEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            init.copy_(module.positional_embedding, position_embeddings)


class Qwen3ASRAudioEncoderLayer(Qwen3OmniMoeAudioEncoderLayer):
    def __init__(self, config: Qwen3ASREncoderConfig):
        super().__init__(config)


def get_audio_valid_indices(chunk_lengths: torch.Tensor, n_window: int, kwargs: dict | None = None) -> torch.Tensor:
    """
    The flat indices of the chunks' non-padding frames after the encoder's three stride-2 convolutions, or pop
    `"valid_indices"` from `kwargs` if precomputed.

    Args:
        chunk_lengths: `(num_chunks,)` pre-CNN chunk lengths.
        n_window: half the chunk size (in raw frames).
        kwargs: optional caller kwargs — if it contains `"valid_indices"` it is popped and returned.
    """
    if kwargs is not None and (valid_indices := kwargs.pop("valid_indices", None)) is not None:
        return valid_indices
    time_steps = Qwen3ASREncoder._post_cnn_length(torch.tensor(n_window * 2))
    valid_mask = (
        torch.arange(time_steps, device=chunk_lengths.device)
        < Qwen3ASREncoder._post_cnn_length(chunk_lengths)[:, None]
    )
    return valid_mask.flatten().nonzero().squeeze(-1)


@auto_docstring(
    custom_intro="""
    The audio model for Qwen3 ASR without any head or projection on top.
    """
)
class Qwen3ASREncoder(Qwen3OmniMoeAudioEncoder):
    config: Qwen3ASREncoderConfig
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]

    def __init__(self, config: Qwen3ASREncoderConfig):
        super().__init__(config)
        embed_dim = config.d_model
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_position_embeddings, embed_dim)
        del self.conv_chunksize
        del self.proj1
        del self.act
        del self.proj2
        del self.max_source_positions
        del self.num_mel_bins

    @staticmethod
    def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
        """Length after three (k=3, s=2, p=1) convolutions; zero-length input stays zero."""
        for _ in range(3):
            lengths = torch.where(lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
        return lengths

    def get_data_dependent_inputs(self, input_features_mask: torch.Tensor, **kwargs) -> dict[str, Any]:
        r"""
        The inputs the forward computes from `input_features_mask` with data-dependent operations (host syncs, loops
        over the clips): the windowed attention's `cu_seqlens` and `max_seqlen`, and the non-padding frames'
        `valid_indices`. Passed back to the forward, they keep `torch.compile` and `torch.export` free of those
        operations.

        Args:
            input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
                1 for valid mel frames and 0 for padding.
        """
        chunk_lengths = input_features_mask.view(input_features_mask.shape[0], -1, self.n_window * 2).sum(-1)
        chunk_lengths = chunk_lengths.reshape(-1).to(torch.long)
        feature_lens = input_features_mask.sum(-1).to(torch.long)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.n_window_infer, self.n_window)
        return {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": get_max_seqlen(cu_seqlens, self.config),
            "valid_indices": get_audio_valid_indices(chunk_lengths, self.n_window),
        }

    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
            1 for valid mel frames and 0 for padding.
        """
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self.n_window * 2

        if padded_feature_length % chunk_len != 0:
            raise ValueError(
                f"Qwen3ASREncoder expects `padded_feature_length` to be a multiple of "
                f"`n_window * 2` ({chunk_len}), but got {padded_feature_length}."
            )

        num_chunks = padded_feature_length // chunk_len

        # Compute cu_seqlens for windowed attention
        feature_lens = input_features_mask.sum(-1).to(torch.long)
        chunk_lengths = (
            input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
        )
        cu_seqlens = get_audio_cu_seqlens(
            chunk_lengths, feature_lens, self.n_window_infer, self.n_window, kwargs=kwargs
        )
        max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs=kwargs)

        # Chunk and process through CNN
        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )

        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out += self.positional_embedding.positional_embedding[:time_steps].to(conv_out.dtype)

        # Select only valid (non-padding) post-CNN positions into a flat packed sequence
        valid_indices = get_audio_valid_indices(chunk_lengths, self.n_window, kwargs=kwargs)
        hidden_states = torch.index_select(conv_out.reshape(-1, conv_out.shape[-1]), 0, valid_indices)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, cu_seqlens, max_seqlen=max_seqlen, **kwargs)
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class Qwen3ASRMultiModalProjector(VoxtralMultiModalProjector):
    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.linear_1 = nn.Linear(config.audio_config.d_model, config.audio_config.d_model)
        self.act = ACT2FN[config.audio_config.activation_function]
        self.linear_2 = nn.Linear(config.audio_config.d_model, config.audio_config.output_dim)


class Qwen3ASRModel(AudioFlamingo3Model):
    def get_data_dependent_inputs(self, input_features_mask: torch.Tensor | None = None, **kwargs) -> dict[str, Any]:
        r"""
        The audio encoder's data-dependent inputs (see [`Qwen3ASREncoder.get_data_dependent_inputs`]), prefixed
        `audio_`, as `get_audio_features` takes them.

        Args:
            input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`, *optional*):
                1 for valid mel frames and 0 for padding.
        """
        if input_features_mask is None:
            return {}
        inputs = self.audio_tower.get_data_dependent_inputs(input_features_mask)
        return {f"audio_{name}": value for name, value in inputs.items()}

    @accepts_precomputed_kwargs(modality="audio")
    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram)."
    )
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
            1 for valid mel frames and 0 for padding.
        """
        audio_output = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        audio_output.pooler_output = self.multi_modal_projector(audio_output.last_hidden_state)
        return audio_output


@auto_docstring(
    custom_intro="""
    The Qwen3ASR model which consists of an audio encoder and a language model.
    """
)
class Qwen3ASRForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}

    def forward(self, **super_kwargs):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import Qwen3ASRForConditionalGeneration, AutoProcessor

        >>> model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        ```"""
        return super().forward(**super_kwargs)

    def get_data_dependent_inputs(self, *args, **kwargs) -> dict[str, Any]:
        r"""See [`Qwen3ASRModel.get_data_dependent_inputs`]."""
        return self.model.get_data_dependent_inputs(*args, **kwargs)


@auto_docstring(
    custom_intro="""
    The Qwen3 ASR model with a token classification head for timestamp prediction (forced alignment).
    """
)
class Qwen3ASRForTokenClassification(GenericForTokenClassification, Qwen3ASRPreTrainedModel):
    pass


__all__ = [
    "Qwen3ASREncoderConfig",
    "Qwen3ASRConfig",
    "Qwen3ASREncoder",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRModel",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ASRForTokenClassification",
]
