# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import (
    create_bidirectional_mask,
    find_packed_sequence_indices,
    packed_sequence_mask_function,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING
from ..blip_2.modeling_blip_2 import Blip2QFormerMultiHeadAttention
from ..glmasr.modeling_glmasr import GlmAsrMLP
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import GraniteAttention, GraniteDecoderLayer, GraniteModel, repeat_kv
from ..granite_speech.modeling_granite_speech import (
    GraniteSpeechConformerBlock,
    GraniteSpeechCTCEncoder,
    GraniteSpeechEncoderProjector,
    GraniteSpeechModel,
)
from ..granite_speech_plus.configuration_granite_speech_plus import GraniteSpeechPlusEncoderConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarEncoderConfig(GraniteSpeechPlusEncoderConfig):
    r"""
    feedforward_mult (`int`, *optional*, defaults to 4):
        Multiplier for the feedforward layers; intermediate dim = `hidden_dim * feedforward_mult`.
    output_dim (`int`, *optional*, defaults to 348):
        Output dimension of the mid-layer CTC prediction head.
    context_size (`int`, *optional*, defaults to 200):
        Context size for block-wise conformer attention.
    max_pos_emb (`int`, *optional*, defaults to 512):
        Maximum relative positional embedding index (Shaw's relative positional encoding).
    conv_expansion_factor (`int`, *optional*, defaults to 2):
        Expansion factor for the conformer convolution module.
    self_conditioning_layer (`int`, *optional*):
        Layer index at which self-conditioning (mid-layer CTC feedback) is applied.
        Defaults to `num_layers // 2`.
    bpe_output_dim (`int`, *optional*, defaults to 49153):
        Vocabulary size for the BPE CTC head.
    bpe_pooling_window (`int`, *optional*, defaults to 4):
        Window size for posterior-weighted pooling before the BPE CTC head.
    cat_hidden_layers (`list[int]`, *optional*, defaults to `[4, 8, 12]`):
        Indices of intermediate encoder layers whose outputs are concatenated (with the always-appended
        final layer) to form the projector input.

    Example:

    ```python
    >>> from transformers import GraniteSpeechNarEncoderConfig

    >>> configuration = GraniteSpeechNarEncoderConfig()
    >>> print(configuration.hidden_size)
    1024
    ```"""

    model_type = "granite_speech_nar_encoder"
    attribute_map = {}

    num_layers: int = 16
    output_dim: int = 348
    self_conditioning_layer: int | None = None
    bpe_output_dim: int = 49153
    bpe_pooling_window: int = 4
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.self_conditioning_layer is None:
            self.self_conditioning_layer = self.num_layers // 2
        if self.cat_hidden_layers is None:
            self.cat_hidden_layers = [4, 8, 12]


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarProjectorConfig(PreTrainedConfig):
    r"""
    Configuration for the windowed Q-Former audio projector in GraniteSpeechNar.

    encoder_dim (`int`, *optional*, defaults to 1024):
        Hidden dimension of each concatenated encoder layer fed to the projector.
    downsample_rate (`int`, *optional*, defaults to 5):
        Temporal downsampling rate within each window block.
    num_encoder_layers (`int`, *optional*, defaults to 4):
        Number of encoder layers concatenated as projector input.
    num_layers (`int`, *optional*, defaults to 2):
        Number of Q-Former layers.
    block_size (`int`, *optional*, defaults to 15):
        Window size for blocked cross-attention in the projector.
    layernorm_eps (`float`, *optional*, defaults to 1e-6):
        Epsilon for layer normalization.
    attn_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in the Q-Former output projection.
    encoder_hidden_size (`int`, *optional*):
        Hidden size of the cross-attention key/value inputs. Defaults to `hidden_size`.

    Example:

    ```python
    >>> from transformers import GraniteSpeechNarProjectorConfig

    >>> configuration = GraniteSpeechNarProjectorConfig()
    >>> print(configuration.hidden_size)
    2048
    ```"""

    model_type = "granite_speech_nar_projector"

    encoder_dim: int = 1024
    downsample_rate: int = 5
    num_encoder_layers: int = 4
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_layers: int = 2
    dropout_prob: float = 0.1
    block_size: int = 15
    layernorm_eps: float = 1e-6
    attn_bias: bool = True
    attention_probs_dropout_prob: float = 0.0
    encoder_hidden_size: int | None = None
    hidden_act: str = "silu"
    intermediate_size: int = 4096

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.encoder_hidden_size is None:
            self.encoder_hidden_size = self.hidden_size


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarTextConfig(GraniteConfig):
    r"""
    Configuration for the bidirectional Granite language-model backbone of GraniteSpeechNar.

    A copy of [`GraniteConfig`] with a dedicated `model_type`, so that `AutoModel.from_config`
    resolves it to the non-causal [`GraniteSpeechNarTextModel`] rather than the causal `GraniteModel`.
    """

    model_type = "granite_speech_nar_text"


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarConfig(PreTrainedConfig):
    r"""
    Configuration for the GraniteSpeechNar non-autoregressive ASR model.

    This model uses a conformer encoder with BPE CTC head, a windowed Q-Former projector,
    and a bidirectional Granite LLM backbone for single-pass speech recognition.

    encoder_config (`GraniteSpeechNarEncoderConfig` or `dict`, *optional*):
        Configuration for the conformer encoder.
    projector_config (`GraniteSpeechNarProjectorConfig` or `dict`, *optional*):
        Configuration for the windowed Q-Former audio projector.
    blank_token_id (`int`, *optional*, defaults to 100257):
        Token ID used as the CTC blank symbol (the checkpoint reuses the EOS token).
    ce_loss_lambda (`float`, *optional*, defaults to 0.0):
        Weight for the auxiliary cross-entropy loss on the LLM output.
    encoder_ctc_loss_lambda (`float`, *optional*, defaults to 0.0):
        Weight for the auxiliary encoder BPE CTC loss.
    downsample_rate (`int`, *optional*, defaults to 5):
        Temporal downsampling rate of the windowed projector: each `window_size`-frame window is
        reduced to `window_size // downsample_rate` query tokens.
    window_size (`int`, *optional*, defaults to 15):
        Number of encoder frames per projector window.

    Example:

    ```python
    >>> from transformers import GraniteSpeechNarConfig, GraniteSpeechNarForCTC

    >>> configuration = GraniteSpeechNarConfig()
    >>> model = GraniteSpeechNarForCTC(configuration)
    >>> print(configuration.model_type)
    granite_speech_nar
    ```"""

    model_type = "granite_speech_nar"
    sub_configs = {
        "encoder_config": GraniteSpeechNarEncoderConfig,
        "projector_config": GraniteSpeechNarProjectorConfig,
        "text_config": GraniteSpeechNarTextConfig,
    }

    encoder_config: dict | PreTrainedConfig | None = None
    projector_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    tie_word_embeddings: bool = True
    blank_token_id: int = 100257
    ce_loss_lambda: float = 0.0
    encoder_ctc_loss_lambda: float = 0.0
    downsample_rate: int = 5
    window_size: int = 15

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "granite_speech_nar_text")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["granite_speech_nar_text"]()

        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "granite_speech_nar_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["granite_speech_nar_encoder"]()

        if isinstance(self.projector_config, dict):
            self.projector_config["model_type"] = self.projector_config.get(
                "model_type", "granite_speech_nar_projector"
            )
            self.projector_config = CONFIG_MAPPING[self.projector_config["model_type"]](**self.projector_config)
        elif self.projector_config is None:
            self.projector_config = CONFIG_MAPPING["granite_speech_nar_projector"]()

        super().__post_init__(**kwargs)


# A single module-level `eager_attention_forward` is shared by both attention modules:
# GQA language-model attention and MHA Q-Former cross-attention.
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    num_key_value_groups = getattr(module, "num_key_value_groups", 1)
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@auto_docstring(
    custom_intro="""
    Output of [`GraniteSpeechNarCTCEncoder`].
    """
)
@dataclass
class GraniteSpeechNarEncoderOutput(BaseModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_selected_layers * hidden_dim)`):
        Concatenation of the encoder layers selected by `config.cat_hidden_layers` (plus the final layer),
        fed to the projector.
    pooled_hidden_states (`torch.FloatTensor`):
        Posterior-weighted pooled hidden states, one per BPE window.
    """

    pooled_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Output of [`GraniteSpeechNarModel`] (the backbone, without the LM head).
    """
)
@dataclass
class GraniteSpeechNarModelOutput(BaseModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(1, sum(seq_lengths), hidden_size)`):
        Flat (packed) language-model hidden states at the text positions, before `lm_head`.
    seq_lengths (`list[int]`, *optional*):
        Per-sample text sequence lengths, used to split `last_hidden_state` / logits.
    inserted_ctc_token_ids (`list[torch.Tensor]`, *optional*):
        Encoder CTC predictions with blank insertion slots, fed to the language model.
    bpe_logits (`torch.Tensor`, *optional*):
        BPE CTC logits from the encoder, returned when the encoder CTC loss is active.
    bpe_lengths (`torch.Tensor`, *optional*):
        Number of valid BPE (pooled) windows per sample, `ceil(audio_lengths / bpe_pooling_window)`.
    audio_embeds (`torch.FloatTensor`, *optional*):
        Projected, embedding-scaled audio features. Returned so iterative-editing `generate` can reuse
        them across steps without re-running the encoder and projector.
    """

    seq_lengths: list[int] | None = None
    inserted_ctc_token_ids: list[torch.Tensor] | None = None
    bpe_logits: torch.Tensor | None = None
    bpe_lengths: torch.Tensor | None = None
    audio_embeds: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Output of the GraniteSpeechNarForCTC model.
    """
)
@dataclass
class GraniteSpeechNarOutput(ModelOutput):
    r"""
    loss (`torch.Tensor`, *optional*, returned when `labels` is provided):
        The (combined CTC + auxiliary) training loss.
    logits (`torch.FloatTensor` of shape `(1, sum(seq_lengths), vocab_size)`):
        Flat (packed) prediction scores over the vocabulary (after `lm_head` and logits scaling).
    seq_lengths (`list[int]`, *optional*):
        Per-sample text sequence lengths, used to split `logits`.
    """

    loss: torch.Tensor | None = None
    logits: torch.FloatTensor | None = None
    seq_lengths: list[int] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class GraniteSpeechNarGenerateOutput(ModelOutput):
    """
    Output of [`GraniteSpeechNarForCTC.generate`].

    Args:
        sequences (`list[torch.LongTensor]`):
            The generated (CTC-collapsed) token-id sequences, one 1-D tensor per sample. Returned as a
            list rather than a padded tensor because CTC decoding yields variable lengths.
        logits (`torch.FloatTensor`, *optional*):
            Flat (packed) language-model logits of shape `(1, sum(seq_lengths), vocab_size)` from the
            single forward pass (before CTC collapse); split per-sample with `seq_lengths`.
        seq_lengths (`list[int]`, *optional*):
            Per-sample text lengths, used to split `logits`.
    """

    sequences: list[torch.LongTensor] | None = None
    logits: torch.FloatTensor | None = None
    seq_lengths: list[int] | None = None


class GraniteSpeechNarConformerBlock(GradientCheckpointingLayer, GraniteSpeechConformerBlock): ...


@auto_docstring
class GraniteSpeechNarPreTrainedModel(PreTrainedModel):
    config_class = GraniteSpeechNarConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["GraniteSpeechNarCTCEncoder", "GraniteSpeechNarProjector", "GraniteSpeechNarDecoderLayer"]
    input_modalities = ("audio",)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GraniteSpeechNarProjector):
            init.normal_(module.query, mean=0.0, std=module.hidden_size**-0.5)
        elif isinstance(module, GraniteSpeechNarQFormerModel):
            init.normal_(module.window_positions, mean=0.0, std=module.config.hidden_size**-0.5)
        elif isinstance(module, GraniteSpeechNarCTCEncoder):
            context_size = module.config.context_size
            seq = torch.arange(context_size)
            relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
            attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + module.config.max_pos_emb
            init.copy_(module.attention_dists, attention_dists)


class GraniteSpeechNarQFormerCrossAttention(Blip2QFormerMultiHeadAttention): ...


class GraniteSpeechNarQFormerMLP(GlmAsrMLP): ...


class GraniteSpeechNarQFormerLayer(nn.Module):
    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps, elementwise_affine=False)
        self.cross_attention = GraniteSpeechNarQFormerCrossAttention(config, is_cross_attention=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_bias)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps, elementwise_affine=False)
        self.mlp = GraniteSpeechNarQFormerMLP(config)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.cross_attention(
            self.attn_norm(hidden_states), encoder_hidden_states=encoder_hidden_states
        )
        hidden_states = hidden_states + self.o_proj(attn_output)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class GraniteSpeechNarQFormerModel(GraniteSpeechNarPreTrainedModel):
    config: GraniteSpeechNarProjectorConfig

    def __init__(self, config: GraniteSpeechNarProjectorConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.window_positions = nn.Parameter(torch.zeros(1, config.block_size, config.hidden_size))
        self.layers = nn.ModuleList([GraniteSpeechNarQFormerLayer(config) for _ in range(config.num_layers)])
        self.post_init()

    def forward(
        self, query_embeds: torch.Tensor, encoder_hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.Tensor:
        mean_pool = encoder_hidden_states.unflatten(1, (-1, self.config.downsample_rate)).mean(-2)
        hidden_states = self.dropout(query_embeds + mean_pool)
        encoder_hidden_states = self.dropout(encoder_hidden_states + self.window_positions)

        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class GraniteSpeechNarProjector(GraniteSpeechEncoderProjector):
    def __init__(self, config: GraniteSpeechNarConfig):
        super().__init__(config)
        self.qformer = GraniteSpeechNarQFormerModel(config.projector_config)
        self.layer_norm = nn.LayerNorm(
            config.projector_config.encoder_dim, eps=config.projector_config.layernorm_eps, elementwise_affine=False
        )
        self.layer_projector = nn.Linear(
            config.projector_config.encoder_dim * config.projector_config.num_encoder_layers,
            config.projector_config.hidden_size,
        )
        self.projector_act = nn.GELU()
        self.dropout = nn.Dropout(config.projector_config.dropout_prob)
        self.out_norm = nn.LayerNorm(
            config.projector_config.hidden_size, eps=config.projector_config.layernorm_eps, elementwise_affine=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Normalize each concatenated encoder layer independently
        hidden_states = self.layer_norm(hidden_states.unflatten(-1, (-1, self.layer_norm.normalized_shape[0])))
        hidden_states = self.projector_act(self.layer_projector(hidden_states.flatten(-2)))

        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        if pad > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
        hidden_states = hidden_states.view(batch_size * nblocks, self.window_size, self.hidden_size)

        hidden_states = self.qformer(query_embeds=self.query, encoder_hidden_states=hidden_states)

        hidden_states = hidden_states.view(batch_size, nblocks * self.query.shape[1], -1)
        hidden_states = self.dropout(self.out_norm(hidden_states))
        return self.linear(hidden_states)


class GraniteSpeechNarAttention(GraniteAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.is_causal = False


class GraniteSpeechNarDecoderLayer(GraniteDecoderLayer): ...


class GraniteSpeechNarTextModel(GraniteModel):
    config_class = GraniteSpeechNarTextConfig
    _can_record_outputs = {
        "hidden_states": GraniteSpeechNarDecoderLayer,
        "attentions": GraniteSpeechNarAttention,
    }

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        packed_seq_mask = find_packed_sequence_indices(position_ids)
        and_mask_fn = packed_sequence_mask_function(packed_seq_mask) if packed_seq_mask is not None else None
        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            and_mask_function=and_mask_fn,
            attention_mask=None,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        kwargs["use_cache"] = False
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=bidirectional_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class GraniteSpeechNarCTCEncoder(GraniteSpeechCTCEncoder):
    def _posterior_weighted_pool(self, hidden_states: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        """Pool every `bpe_pooling_window` frames, weighting each frame by its (non-blank) importance."""
        window_size = self.config.bpe_pooling_window
        seq_len = hidden_states.shape[1]
        pad_len = -seq_len % window_size
        if pad_len:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            importance = F.pad(importance, (0, pad_len))

        hidden_states = hidden_states.unflatten(1, (-1, window_size))
        importance = importance.unflatten(1, (-1, window_size))
        weights = importance / importance.sum(-1, keepdim=True).clamp_min(1e-8)

        return (weights.unsqueeze(-2) @ hidden_states).squeeze(-2)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        input_features: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> GraniteSpeechNarEncoderOutput:
        hidden_states = self.input_linear(input_features.to(self.input_linear.weight.dtype))

        cat_layers = set(self.config.cat_hidden_layers or [])
        exported_hidden_states = []
        if 0 in cat_layers:
            exported_hidden_states.append(hidden_states)

        blank_probs = None
        for layer_idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)

            if layer_idx in cat_layers:
                exported_hidden_states.append(hidden_states)

            if layer_idx == self.config.self_conditioning_layer:
                mid_logits = self.out(hidden_states)
                mid_probs = torch.softmax(mid_logits.float(), dim=-1)
                blank_probs = mid_probs[:, :, 0]
                hidden_states = hidden_states + self.out_mid(mid_probs.to(hidden_states.dtype))

        importance = 1.0 - blank_probs
        pooled = self._posterior_weighted_pool(hidden_states.float(), importance).to(hidden_states.dtype)
        exported_hidden_states.append(hidden_states)
        last_hidden_state = torch.cat(exported_hidden_states, dim=-1)

        return GraniteSpeechNarEncoderOutput(
            last_hidden_state=last_hidden_state,
            pooled_hidden_states=pooled,
        )


@auto_docstring(
    custom_intro="""
    The GraniteSpeechNar base model consisting of a conformer encoder, QFormer projector,
    and a bidirectional Granite language model backbone.
    """
)
class GraniteSpeechNarModel(GraniteSpeechModel):
    def __init__(self, config: GraniteSpeechNarConfig):
        super().__init__(config)
        self.projector = GraniteSpeechNarProjector(config)
        self.out_bpe = nn.Linear(config.encoder_config.hidden_dim, config.encoder_config.bpe_output_dim, bias=True)
        self.post_init()

    def get_placeholder_mask(
        self,
    ):
        raise NotImplementedError("get_placeholder_mask is not implemented for GraniteSpeechNarModel")

    def get_merged_audio_embeddings(
        self,
        inserted_ctc_token_ids: list[torch.Tensor],
        audio_embeds: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """"""
        audio_lengths = audio_lengths // self.projector.downsample_rate
        text_lengths = [toks.shape[0] for toks in inserted_ctc_token_ids]
        inputs_embeds = [self.language_model.embed_tokens(toks) for toks in inserted_ctc_token_ids]
        audio_embeds = [audio_embeds[i, :length].to(inputs_embeds[0].device) for i, length in enumerate(audio_lengths)]

        inputs_embeds = torch.cat([x for pair in zip(audio_embeds, inputs_embeds) for x in pair])
        lengths = torch.tensor([l for el in zip(audio_lengths, text_lengths) for l in el], device=inputs_embeds.device)
        text_mask = (
            torch.tensor([False, True], device=inputs_embeds.device)
            .repeat(len(audio_lengths))
            .repeat_interleave(lengths)
        )

        position_ids = torch.cat(
            [
                torch.arange(el1.shape[0] + el2.shape[0], device=inputs_embeds.device)
                for el1, el2 in zip(audio_embeds, inserted_ctc_token_ids)
            ]
        )

        return inputs_embeds[None, ...], position_ids[None, ...], text_mask, text_lengths

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        input_ids: list[torch.Tensor] | None = None,
        input_features_mask: torch.Tensor | None = None,
        output_bpe_logits: bool = False,
        audio_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> GraniteSpeechNarModelOutput:
        r"""
        input_ids (`list[torch.Tensor]`, *optional*):
            Pre-computed (CTC-collapsed) token ids fed as the text input. Provided together with
            `audio_embeds` for the cached iterative-editing path; must be omitted on the `input_features`
            path, where they are derived from the encoder's own CTC predictions.
        input_features_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            Mask over the encoder frames (`True` for valid frames, `False` for padding).
        output_bpe_logits (`bool`, *optional*, defaults to `False`):
            Whether to return the encoder BPE CTC logits in `bpe_logits` (needed for the auxiliary
            encoder CTC loss).
        audio_embeds (`torch.FloatTensor`, *optional*):
            Pre-computed projected audio features, passed instead of `input_features` to skip the encoder
            and projector and reuse them directly (used by iterative-editing `generate`). Mutually
            exclusive with `input_features`; when provided, `input_ids` must be supplied as the text input.
        """
        if (input_features is None) ^ (audio_embeds is not None):
            raise ValueError("You must specify exactly one of input_features or audio_embeds")
        if (audio_embeds is None) != (input_ids is None):
            raise ValueError("audio_embeds and input_ids must be provided together")

        if input_features_mask is None:
            if input_features is None:
                raise ValueError("`input_features_mask` must be provided together with `audio_embeds`.")
            input_features_mask = torch.ones(input_features.shape[:-1], dtype=torch.bool, device=input_features.device)

        audio_lengths = input_features_mask.sum(dim=1)
        bpe_logits, bpe_lengths = None, None
        if audio_embeds is None:
            audio_outputs = self.get_audio_features(input_features, input_features_mask=input_features_mask)
            audio_embeds = audio_outputs.pooler_output / self.config.text_config.embedding_multiplier

            bpe_logits = self.out_bpe(audio_outputs.pooled_hidden_states)
            bpe_lengths = -(
                audio_lengths // -self.config.encoder_config.bpe_pooling_window
            )  # ceil(audio_lengths / bpe_pooling_window)
            input_ids = [self._ctc_greedy_decode(logits[:length]) for logits, length in zip(bpe_logits, bpe_lengths)]

        inserted_ctc_token_ids = [self._add_insertion_slots(ids) for ids in input_ids]
        inputs_embeds, position_ids, text_mask, text_lengths = self.get_merged_audio_embeddings(
            inserted_ctc_token_ids=inserted_ctc_token_ids,
            audio_embeds=audio_embeds,
            audio_lengths=audio_lengths,
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs,
        )

        return GraniteSpeechNarModelOutput(
            last_hidden_state=outputs.last_hidden_state[:, text_mask.to(outputs.last_hidden_state.device)],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            seq_lengths=text_lengths,
            inserted_ctc_token_ids=inserted_ctc_token_ids,
            bpe_logits=bpe_logits if output_bpe_logits else None,
            bpe_lengths=bpe_lengths if output_bpe_logits else None,
            audio_embeds=audio_embeds,
        )

    def _add_insertion_slots(self, ctc_token_ids: torch.Tensor) -> torch.Tensor:
        """[a, b] -> [blank, a, blank, b, blank]"""
        seq_length = ctc_token_ids.shape[0]
        inserted_token_ids = ctc_token_ids.new_full((seq_length * 2 + 1,), self.config.blank_token_id)
        inserted_token_ids[1 : 2 * seq_length : 2] = ctc_token_ids.reshape(-1)
        return inserted_token_ids

    def _ctc_greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        pred = torch.unique_consecutive(logits.argmax(-1))
        return pred[pred != self.config.blank_token_id]


@auto_docstring(
    custom_intro="""
    The GraniteSpeechNar model for non-autoregressive CTC-based speech recognition.
    Consists of a conformer encoder with BPE CTC head, a QFormer-based projector,
    and a bidirectional Granite LLM backbone that refines CTC predictions in a single pass.
    """
)
class GraniteSpeechNarForCTC(GraniteSpeechNarPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: GraniteSpeechNarConfig):
        super().__init__(config)
        self.model = GraniteSpeechNarModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> GraniteSpeechNarOutput:
        r"""
        Args:
            input_features (`torch.Tensor` of shape `(batch_size, seq_len, input_dim)`):
                Mel spectrogram features.
            input_features_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask over the encoder frames (`True` for valid frames, `False` for padding).
            labels (`torch.Tensor` of shape `(batch_size, max_label_len)`, *optional*):
                Ground-truth (BPE) token IDs for CTC training.
            label_lengths (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Number of valid tokens per sample in `labels`.

        Returns:
            [`GraniteSpeechNarOutput`]

        Example:

        ```python
        >>> from transformers import AutoModel, AutoProcessor
        >>> from transformers.audio_utils import load_audio

        >>> model_id = "ibm-granite/granite-speech-4.1-2b-nar"
        >>> revision = "refs/pr/6"  # native-format weights; drop once merged to `main`
        >>> processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        >>> model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")

        >>> url = "https://huggingface.co/buckets/huggingface/audio-samples/resolve/mister-quilter.mp3"
        >>> audio = load_audio(url, sampling_rate=processor.feature_extractor.sampling_rate)

        >>> inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate)
        >>> inputs.to(model.device, dtype=model.dtype)
        >>> output = model.generate(**inputs, return_dict_in_generate=True)
        >>> processor.batch_decode(output.sequences, skip_special_tokens=True)
        ['mrister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
        ```"""
        # The encoder BPE logits are only needed for the auxiliary encoder CTC loss during training.
        output_bpe_logits = labels is not None and self.config.encoder_ctc_loss_lambda > 0.0
        model_out = self.model(
            input_features=input_features,
            input_features_mask=input_features_mask,
            output_bpe_logits=output_bpe_logits,
            **kwargs,
        )

        logits = self.lm_head(model_out.last_hidden_state)
        logits = logits / self.config.text_config.logits_scaling

        loss = None
        if labels is not None:
            seq_lengths = model_out.seq_lengths
            ctc_loss_kwargs = {
                "targets": labels,
                "target_lengths": label_lengths,
                "blank": self.config.blank_token_id,
                "reduction": "sum",
                "zero_infinity": True,
            }

            log_probs = torch.log_softmax(logits.squeeze(0).float(), dim=-1)
            log_probs_padded = nn.utils.rnn.pad_sequence(log_probs.split(seq_lengths))
            input_lengths = torch.tensor(seq_lengths, device=logits.device)

            loss = (
                nn.functional.ctc_loss(log_probs_padded, input_lengths=input_lengths, **ctc_loss_kwargs)
                / input_lengths.sum()
            )

            if self.config.ce_loss_lambda > 0.0:
                # Auxiliary cross-entropy against the model's own (insertion-slotted) CTC predictions.
                ce_targets = torch.cat(model_out.inserted_ctc_token_ids)
                ce_loss = nn.functional.cross_entropy(
                    logits.squeeze(0), ce_targets.long(), reduction="mean", ignore_index=-100
                )
                loss = loss + self.config.ce_loss_lambda * ce_loss

            if self.config.encoder_ctc_loss_lambda > 0.0:
                bpe_lengths = model_out.bpe_lengths
                bpe_log_probs = torch.log_softmax(model_out.bpe_logits.float(), dim=-1)
                encoder_ctc_loss = (
                    nn.functional.ctc_loss(bpe_log_probs.transpose(0, 1), input_lengths=bpe_lengths, **ctc_loss_kwargs)
                    / bpe_lengths.sum()
                )
                loss = loss + self.config.encoder_ctc_loss_lambda * encoder_ctc_loss

        return GraniteSpeechNarOutput(
            loss=loss,
            logits=logits,
            seq_lengths=model_out.seq_lengths,
            hidden_states=model_out.hidden_states,
            attentions=model_out.attentions,
        )

    @torch.no_grad()
    @auto_docstring
    def generate(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        num_editing_steps: int = 1,
        return_dict_in_generate: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.LongTensor] | GraniteSpeechNarGenerateOutput:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            Mask over the encoder frames (`True` for valid frames, `False` for padding).
        num_editing_steps (`int`, *optional*, defaults to 1):
            Number of non-autoregressive editing passes. The first pass decodes from the encoder's CTC
            predictions; each subsequent pass collapses the previous LLM output via CTC and feeds it back
            as the text input for refinement, reusing the cached audio embeddings (so the encoder and
            projector run only once). `1` reproduces the single-pass behavior.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`GraniteSpeechNarGenerateOutput`], as opposed to returning
            exclusively the generated sequences.
        """
        input_ids, audio_embeds = None, None
        for _ in range(num_editing_steps):
            model_out = self.model(
                input_features=input_features,
                input_features_mask=input_features_mask,
                audio_embeds=audio_embeds,
                input_ids=input_ids,
                **kwargs,
            )
            logits = self.lm_head(model_out.last_hidden_state) / self.config.text_config.logits_scaling
            logits_per_sample = logits.split(model_out.seq_lengths, dim=1)

            input_ids = [self.model._ctc_greedy_decode(sample_logits) for sample_logits in logits_per_sample]
            audio_embeds = model_out.audio_embeds
            input_features = None

        if not return_dict_in_generate:
            return input_ids
        return GraniteSpeechNarGenerateOutput(
            sequences=input_ids,
            logits=logits,
            seq_lengths=model_out.seq_lengths,
        )


__all__ = [
    "GraniteSpeechNarConfig",
    "GraniteSpeechNarEncoderConfig",
    "GraniteSpeechNarProjectorConfig",
    "GraniteSpeechNarTextConfig",
    "GraniteSpeechNarCTCEncoder",
    "GraniteSpeechNarForCTC",
    "GraniteSpeechNarTextModel",
    "GraniteSpeechNarModel",
    "GraniteSpeechNarPreTrainedModel",
]
