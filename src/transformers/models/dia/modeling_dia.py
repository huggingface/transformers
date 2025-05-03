# coding=utf-8
# Copyright 2025 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dia model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from .configuration_dia import DiaConfig
from .generation_dia import DiaGenerationMixin


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask

if is_flash_attn_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 1

_CONFIG_FOR_DOC = "DiaConfig"
_CHECKPOINT_FOR_DOC = "nari-labs/Dia-1.6B"


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class DiaPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]
        else:
            return self.weight[position_ids]



# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Dia, MBART->DIA
class DiaEncoderLayer(nn.Module):
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd
        self.compute_dtype = compute_dtype

        self.pre_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.self_attention = Attention(
            config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head,
            head_dim=enc_config.head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )
        self.post_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.mlp = MlpBlock(embed_dim=embed_dim, intermediate_dim=enc_config.n_hidden, compute_dtype=compute_dtype)


    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: EncoderInferenceState,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x).to(self.compute_dtype)

        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=state.positions,
            kv_positions=state.positions,
            attn_mask=state.attn_mask,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.post_sa_norm(x).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class DiaPreTrainedModel(PreTrainedModel):
    config_class = DiaConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DiaEncoderLayer", "DiaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, DiaEncoder):
            module.embed_positions.weight.copy_(sinusoids(*module.embed_positions.weight.shape))
        elif isinstance(module, DiaForAudioClassification):
            if self.config.use_weighted_layer_sum:
                module.layer_weights.data.fill_(1.0 / (self.config.num_hidden_layers + 1))



class DiaEncoder(DiaPreTrainedModel):
    """Transformer Encoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        self.compute_dtype = compute_dtype

        self.embedding = nn.Embedding(
            model_config.src_vocab_size,
            enc_config.n_embd,
            dtype=compute_dtype,
        )
        self.layers = nn.ModuleList([EncoderLayer(config, compute_dtype) for _ in range(enc_config.n_layer)])
        self.norm = RMSNorm(
            enc_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

    def forward(
        self,
        x_ids: torch.Tensor,
        state: EncoderInferenceState,
    ) -> torch.Tensor:
        x = self.embedding(x_ids)

        for layer in self.layers:
            x = layer(x, state)

        x = self.norm(x).to(self.compute_dtype)
        return x


class DiaDecoderLayer(nn.Module):
    def __init__(self, config: DiaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attention = DIA_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.pre_sa_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.pre_ca_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.pre_mlp_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.cross_attention = DIA_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.mlp = MlpBlock(
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: DecoderInferenceState,
        self_attn_cache: KVCache | None = None,
        cross_attn_cache: KVCache | None = None,
        prefill: bool = False,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x).to(self.compute_dtype)

        sa_out = self.self_attention(
            Xq=x_norm,  # (2, 1, D)
            Xkv=x_norm,  # (2, 1, D)
            q_positions=state.dec_positions,  # (2, 1)
            kv_positions=state.dec_positions,  # (2, 1)
            attn_mask=None,
            cache=self_attn_cache,
            prefill=prefill,
            is_causal=prefill,
        )

        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x).to(self.compute_dtype)
        ca_out = self.cross_attention(
            Xq=x_norm,
            Xkv=state.enc_out,
            q_positions=state.dec_positions,
            kv_positions=state.enc_positions,
            attn_mask=state.dec_cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x



class Decoder(DiaPreTrainedModel):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        data_config = config.data
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(model_config.tgt_vocab_size, dec_config.n_embd, dtype=compute_dtype)
                for _ in range(self.num_channels)
            ]
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config=config, compute_dtype=compute_dtype) for _ in range(self.num_layers)]
        )

        self.norm = RMSNorm(
            dec_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

    def precompute_cross_attn_cache(
        self,
        enc_out: torch.Tensor,  # (B, S, E)
        enc_positions: torch.Tensor,  # (B, S)
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: list[KVCache] = []

        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k_proj = cross_attn_module.k_proj(enc_out)
            v_proj = cross_attn_module.v_proj(enc_out)

            k_proj = cross_attn_module.rotary_emb(k_proj, position=enc_positions)
            k = k_proj.transpose(1, 2)
            v = v_proj.transpose(1, 2)

            per_layer_kv_cache.append(KVCache.from_kv(k, v))

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: torch.Tensor,  # [B, 1, C]
        state: DecoderInferenceState,
    ) -> torch.Tensor:
        """
        Performs a single decoding step, managing KV caches layer by layer.

        Returns:
            A tuple containing:
            - logits_Bx1xCV: The final output logits for the current step (B, 1, C*V), cast to float32.
        """

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(
                x,  # (2, 1, D)
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )

        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)

        return logits_Bx1xCxV.to(torch.float32)

    def forward(self, tgt_ids_BxTxC: torch.Tensor, state: DecoderInferenceState) -> torch.Tensor:
        """
        Forward pass for the Decoder stack, managing KV caches.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            encoder_out: Output from the encoder (B, S, E).
            tgt_positions: Positions for target sequence (B, T).
            src_positions: Positions for source sequence (B, S).
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.
            past_key_values: List containing the self-attention KV cache for each layer
                             from the previous decoding step. `len(past_key_values)` should
                             equal `num_layers`.
            precomputed_cross_attn_kv: A single tuple containing the pre-computed K/V cache
                                      derived from `encoder_out`. This is passed identically
                                      to all layers.

        Returns:
            A tuple containing:
            - logits: The final output logits (B, T, C * V), cast to float32.
            - present_key_values: A list containing the updated self-attention KV cache
                                 for each layer for the *current* decoding step.
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(x, state, self_attn_cache=self_cache, cross_attn_cache=cross_cache, prefill=True)

        # Final Norm
        x = self.norm(x)
        logits_BxTxCxV = self.logits_dense(x)

        return logits_BxTxCxV.to(torch.float32)



class DiaModel(DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.encoder = DiaEncoder(config)
        self.decoder = DiaDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Dia encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    @add_start_docstrings_to_model_forward(DIA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, DiaModel
         >>> from datasets import load_dataset

         >>> model = DiaModel.from_pretrained("openai/dia-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/dia-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



__all__ = [
    "DiaModel",
    "DiaPreTrainedModel",
]
