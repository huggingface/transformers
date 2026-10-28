# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import json
import os
from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import DynamicCache
from ...convert_slow_tokenizer import bytes_to_unicode
from ...integrations.mlstm import (
    mlstm_chunkwise_native_autograd,
    mlstm_recurrent_sequence_native,
    mlstm_recurrent_step_native,
    soft_cap,
    wrap_chunkwise_arbitrary_sequence_length,
)
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import Unpack
from ...tokenization_python import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..byt5.tokenization_byt5 import ByT5Tokenizer
from ..olmo3.configuration_olmo3 import Olmo3Config
from ..olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3PreTrainedModel,
    Olmo3RMSNorm,
    Olmo3RotaryEmbedding,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from .generation_bolmo import BolmoGenerationMixin


@auto_docstring(checkpoint="allenai/Bolmo-7B")
@strict
class BolmoConfig(Olmo3Config):
    r"""
    subword_vocab_size (`int`, *optional*, defaults to 100278):
        Size of the subword vocabulary whose embeddings are added to the byte embeddings.
    num_local_encoder_layers (`int`, *optional*, defaults to 1):
        Number of mLSTM layers in the byte-level encoder.
    num_local_decoder_layers (`int`, *optional*, defaults to 4):
        Number of mLSTM layers in the byte-level decoder.
    num_local_heads (`int`, *optional*, defaults to 16):
        Number of heads of the mLSTM layers.
    local_intermediate_size (`int`, *optional*, defaults to 5504):
        Dimension of the MLPs of the byte-level layers.
    local_rms_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the norms around the patch pooling.
    bpe_token_end_id (`int`, *optional*, defaults to 3):
        Id emitted by rows that wait for the rest of the batch to reach a patch boundary.

    Example:

    ```python
    >>> from transformers import BolmoModel, BolmoConfig

    >>> configuration = BolmoConfig()
    >>> model = BolmoModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "bolmo"
    base_model_pp_plan = AttributeError()

    vocab_size: int = 520
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 1
    subword_vocab_size: int = 100278
    num_local_encoder_layers: int = 1
    num_local_decoder_layers: int = 4
    num_local_heads: int = 16
    local_intermediate_size: int = 5504
    local_rms_norm_eps: float = 1e-5
    bpe_token_end_id: int = 3


class BolmoCache(DynamicCache):
    """`DynamicCache` of the global layers, plus the recurrent state of the byte-level encoder and decoder."""

    def __init__(self, config: BolmoConfig):
        super().__init__(config=config)
        self.encoder_states = None
        self.decoder_states = None
        self.last_byte_hidden = None
        self.last_patch = None
        self.patch_mask = None


class BolmoRMSNorm(Olmo3RMSNorm):
    pass


class BolmoAttention(Olmo3Attention):
    pass


class BolmoMLP(Qwen2MoeMLP):
    pass


class BolmoDecoderLayer(Olmo3DecoderLayer):
    pass


class BolmoRotaryEmbedding(Olmo3RotaryEmbedding):
    pass


class BolmoMultiHeadLayerNorm(nn.Module):
    """Per-head layer norm (no bias) of the mLSTM output."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = F.layer_norm(hidden_states.float(), hidden_states.shape[-1:], eps=self.eps)
        return self.weight * hidden_states.to(input_dtype).flatten(2)


class BolmoMLSTMLayer(nn.Module):
    """xLSTM-large mLSTM layer (`weight_mode="single"`) running the native PyTorch kernels."""

    # xLSTM-large defaults, shared by every Bolmo checkpoint
    gate_soft_cap = 15.0

    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.num_heads = config.num_local_heads
        hidden_size = config.hidden_size
        self.q = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ogate_preact = nn.Linear(hidden_size, hidden_size, bias=False)
        self.igate_preact = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.fgate_preact = nn.Linear(hidden_size, self.num_heads, bias=True)
        self.multihead_norm = BolmoMultiHeadLayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        sequence_start: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        head_shape = (batch_size, seq_len, self.num_heads, -1)
        query = self.q(hidden_states).view(head_shape).transpose(1, 2)
        key = self.k(hidden_states).view(head_shape).transpose(1, 2)
        value = self.v(hidden_states).view(head_shape).transpose(1, 2)
        igate = soft_cap(self.igate_preact(hidden_states), self.gate_soft_cap).transpose(1, 2)
        fgate = soft_cap(self.fgate_preact(hidden_states), self.gate_soft_cap)
        if sequence_start is not None:
            # closing the forget gate on the first real byte drops the left padding from the state
            is_start = torch.arange(seq_len, device=fgate.device) == sequence_start[:, None]
            fgate = fgate.masked_fill(is_start[..., None], -100_000)
        fgate = fgate.transpose(1, 2)

        c_initial, n_initial, m_initial = state if state is not None else (None, None, None)
        mlstm_output, state = wrap_chunkwise_arbitrary_sequence_length(
            mlstm_chunkwise_kernel=mlstm_chunkwise_native_autograd,
            mlstm_sequence_kernel=partial(mlstm_recurrent_sequence_native, dtype_state=torch.float32),
            mlstm_step_kernel=partial(mlstm_recurrent_step_native, dtype_state=torch.float32),
            query=query,
            key=key,
            value=value,
            fgate=fgate,
            igate=igate,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            autocast_kernel_dtype=torch.float32,
        )
        mlstm_output = self.multihead_norm(mlstm_output.transpose(1, 2))
        output = self.out_proj(torch.sigmoid(self.ogate_preact(hidden_states)) * mlstm_output)
        return output, state


class BolmoLocalLayer(GradientCheckpointingLayer):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.pre_xlstm_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.xlstm = BolmoMLSTMLayer(config)
        self.pre_feedforward_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BolmoMLP(config, intermediate_size=config.local_intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        sequence_start: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states, state = self.xlstm(self.pre_xlstm_layernorm(hidden_states), state, sequence_start)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return hidden_states, state


class BolmoBoundaryPredictor(nn.Module):
    """A byte closes a patch when its projection is anti-aligned with the next byte's (P(boundary) = (1 - cos) / 2 > 0.5)."""

    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.q_proj_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, sequence_start: torch.LongTensor | None = None) -> torch.BoolTensor:
        query = F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1)
        key = F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1)
        boundary_mask = (query * key).sum(-1).float() < 0
        # the last byte has no lookahead yet, so it never closes a patch
        boundary_mask = F.pad(boundary_mask, (0, 1), value=False)
        positions = torch.arange(boundary_mask.shape[1], device=boundary_mask.device)
        start = sequence_start[:, None] if sequence_start is not None else 0
        # the first real byte always opens a patch, padding never closes one
        return (boundary_mask & (positions > start)) | (positions == start)


class BolmoLocalEncoder(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.byte_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.subword_embedding = nn.Embedding(config.subword_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BolmoLocalLayer(config) for _ in range(config.num_local_encoder_layers)])
        self.post_last_block_norm = BolmoRMSNorm(config.hidden_size, eps=config.local_rms_norm_eps)
        self.out_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.boundary_predictor_module = BolmoBoundaryPredictor(config)

    def forward(self, input_ids, expanded_input_ids, states=None, sequence_start=None):
        hidden_states = self.byte_embedding(input_ids) + self.subword_embedding(expanded_input_ids)
        new_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, state = layer(hidden_states, states[i] if states else None, sequence_start)
            new_states.append(state)
        return self.post_last_block_norm(hidden_states), new_states


class BolmoLocalDecoder(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.initial_norm = BolmoRMSNorm(config.hidden_size, eps=config.local_rms_norm_eps)
        self.in_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.layers = nn.ModuleList([BolmoLocalLayer(config) for _ in range(config.num_local_decoder_layers)])

    def forward(self, byte_hidden_states, patch_hidden_states, states=None, sequence_start=None):
        hidden_states = self.in_projection(byte_hidden_states) + patch_hidden_states
        new_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, state = layer(hidden_states, states[i] if states else None, sequence_start)
            new_states.append(state)
        return hidden_states, new_states


def keep_where(keep: torch.BoolTensor, old_states: list, new_states: list) -> list:
    """Per batch row, keep the old recurrent state where `keep` is set."""
    return [
        tuple(torch.where(keep.view(-1, *[1] * (old.dim() - 1)), old, new) for old, new in zip(old_layer, new_layer))
        for old_layer, new_layer in zip(old_states, new_states)
    ]


class BolmoPreTrainedModel(Olmo3PreTrainedModel):
    _no_split_modules = ["BolmoDecoderLayer", "BolmoLocalLayer"]
    _can_compile_fullgraph = False  # patching is data-dependent
    _is_stateful = True


@auto_docstring
class BolmoModel(BolmoPreTrainedModel):
    def __init__(self, config: BolmoConfig):
        super().__init__(config)
        self.boundary_offset = config.vocab_size // 2
        self.local_encoder = BolmoLocalEncoder(config)
        self.local_decoder = BolmoLocalDecoder(config)
        self.layers = nn.ModuleList(
            [BolmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BolmoRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.local_encoder.byte_embedding

    def set_input_embeddings(self, value):
        self.local_encoder.byte_embedding = value

    def predict_boundaries(self, input_ids, expanded_input_ids, attention_mask=None) -> torch.BoolTensor:
        """Patch boundaries of a byte sequence, `True` on the last byte of every patch."""
        sequence_start = (attention_mask == 0).sum(-1) if attention_mask is not None else None
        byte_hidden_states, _ = self.local_encoder(
            self._unfuse(input_ids), expanded_input_ids, sequence_start=sequence_start
        )
        return self.local_encoder.boundary_predictor_module(byte_hidden_states, sequence_start)

    def _unfuse(self, input_ids):
        """Byte ids at or above `boundary_offset` also mark a boundary, the encoder only reads the byte."""
        return torch.where(input_ids >= self.boundary_offset, input_ids - self.boundary_offset, input_ids)

    def _global_forward(self, patch_embeds, patch_mask, position_ids, past_key_values, **kwargs):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": patch_embeds,
            "attention_mask": patch_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        position_embeddings = {
            layer_type: self.rotary_emb(patch_embeds, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        hidden_states = patch_embeds
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                **kwargs,
            )
        return self.local_decoder.initial_norm(hidden_states)

    def _prefill(self, input_ids, expanded_input_ids, attention_mask, boundary_mask, past_key_values, **kwargs):
        sequence_start = (attention_mask == 0).sum(-1) if attention_mask is not None else None
        byte_hidden_states, encoder_states = self.local_encoder(
            self._unfuse(input_ids), expanded_input_ids, sequence_start=sequence_start
        )
        if boundary_mask is None:
            boundary_mask = self.local_encoder.boundary_predictor_module(byte_hidden_states, sequence_start)

        # pool: one patch per boundary byte, left-padded so the global model sees a regular padded batch
        num_patches = boundary_mask.sum(-1)
        max_patches = int(num_patches.max())
        patch_padding = max_patches - num_patches
        boundary_positions = torch.sort((~boundary_mask).int(), dim=1, stable=True).indices[:, :max_patches]
        slot = torch.arange(max_patches, device=input_ids.device) - patch_padding[:, None]
        patch_mask = slot >= 0
        boundary_positions = boundary_positions.gather(1, slot.clamp(min=0))
        patch_embeds = byte_hidden_states.gather(
            1, boundary_positions[..., None].expand_as(byte_hidden_states[:, :max_patches])
        )
        patch_embeds = self.local_encoder.out_projection(patch_embeds)
        position_ids = (patch_mask.cumsum(-1) - 1).clamp(min=0)
        patch_hidden_states = self._global_forward(
            patch_embeds, patch_mask.long(), position_ids, past_key_values, **kwargs
        )

        # depool: every byte reads the patch it closes, or the last one closed before it
        patch_index = (boundary_mask.cumsum(-1) - 1).clamp(min=0) + patch_padding[:, None]
        patch_index = patch_index.clamp(max=max_patches - 1)
        depooled = patch_hidden_states.gather(1, patch_index[..., None].expand_as(byte_hidden_states))
        hidden_states, decoder_states = self.local_decoder(byte_hidden_states, depooled, sequence_start=sequence_start)

        if past_key_values is not None:
            past_key_values.encoder_states = encoder_states
            past_key_values.decoder_states = decoder_states
            past_key_values.last_byte_hidden = byte_hidden_states[:, -1]
            past_key_values.last_patch = patch_hidden_states[:, -1]
            past_key_values.patch_mask = patch_mask.long()
        return hidden_states

    def _decode_step(self, input_ids, expanded_input_ids, past_key_values, **kwargs):
        # rows that emitted `bpe_token_end` (or are finished) wait, rows on a boundary need a new patch
        last_token = input_ids[:, -1]
        waiting = (last_token == self.config.bpe_token_end_id) | (last_token == self.config.pad_token_id)
        at_boundary = waiting | (last_token >= self.boundary_offset)

        byte_hidden_states, encoder_states = self.local_encoder(
            self._unfuse(input_ids), expanded_input_ids, states=past_key_values.encoder_states
        )
        past_key_values.encoder_states = keep_where(waiting, past_key_values.encoder_states, encoder_states)
        past_key_values.last_byte_hidden = torch.where(
            waiting[:, None], past_key_values.last_byte_hidden, byte_hidden_states[:, -1]
        )
        byte_hidden_states = past_key_values.last_byte_hidden[:, None]

        # the global model only steps once every row closed its patch
        decoder_waits = at_boundary
        if at_boundary.all():
            patch_mask = F.pad(past_key_values.patch_mask, (0, 1), value=1)
            past_key_values.patch_mask = patch_mask
            position_ids = patch_mask.sum(-1, keepdim=True) - 1
            patch_embeds = self.local_encoder.out_projection(byte_hidden_states)
            patch_hidden_states = self._global_forward(
                patch_embeds, patch_mask, position_ids, past_key_values, **kwargs
            )
            past_key_values.last_patch = patch_hidden_states[:, -1]
            decoder_waits = torch.zeros_like(at_boundary)

        hidden_states, decoder_states = self.local_decoder(
            byte_hidden_states, past_key_values.last_patch[:, None], states=past_key_values.decoder_states
        )
        past_key_values.decoder_states = keep_where(decoder_waits, past_key_values.decoder_states, decoder_states)
        return hidden_states

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        expanded_input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        boundary_mask: torch.BoolTensor | None = None,
        past_key_values: BolmoCache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        expanded_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Id of the longest subword ending at each byte, as returned by [`BolmoTokenizer`].
        boundary_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            `True` on the last byte of each patch. Predicted by the model when not provided.
        """
        if use_cache and not isinstance(past_key_values, BolmoCache):
            past_key_values = BolmoCache(config=self.config)

        if past_key_values is not None and past_key_values.last_patch is not None:
            hidden_states = self._decode_step(input_ids, expanded_input_ids, past_key_values, **kwargs)
        else:
            hidden_states = self._prefill(
                input_ids, expanded_input_ids, attention_mask, boundary_mask, past_key_values, **kwargs
            )

        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values)


class BolmoForCausalLM(BolmoPreTrainedModel, BolmoGenerationMixin):
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: BolmoConfig):
        super().__init__(config)
        self.model = BolmoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        expanded_input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        boundary_mask: torch.BoolTensor | None = None,
        past_key_values: BolmoCache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        expanded_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Id of the longest subword ending at each byte, as returned by [`BolmoTokenizer`].
        boundary_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            `True` on the last byte of each patch. Predicted by the model when not provided.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BolmoForCausalLM

        >>> model = BolmoForCausalLM.from_pretrained("allenai/Bolmo-7B")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-7B")

        >>> inputs = tokenizer("Language modeling is ", return_tensors="pt")
        >>> output = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=32)
        >>> tokenizer.decode(output[0], skip_special_tokens=True)
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            expanded_input_ids=expanded_input_ids,
            attention_mask=attention_mask,
            boundary_mask=boundary_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BolmoTokenizer(ByT5Tokenizer):
    r"""
    Byte tokenizer of Bolmo: UTF-8 bytes after 4 special tokens. Also returns `expanded_input_ids`, the id of the longest
    subword of `vocab_file` ending at each byte.

    Args:
        vocab_file (`str`):
            Subword vocabulary (`{token: id}`, GPT-2 byte-level strings) used for `expanded_input_ids`.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "expanded_input_ids", "attention_mask"]
    padding_side = "left"

    def __init__(
        self,
        vocab_file: str,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<bos>",
        bpe_token_end_token="<bpe_token_end>",
        **kwargs,
    ):
        special_tokens = [str(pad_token), str(bos_token), "<eos>", str(bpe_token_end_token)]
        self._added_tokens_decoder = {i: AddedToken(token, special=True) for i, token in enumerate(special_tokens)}
        self.offset = len(special_tokens)
        self._utf_vocab_size = 2**8
        self.boundary_offset = self.offset + self._utf_vocab_size
        self.bpe_token_end_id = special_tokens.index(str(bpe_token_end_token))
        with open(vocab_file, encoding="utf-8") as f:
            self.subword_vocab = json.load(f)
        self._subword_trie = None
        kwargs.setdefault("extra_special_tokens", [str(bpe_token_end_token)])
        PreTrainedTokenizer.__init__(self, pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, **kwargs)

    @property
    def subword_trie(self):
        """Subword vocab as a trie over reversed byte ids, to find the longest subword ending at a byte."""
        if self._subword_trie is None:
            byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
            special_ids = {str(token): i for i, token in self._added_tokens_decoder.items()}
            self._subword_trie = {}
            for token, subword_id in self.subword_vocab.items():
                if token in special_ids:
                    byte_ids = [special_ids[token]]
                else:
                    byte_ids = [self.offset + byte_decoder[char] for char in token]
                node = self._subword_trie
                for byte_id in reversed(byte_ids):
                    node = node.setdefault(byte_id, {})
                node[None] = subword_id
        return self._subword_trie

    def expand_byte_ids(self, byte_ids: list[int], n_last: int | None = None) -> list[int]:
        """Id of the longest subword ending at each byte (of the `n_last` bytes only, if given)."""
        expanded_ids = []
        for end in range(len(byte_ids) - (n_last or len(byte_ids)), len(byte_ids)):
            node, subword_id = self.subword_trie, None
            for byte_id in reversed(byte_ids[: end + 1]):
                if byte_id == self.bpe_token_end_id:
                    continue
                if byte_id >= self.boundary_offset:
                    byte_id -= self.boundary_offset
                if byte_id not in node:
                    break
                node = node[byte_id]
                subword_id = node.get(None, subword_id)
            expanded_ids.append(subword_id)
        return expanded_ids

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        encoding = PreTrainedTokenizer.__call__(self, *args, **kwargs)
        input_ids = encoding["input_ids"]
        rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        if rows and isinstance(rows[0], list):
            encoding["expanded_input_ids"] = [self.expand_byte_ids(row) for row in rows]
        else:
            encoding["expanded_input_ids"] = self.expand_byte_ids(rows)
        return BatchEncoding(encoding.data, tensor_type=kwargs.get("return_tensors"))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0
        return [self.bos_token_id] + token_ids_0 + [self.bos_token_id] + token_ids_1

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return PreTrainedTokenizer.get_special_tokens_mask(self, token_ids_0, token_ids_1, True)
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0)
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def _decode(self, token_ids, **kwargs) -> str:
        token_ids = [token_ids] if isinstance(token_ids, int) else token_ids
        token_ids = [i - self.boundary_offset if i >= self.boundary_offset else i for i in token_ids]
        return PreTrainedTokenizer._decode(self, token_ids, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.subword_vocab, f, ensure_ascii=False)
        return (vocab_file,)


__all__ = ["BolmoConfig", "BolmoForCausalLM", "BolmoModel", "BolmoPreTrainedModel", "BolmoTokenizer"]
