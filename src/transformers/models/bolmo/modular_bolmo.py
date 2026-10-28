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

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from torch import nn

from ...cache_utils import DynamicCache, LinearAttentionLayer
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...convert_slow_tokenizer import bytes_to_unicode
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..olmo3.modeling_olmo3 import Olmo3RMSNorm
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from ..xlstm.modeling_xlstm import eager_mlstm_forward, soft_cap
from .generation_bolmo import BolmoGenerationMixin


@auto_docstring(checkpoint="allenai/Bolmo-7B")
@strict
class BolmoConfig(PreTrainedConfig):
    r"""
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Config of the global model running on patches, an [`Olmo3Config`] by default. Its `vocab_size` is the subword
        vocabulary whose embeddings are added to the byte embeddings.
    num_local_encoder_layers (`int`, *optional*, defaults to 1):
        Number of mLSTM layers in the byte-level encoder.
    num_local_decoder_layers (`int`, *optional*, defaults to 4):
        Number of mLSTM layers in the byte-level decoder.
    num_local_heads (`int`, *optional*, defaults to 16):
        Number of heads of the mLSTM layers.
    local_intermediate_size (`int`, *optional*, defaults to 5504):
        Dimension of the MLPs of the byte-level layers.
    local_rms_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the norm closing the byte-level encoder.
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
    sub_configs_defaults = {"text_config": SubConfigSpec(config_class=AutoConfig, model_type="olmo3")}

    text_config: dict | PreTrainedConfig | None = None
    vocab_size: int = 520
    hidden_size: int = 4096
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    num_local_encoder_layers: int = 1
    num_local_decoder_layers: int = 4
    num_local_heads: int = 16
    local_intermediate_size: int = 5504
    local_rms_norm_eps: float = 1e-5
    bpe_token_end_id: int = 3
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        # TODO: remove once the Hub configs nest the global model under `text_config`
        if self.text_config is None:
            olmo3_keys = set(CONFIG_MAPPING["olmo3"].__dataclass_fields__) | {"rope_scaling", "rope_theta"}
            self.text_config = {
                "model_type": "olmo3",
                "vocab_size": kwargs.pop("subword_vocab_size", 100278),
                "max_position_embeddings": 65536,
                **{key: kwargs.pop(key) for key in list(kwargs) if key in olmo3_keys},
                "hidden_size": self.hidden_size,
                "hidden_act": self.hidden_act,
                "rms_norm_eps": self.rms_norm_eps,
            }
        super().__post_init__(**kwargs)


class BolmoCache(DynamicCache):
    """
    `DynamicCache` of the global layers, followed by one `LinearAttentionLayer` per mLSTM layer (its `(c, n, m)`
    state) and one holding the last encoded byte and the last patch:

        layers: [global 0 .. N-1][encoder mLSTM][decoder mLSTM][last byte hidden state, last patch]

    `patch_mask` is the attention mask of the (left-padded) patches seen by the global layers.
    """

    def __init__(self, config: BolmoConfig):
        super().__init__(config=config)
        self.encoder_start = len(self.layers)
        self.decoder_start = self.encoder_start + config.num_local_encoder_layers
        self.byte_state_idx = self.decoder_start + config.num_local_decoder_layers
        num_mlstm_layers = config.num_local_encoder_layers + config.num_local_decoder_layers
        self.layers += [LinearAttentionLayer(number_of_states=3) for _ in range(num_mlstm_layers)]
        self.layers.append(LinearAttentionLayer(number_of_states=2))
        self.patch_mask = None

    def get_states(self, start: int, num_layers: int = 1) -> list[tuple[torch.Tensor, ...]]:
        return [tuple(layer.recurrent_states.values()) for layer in self.layers[start : start + num_layers]]

    def update_states(self, start: int, states: list, keep: torch.BoolTensor | None = None) -> None:
        """Store `states` from layer `start` on, keeping the old state of the batch rows where `keep` is set."""
        for layer_idx, layer_states in enumerate(states, start=start):
            for state_idx, state in enumerate(layer_states):
                if keep is not None:
                    old_state = self.layers[layer_idx].recurrent_states[state_idx]
                    state = torch.where(keep.view(-1, *[1] * (state.dim() - 1)), old_state, state)
                self.update_recurrent_state(state, layer_idx, state_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        self.patch_mask = self.patch_mask.index_select(0, beam_idx.to(self.patch_mask.device))


class BolmoRMSNorm(Olmo3RMSNorm):
    pass


class BolmoMLP(Qwen2MoeMLP):
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
    """
    xLSTM-large mLSTM layer (`weight_mode="single"`). Every released checkpoint
    (Bolmo-1B, Bolmo-7B and their Stage1) uses the xLSTM-large defaults, hence the constants: gate soft cap 15,
    `qk_dim = hidden_size // 2`, `v_dim = hidden_size`.
    """

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
            fgate = fgate.masked_fill(is_start[..., None], torch.finfo(fgate.dtype).min)
        fgate = fgate.transpose(1, 2)

        mlstm_output, state = eager_mlstm_forward(query, key, value, input_gate=igate, forget_gate=fgate, state=state)
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
        self.layers = nn.ModuleList([BolmoLocalLayer(config) for _ in range(config.num_local_encoder_layers)])
        self.post_last_block_norm = BolmoRMSNorm(config.hidden_size, eps=config.local_rms_norm_eps)
        self.out_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.boundary_predictor_module = BolmoBoundaryPredictor(config)

    def forward(self, hidden_states, states=None, sequence_start=None):
        new_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, state = layer(hidden_states, states[i] if states else None, sequence_start)
            new_states.append(state)
        return self.post_last_block_norm(hidden_states), new_states


class BolmoLocalDecoder(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.in_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.layers = nn.ModuleList([BolmoLocalLayer(config) for _ in range(config.num_local_decoder_layers)])

    def forward(self, byte_hidden_states, patch_hidden_states, states=None, sequence_start=None):
        hidden_states = self.in_projection(byte_hidden_states) + patch_hidden_states
        new_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, state = layer(hidden_states, states[i] if states else None, sequence_start)
            new_states.append(state)
        return hidden_states, new_states


class BolmoPreTrainedModel(PreTrainedModel):
    config: BolmoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BolmoLocalLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False  # patching is data-dependent
    _is_stateful = True


@auto_docstring
class BolmoModel(BolmoPreTrainedModel):
    def __init__(self, config: BolmoConfig):
        super().__init__(config)
        self.boundary_offset = config.vocab_size // 2
        self.local_encoder = BolmoLocalEncoder(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.local_decoder = BolmoLocalDecoder(config)
        self.norm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.local_encoder.byte_embedding

    def set_input_embeddings(self, value):
        self.local_encoder.byte_embedding = value

    def _embed(self, input_ids, expanded_input_ids):
        """Bytes (ids at or above `boundary_offset` also close a patch) plus the subword ending at each byte."""
        byte_ids = torch.where(input_ids >= self.boundary_offset, input_ids - self.boundary_offset, input_ids)
        subword_embeds = self.language_model.get_input_embeddings()(expanded_input_ids)
        return self.local_encoder.byte_embedding(byte_ids) + subword_embeds

    def predict_boundaries(self, input_ids, expanded_input_ids, attention_mask=None) -> torch.BoolTensor:
        """Patch boundaries of a byte sequence, `True` on the last byte of every patch."""
        sequence_start = (attention_mask == 0).sum(-1) if attention_mask is not None else None
        byte_hidden_states, _ = self.local_encoder(
            self._embed(input_ids, expanded_input_ids), sequence_start=sequence_start
        )
        return self.local_encoder.boundary_predictor_module(byte_hidden_states, sequence_start)

    def _prefill(self, input_ids, expanded_input_ids, attention_mask, boundary_mask, past_key_values, **kwargs):
        sequence_start = (attention_mask == 0).sum(-1) if attention_mask is not None else None
        byte_hidden_states, encoder_states = self.local_encoder(
            self._embed(input_ids, expanded_input_ids), sequence_start=sequence_start
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
        global_outputs = self.language_model(
            inputs_embeds=patch_embeds,
            attention_mask=patch_mask.long(),
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            **kwargs,
        )
        patch_hidden_states = global_outputs.last_hidden_state

        # depool: every byte reads the patch it closes, or the last one closed before it
        patch_index = (boundary_mask.cumsum(-1) - 1).clamp(min=0) + patch_padding[:, None]
        patch_index = patch_index.clamp(max=max_patches - 1)
        depooled = patch_hidden_states.gather(1, patch_index[..., None].expand_as(byte_hidden_states))
        hidden_states, decoder_states = self.local_decoder(byte_hidden_states, depooled, sequence_start=sequence_start)

        if past_key_values is not None:
            past_key_values.update_states(past_key_values.encoder_start, encoder_states)
            past_key_values.update_states(past_key_values.decoder_start, decoder_states)
            past_key_values.update_states(
                past_key_values.byte_state_idx, [(byte_hidden_states[:, -1], patch_hidden_states[:, -1])]
            )
            past_key_values.patch_mask = patch_mask.long()
        return hidden_states, global_outputs

    def _decode_step(self, input_ids, expanded_input_ids, cache, **kwargs):
        # rows that emitted `bpe_token_end` (or are finished) wait, rows on a boundary need a new patch
        last_token = input_ids[:, -1]
        waiting = (last_token == self.config.bpe_token_end_id) | (last_token == self.config.pad_token_id)
        at_boundary = waiting | (last_token >= self.boundary_offset)

        byte_hidden_states, encoder_states = self.local_encoder(
            self._embed(input_ids, expanded_input_ids),
            states=cache.get_states(cache.encoder_start, len(self.local_encoder.layers)),
        )
        cache.update_states(cache.encoder_start, encoder_states, keep=waiting)
        ((last_byte_hidden, last_patch),) = cache.get_states(cache.byte_state_idx)
        last_byte_hidden = torch.where(waiting[:, None], last_byte_hidden, byte_hidden_states[:, -1])
        byte_hidden_states = last_byte_hidden[:, None]

        # the global model only steps once every row closed its patch
        decoder_waits = at_boundary
        global_outputs = None
        if at_boundary.all():
            cache.patch_mask = F.pad(cache.patch_mask, (0, 1), value=1)
            position_ids = cache.patch_mask.sum(-1, keepdim=True) - 1
            patch_embeds = self.local_encoder.out_projection(byte_hidden_states)
            global_outputs = self.language_model(
                inputs_embeds=patch_embeds,
                attention_mask=cache.patch_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                **kwargs,
            )
            last_patch = global_outputs.last_hidden_state[:, -1]
            decoder_waits = torch.zeros_like(at_boundary)
        cache.update_states(cache.byte_state_idx, [(last_byte_hidden, last_patch)])

        hidden_states, decoder_states = self.local_decoder(
            byte_hidden_states,
            last_patch[:, None],
            states=cache.get_states(cache.decoder_start, len(self.local_decoder.layers)),
        )
        cache.update_states(cache.decoder_start, decoder_states, keep=decoder_waits)
        return hidden_states, global_outputs

    @merge_with_config_defaults
    @can_return_tuple
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

        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            hidden_states, global_outputs = self._decode_step(input_ids, expanded_input_ids, past_key_values, **kwargs)
        else:
            hidden_states, global_outputs = self._prefill(
                input_ids, expanded_input_ids, attention_mask, boundary_mask, past_key_values, **kwargs
            )

        # hidden states and attentions are the global model's, one per patch
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
            hidden_states=global_outputs.hidden_states if global_outputs is not None else None,
            attentions=global_outputs.attentions if global_outputs is not None else None,
        )


@auto_docstring
class BolmoForCausalLM(BolmoPreTrainedModel, BolmoGenerationMixin):
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


class BolmoTokenizer(TokenizersBackend):
    r"""
    Byte tokenizer of Bolmo, built from `tokenizers` primitives: byte-level BPE without merges over 4 special tokens
    and 256 bytes, so `"Hi"` -> `[<bos>, H, i]` = `[1, 76, 109]`. Ids shifted by 260 are the same tokens closing a
    patch (the model generates them), decoded like the plain ones.

    Also returns `expanded_input_ids`: the id of the longest subword of `vocab_file` ending at each byte, which
    `tokenizers` has no primitive for.

    Args:
        vocab_file (`str`):
            Subword vocabulary (`{token: id}`, GPT-2 byte-level strings) used for `expanded_input_ids`.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "expanded_input_ids", "attention_mask"]
    padding_side = "left"
    special_tokens = ["<pad>", "<bos>", "<eos>", "<bpe_token_end>"]

    def __init__(self, vocab_file: str, pad_token="<pad>", bos_token="<bos>", eos_token="<bos>", **kwargs):
        self.offset = len(self.special_tokens)
        self.boundary_offset = self.offset + 256
        self.bpe_token_end_id = self.special_tokens.index("<bpe_token_end>")
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        vocab.update({char: self.offset + byte for byte, char in bytes_to_unicode().items()})
        # `<b>` marks a token closing a patch, it is dropped when decoding
        vocab.update({token + "<b>": i + self.boundary_offset for token, i in list(vocab.items())})
        self._tokenizer = Tokenizer(BPE(vocab=vocab, merges=[]))
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        self._tokenizer.decoder = decoders.Sequence([decoders.Replace("<b>", ""), decoders.ByteLevel()])

        with open(vocab_file, encoding="utf-8") as f:
            self.subword_vocab = json.load(f)
        self._subword_trie = None
        kwargs.setdefault("add_bos_token", True)
        kwargs.setdefault(
            "extra_special_tokens", ["<bpe_token_end>"] + [token + "<b>" for token in self.special_tokens]
        )
        super().__init__(pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, **kwargs)

    @property
    def subword_trie(self):
        """Subword vocab as a trie over reversed byte ids, to find the longest subword ending at a byte."""
        if self._subword_trie is None:
            byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
            self._subword_trie = {}
            for token, subword_id in self.subword_vocab.items():
                if token in self.special_tokens:
                    byte_ids = [self.special_tokens.index(token)]
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
        encoding = super().__call__(*args, **kwargs)
        input_ids = encoding["input_ids"]
        rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        if rows and isinstance(rows[0], list):
            encoding["expanded_input_ids"] = [self.expand_byte_ids(row) for row in rows]
        else:
            encoding["expanded_input_ids"] = self.expand_byte_ids(rows)
        encoding.convert_to_tensors(kwargs.get("return_tensors"))
        return encoding


__all__ = ["BolmoConfig", "BolmoForCausalLM", "BolmoModel", "BolmoPreTrainedModel", "BolmoTokenizer"]
