# coding=utf-8
# Copyright 2023 Stanford University and the HuggingFace Inc. team.
# All rights reserved.
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
"""PyTorch H3 model."""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# TODO remove einops dependency
from einops import rearrange

# custom kernel
from src.models.h3 import H3

from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_h3 import H3Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "stanford/H3-125m"
_CONFIG_FOR_DOC = "H3Config"

H3_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "stanford/H3-125m",
    # See all H3 models at https://huggingface.co/models?filter=h3
]


def stochastic_depth(input: torch.Tensor, p: float, mode: str, training: bool = True) -> torch.Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


class H3StochasticDepth(nn.Module):
    """
    Stochastic depth implementation, taken from Torchvision's op: https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


class H3Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None, word_embed_proj_dim=None):
        """
        If max_position_embeddings <= 0, there's no position embeddings. If word_embed_proj_dim is not None (e.g.,
        OPT-350m), we embed to that dimension the projection up to embed_dim.
        """
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim, padding_idx=padding_idx)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)

    def forward(self, input_ids, position_ids=None):
        seqlen = input_ids.shape[1]
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Args:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    
    Returns:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, "b t h d -> (b h) t d")
    k = rearrange(k, "b s h d -> (b h) d s")
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale), "(b h) t s -> b h t s", h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = nn.functional.dropout(attention, dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    return output.to(dtype=qkv.dtype), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, config, bias=True, attention_dropout=0.0, causal=False):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attention_dropout = attention_dropout
        self.num_heads = config.num_attention_heads
        self.causal = causal

        assert self.embed_dim % self.num_heads == 0, "self.embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
        # self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        qkv = self.Wqkv(hidden_states)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        context, attn_weights = attention_pytorch(qkv, dropout_p=self.attention_dropout, causal=self.causal)
        # TODO support outputting attention weights
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)"))


class H3MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=nn.functional.gelu,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class H3Block(nn.Module):
    """
    Supports either multi-head attention or H3 as mixer class.

    Based on:
    https://github.com/HazyResearch/flash-attention/blob/6b4a48218edb55fb67e087f4df8d7ba4711e75bb/flash_attn/modules/block.py#L22.
    """

    def __init__(
        self,
        config,
        layer_idx,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        drop_path1=0.0,
        drop_path2=0.0,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        if layer_idx in config.attn_layer_idx:
            self.mixer = MultiHeadAttention(config, causal=True)
        else:
            self.mixer = H3(
                d_model=config.hidden_size, layer_idx=layer_idx, mode=config.ssm_mode, measure=config.ssm_measure
            )
        self.dropout1 = nn.Dropout(resid_dropout1)
        self.drop_path1 = H3StochasticDepth(drop_path1, mode="row")
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        self.mlp = H3MLP(
            in_features=config.hidden_size,
            hidden_features=inner_dim,
            activation=partial(nn.functional.gelu, approximate="tanh"),
        )
        self.dropout2 = nn.Dropout(resid_dropout2)
        self.drop_path2 = H3StochasticDepth(drop_path2, mode="row")
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, residual):
        # 1. apply prenorm
        dropped = self.drop_path1(self.dropout1(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # 2. apply mixer
        hidden_states = self.mixer(hidden_states)

        # 3. apply second norm
        dropped = self.drop_path2(self.dropout2(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # 4. apply MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# def create_mixer_cls(config, layer_idx=None):
#     ssm_cfg = dict(mode=config.ssm_mode, measure=config.ssm_measure)
#     attn_layer_idx = config.attn_layer_idx

#     if attn_layer_idx is not None and layer_idx in attn_layer_idx:
#         # attn_cfg = dict(num_heads=config.num_attention_heads)
#         # causal = True if attn_cfg is None else attn_cfg.pop("causal", True)
#         # mixer_cls = partial(MHA, layer_idx=layer_idx, causal=causal, **(attn_cfg if attn_cfg is not None else {}))
#         mixer_cls = partial(MultiHeadAttention(causal=True))
#     else:
#         mixer_cls = partial(H3, layer_idx=layer_idx, **(ssm_cfg if ssm_cfg is not None else {}))
#     return mixer_cls


# def create_mlp_cls(config):
#     inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
#     if not config.fused_mlp:
#         mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=partial(nn.functional.gelu, approximate="tanh"))
#     else:
#         mlp_cls = partial(FusedMLP, hidden_features=inner_dim)
#     return mlp_cls


# def create_block(config, layer_idx):
#     # d_model, n_inner=None,
#     # ssm_cfg=None, attn_layer_idx=None,
#     # attn_cfg=None, layer_norm_epsilon=1e-5,
#     # resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
#     # fused_mlp=False, fused_dropout_add_ln=False, layer_idx=None):
#     fused_dropout_add_ln = config.fused_dropout_add_ln
#     if fused_dropout_add_ln and dropout_add_layer_norm is None:
#         raise ImportError("dropout_add_layer_norm is not installed")

#     # a block consists of a mixer class, an MLP and a layernorm class
#     mixer_cls = create_mixer_cls(config, layer_idx=layer_idx)
#     mlp_cls = create_mlp_cls(config)
#     norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_epsilon)
#     block = Block(
#         config.hidden_size,
#         mixer_cls,
#         mlp_cls,
#         norm_cls=norm_cls,
#         prenorm=True,
#         resid_dropout1=config.embedding_dropout if layer_idx == 0 else config.residual_dropout,
#         resid_dropout2=config.residual_dropout,
#         fused_dropout_add_ln=fused_dropout_add_ln,
#         residual_in_fp32=config.residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block


class H3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = H3Config
    base_model_prefix = "h3"
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    nn.init.normal_(
                        p, mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
                    )
                # If using GLU activation for now, we scale the std by 2
                elif name in ["output_linear.0.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    if not self.config.glu_act:
                        nn.init.normal_(
                            p,
                            mean=0.0,
                            std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
                        )
                    else:
                        out_features = p.shape[0]
                        # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                        # on average.
                        nn.init.normal_(
                            p[: out_features // 2],
                            mean=0.0,
                            std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers) * 2,
                        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, H3Model):
            module.gradient_checkpointing = value


H3_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`H3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

H3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare H3 model outputting raw hidden-states without any specific head on top.",
    H3_START_DOCSTRING,
)
class H3Model(H3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = H3Embeddings(config.hidden_size, config.vocab_size, 0)
        self.residual_in_fp32 = config.residual_in_fp32

        self.blocks = nn.ModuleList(
            [
                H3Block(
                    config,
                    layer_idx=i,
                    resid_dropout1=config.embedding_dropout if i == 0 else config.residual_dropout,
                    resid_dropout2=config.residual_dropout,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_dropout = nn.Dropout(config.residual_dropout)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(H3_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if past_key_values is None:
        #     past_key_values = tuple([None] * len(self.h))

        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        else:
            hidden_states = inputs_embeds

        residual = None

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                )
            else:
                # print("Hidden states before block:", hidden_states.shape)
                # print("Residual before block:", residual)
                # print("Block:", i)
                hidden_states, residual = block(hidden_states, residual)

            # hidden_states = outputs[0]
            # if use_cache is True:
            #     presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        
        dropped = self.final_dropout(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.final_layernorm(residual.to(dtype=self.final_layernorm.weight.dtype))

        # hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        # TODO use class without cross-attentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


@add_start_docstrings(
    """
    The H3 model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    H3_START_DOCSTRING,
)
class H3ForCausalLM(H3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.h3 = H3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        self.lm_head.weight = self.h3.embeddings.word_embeddings.weight

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @add_start_docstrings_to_model_forward(H3_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.h3(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # TODO use class without cross-attentions
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=None,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
