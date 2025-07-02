# coding=utf-8
# Copyright 2022 The Trajectory Transformers paper authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch TrajectoryTransformer model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_trajectory_transformer import TrajectoryTransformerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
_CONFIG_FOR_DOC = "TrajectoryTransformerConfig"


def load_tf_weights_in_trajectory_transformer(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


@dataclass
class TrajectoryTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple[tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`). Contains pre-computed hidden-states (key and values in the
            attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. GPT2Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class TrajectoryTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TrajectoryTransformerConfig
    load_tf_weights = load_tf_weights_in_trajectory_transformer
    base_model_prefix = "trajectory_transformer"
    main_input_name = "trajectories"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, EinLinear):
            for i in range(module.n_models):
                nn.init.kaiming_uniform_(module.weight[i], a=math.sqrt(5) / self.config.kaiming_initializer_range)
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight[i])
                    bound = (1 / math.sqrt(fan_in)) * self.config.initializer_range
                    nn.init.uniform_(module.bias[i], -bound, bound)


TRAJECTORY_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TrajectoryTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        trajectories (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Batch of trajectories, where a trajectory is a sequence of states, actions and rewards.
        past_key_values (`tuple[tuple[torch.Tensor]]` of length `config.n_layers`, *optional*):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        targets (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Desired targets used to compute the loss.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
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


class EinLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
        Args:
            input (`torch.FloatTensor` of shape `(B, n_models, input_dim)`):
                The input to the layer.
        """
        # [ batch_size x n_models x output_dim ]
        output = torch.einsum("eoi,bei->beo", self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_head ({config.n_head}) should be a divisor of n_embd ({config.n_embd})")

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

        # mask previous value estimates
        joined_dim = config.observation_dim + config.action_dim + 2
        self.mask.squeeze()[:, joined_dim - 1 :: joined_dim] = 0

        self.n_head = config.n_head

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        layer_past: Optional[tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        batch_size, sequence_length, embedding_dim = hidden_states.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # [ batch_size x n_heads x sequence_length x head_dim ]
        key = (
            self.key(hidden_states)
            .view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head)
            .transpose(1, 2)
        )
        query = (
            self.query(hidden_states)
            .view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head)
            .transpose(1, 2)
        )
        value = (
            self.value(hidden_states)
            .view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head)
            .transpose(1, 2)
        )

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # causal self-attention
        # [ batch_size x n_heads x sequence_length x sequence_length ]
        attn_weights = (torch.matmul(query, key.transpose(-2, -1))) * (1.0 / math.sqrt(key.size(-1)))
        attn_weights = attn_weights.masked_fill(
            self.mask[:, :, :sequence_length, :sequence_length] == 0, torch.finfo(attn_weights.dtype).min
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        self._attn_map = attn_weights.clone()
        attn_weights = self.attn_drop(attn_weights)

        output = torch.matmul(attn_weights, value)
        # [ batch_size x sequence_length x embedding_dim ]
        # re-assemble all head outputs side by side
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)

        # output projection
        output = self.resid_drop(self.proj(output))

        outputs = (output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Block(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        # MLP
        self.l1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.l2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        layer_past: Optional[tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        attn_outputs = self.attn(
            hidden_states, layer_past=layer_past, use_cache=use_cache, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.l1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.l2(hidden_states)
        hidden_states = residual + self.drop(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


@add_start_docstrings(
    "The bare TrajectoryTransformer Model transformer outputting raw hidden-states without any specific head on top.",
    TRAJECTORY_TRANSFORMER_START_DOCSTRING,
)
class TrajectoryTransformerModel(TrajectoryTransformerPreTrainedModel):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__(config)

        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)

        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size

        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.embedding_dim = config.n_embd

        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight

        self.gradient_checkpointing = False

        self.post_init()

    def get_block_size(self):
        return self.block_size

    def offset_tokens(self, trajectories):
        _, sequence_length = trajectories.shape

        n_states = int(np.ceil(sequence_length / self.transition_dim))

        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(trajectories.device)

        offset_trajectories = trajectories + offsets[:sequence_length]
        offset_trajectories[trajectories == self.vocab_size] = self.stop_token
        return offset_trajectories

    def pad_to_full_observation(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape

        n_pad = (self.transition_dim - sequence_length % self.transition_dim) % self.transition_dim
        padding = torch.zeros(batch_size, n_pad, self.embedding_dim, device=hidden_states.device)

        # [ batch_size x padded_sequence_length' x embedding_dim ]
        hidden_states_pad = torch.cat([hidden_states, padding], dim=1)
        hidden_states_pad = hidden_states_pad.view(-1, self.transition_dim, self.embedding_dim)

        return hidden_states_pad, n_pad

    @add_start_docstrings_to_model_forward(
        TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TrajectoryTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        trajectories: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        targets: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TrajectoryTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import TrajectoryTransformerModel
        >>> import torch

        >>> model = TrajectoryTransformerModel.from_pretrained(
        ...     "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
        ... )
        >>> model.to(device)
        >>> model.eval()

        >>> observations_dim, action_dim, batch_size = 17, 6, 256
        >>> seq_length = observations_dim + action_dim + 1

        >>> trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(
        ...     device
        ... )
        >>> targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)

        >>> outputs = model(
        ...     trajectories,
        ...     targets=targets,
        ...     use_cache=True,
        ...     output_attentions=True,
        ...     output_hidden_states=True,
        ...     return_dict=True,
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        batch_size, sequence_length = trajectories.size()

        if sequence_length > self.block_size:
            raise ValueError("Cannot forward, model block size is exhausted.")

        offset_trajectories = self.offset_tokens(trajectories)
        # [ batch_size x sequence_length x embedding_dim ]
        # forward the GPT model
        token_embeddings = self.tok_emb(offset_trajectories)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :sequence_length, :]  # each position maps to a (learnable) vector

        hidden_states = self.drop(token_embeddings + position_embeddings)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(hidden_states, layer_past, use_cache, output_attentions)

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # [ batch_size x sequence_length x embedding_dim ]
        hidden_state = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states_pad, n_pad = self.pad_to_full_observation(hidden_state)

        logits = self.head(hidden_states_pad)
        logits = logits.reshape(batch_size, sequence_length + n_pad, self.vocab_size + 1)
        logits = logits[:, :sequence_length]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction="none")
            if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                # make weights
                n_states = int(np.ceil(sequence_length / self.transition_dim))
                weights = torch.cat(
                    [
                        torch.ones(self.observation_dim, device=trajectories.device),
                        torch.ones(self.action_dim, device=trajectories.device) * self.action_weight,
                        torch.ones(1, device=trajectories.device) * self.reward_weight,
                        torch.ones(1, device=trajectories.device) * self.value_weight,
                    ]
                )
                weights = weights.repeat(n_states)
                weights = weights[1:].repeat(batch_size, 1)
                loss = loss * weights.view(-1)
            loss = (loss * attention_mask.view(-1)).mean()
        else:
            loss = None

        if not return_dict:
            return tuple(v for v in [loss, logits, presents, all_hidden_states, all_self_attentions] if v is not None)

        return TrajectoryTransformerOutput(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


__all__ = [
    "TrajectoryTransformerModel",
    "TrajectoryTransformerPreTrainedModel",
    "load_tf_weights_in_trajectory_transformer",
]
