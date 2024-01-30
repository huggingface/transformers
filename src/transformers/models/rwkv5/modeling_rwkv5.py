# coding=utf-8
# Copyright 2023 Bo Peng and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch RWKV5 World model."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_bitsandbytes_available,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)

from .configuration_rwkv5 import Rwkv5Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RWKV/rwkv-5-world-1b5"
_CONFIG_FOR_DOC = "Rwkv5Config"

RWKV5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "RWKV/rwkv-5-world-1b5",
    "RWKV/rwkv-5-world-3b",
    # See all RWKV models at https://huggingface.co/models?filter=rwkv
]

rwkv5_cuda_kernel = None


# Copied from https://github.com/huggingface/transformers/blob/18cbaf13dcaca7145f5652aefb9b19734c56c3cd/src/transformers/models/rwkv/modeling_rwkv.py#L65
def load_wkv5_cuda_kernel(head_size):
    from torch.utils.cpp_extension import load as load_kernel

    global rwkv5_cuda_kernel

    kernel_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "rwkv5"
    cuda_kernel_files = [kernel_folder / f for f in ["wkv5_op.cpp", "wkv5_cuda.cu"]]

    # Only load the kernel if it's not been loaded yet or if we changed the context length
    if rwkv5_cuda_kernel is not None and rwkv5_cuda_kernel.head_size == head_size:
        return

    logger.info(f"Loading CUDA kernel for RWKV at head size of {head_size}.")

    flags = [
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-D_N_={head_size}",
    ]
    rwkv5_cuda_kernel = load_kernel(
        name=f"wkv_{head_size}",
        sources=cuda_kernel_files,
        verbose=(logging.get_verbosity() == logging.DEBUG),
        extra_cuda_cflags=flags,
    )
    rwkv5_cuda_kernel.head_size = head_size


class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, receptance, key, value, time_decay, time_first, state):
        with torch.no_grad():
            Batch = key.shape[0]
            SequenceLength = key.shape[1]
            HiddenSize = key.shape[2]
            HeadSize = HiddenSize // time_decay.shape[0]
            ctx.Batch = Batch
            ctx.SequenceLength = SequenceLength
            ctx.HiddenSize = HiddenSize
            ctx.HeadSize = HeadSize
            e_time_decay = (-torch.exp(time_decay.float())).contiguous()
            ee_time_decay = (torch.exp(e_time_decay)).contiguous()
            ctx.save_for_backward(receptance, key, value, ee_time_decay, e_time_decay, time_first)
            out = torch.empty(
                (Batch, SequenceLength, HiddenSize),
                device=receptance.device,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            rwkv5_cuda_kernel.forward(
                Batch,
                SequenceLength,
                HiddenSize,
                HeadSize,
                receptance,
                key,
                value,
                ee_time_decay,
                time_first,
                out,
                state,
            )
            return out, state

    @staticmethod
    def backward(ctx, gout):
        with torch.no_grad():
            assert gout.dtype == torch.bfloat16
            Batch = ctx.Batch
            SequenceLength = ctx.SequenceLength
            HiddenSize = ctx.HiddenSize
            HeadSize = ctx.HeadSize
            receptance, key, value, ee_time_decay, e_time_decay, time_first = ctx.saved_tensors
            greceptance = torch.empty(
                (Batch, SequenceLength, HiddenSize),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_key = torch.empty(
                (Batch, SequenceLength, HiddenSize),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_value = torch.empty(
                (Batch, SequenceLength, HiddenSize),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_time_decay = torch.empty(
                (Batch, HiddenSize),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            g_time_first = torch.empty(
                (Batch, HiddenSize),
                device=gout.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )
            rwkv5_cuda_kernel.backward(
                Batch,
                SequenceLength,
                HiddenSize,
                HeadSize,
                receptance,
                key,
                value,
                ee_time_decay,
                e_time_decay,
                time_first,
                gout,
                greceptance,
                g_key,
                g_value,
                g_time_decay,
                g_time_first,
            )
            g_time_decay = torch.sum(g_time_decay, 0).view(HeadSize, HiddenSize // HeadSize)
            g_time_first = torch.sum(g_time_first, 0).view(HeadSize, HiddenSize // HeadSize)
            return (None, None, None, None, greceptance, g_key, g_value, g_time_decay, g_time_first)


def rwkv_linear_attention_v5_cpu(
    hidden,
    time_decay,
    time_first,
    receptance,
    key,
    value,
    gate,
    layer_norm_weight,
    layer_norm_bias,
    output_weight,
    state,
):
    Batch = hidden.shape[0]
    AttentionHeads = time_decay.shape[0]
    HeadSize = hidden.shape[-1] // AttentionHeads
    SequenceLength = hidden.shape[1]
    key = key.to(torch.float32).view(Batch, SequenceLength, AttentionHeads, HeadSize).transpose(1, 2).transpose(-2, -1)
    value = value.to(torch.float32).view(Batch, SequenceLength, AttentionHeads, HeadSize).transpose(1, 2)
    receptance = receptance.to(torch.float32).view(Batch, SequenceLength, AttentionHeads, HeadSize).transpose(1, 2)
    time_decay = torch.exp(-torch.exp(time_decay.float())).reshape(-1, 1, 1).reshape(AttentionHeads, -1, 1)
    time_first = time_first.float().reshape(-1, 1, 1).reshape(AttentionHeads, -1, 1)
    layer_norm_weight = layer_norm_weight.float()
    layer_norm_bias = layer_norm_bias.float()
    out = torch.zeros_like(key).reshape(Batch, SequenceLength, AttentionHeads, HeadSize)
    for t in range(SequenceLength):
        rt = receptance[:, :, t : t + 1, :]
        kt = key[:, :, :, t : t + 1]
        vt = value[:, :, t : t + 1, :]
        at = kt @ vt
        out[:, t] = (rt @ (time_first * at + state)).squeeze(2)
        with torch.no_grad():
            state = at + time_decay * state

    out = out.reshape(Batch * SequenceLength, AttentionHeads * HeadSize)
    out = F.group_norm(out, num_groups=AttentionHeads, weight=layer_norm_weight, bias=layer_norm_bias).reshape(
        Batch, SequenceLength, AttentionHeads * HeadSize
    )
    out = out.to(dtype=hidden.dtype) * gate
    out = out @ output_weight

    return out, state


def rwkv_linear_attention(
    hidden,
    time_decay,
    time_first,
    receptance,
    key,
    value,
    gate,
    layer_norm_weight,
    layer_norm_bias,
    output_weight,
    state,
):
    Batch = hidden.shape[0]
    AttentionHeads = time_decay.shape[0]
    HeadSize = hidden.shape[-1] // AttentionHeads
    SequenceLength = hidden.shape[1]
    no_cuda = any(t.device.type != "cuda" for t in [time_decay, time_first, receptance, key, value])
    # Launching the CUDA kernel for just one token will actually be slower (there is no for loop in the CPU version
    # in this case).
    one_token = key.size(1) == 1
    if rwkv5_cuda_kernel is None or no_cuda or one_token:
        return rwkv_linear_attention_v5_cpu(
            hidden,
            time_decay,
            time_first,
            receptance,
            key,
            value,
            gate,
            layer_norm_weight,
            layer_norm_bias,
            output_weight,
            state,
        )
    else:
        out, state = WKV_5.apply(
            Batch,
            SequenceLength,
            AttentionHeads * HeadSize,
            AttentionHeads,
            receptance,
            key,
            value,
            time_decay,
            time_first,
            state,
        )
        out = out.reshape(Batch * SequenceLength, AttentionHeads * HeadSize)
        out = F.group_norm(out, num_groups=AttentionHeads, weight=layer_norm_weight, bias=layer_norm_bias).reshape(
            Batch, SequenceLength, AttentionHeads * HeadSize
        )
        out = out.to(dtype=hidden.dtype) * gate
        out = out @ output_weight
        return out, state


class RwkvSelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        kernel_loaded = rwkv5_cuda_kernel is not None and rwkv5_cuda_kernel.head_size == config.head_size
        if is_ninja_available() and is_torch_cuda_available() and not kernel_loaded:
            try:
                load_wkv5_cuda_kernel(config.context_length)
            except Exception:
                logger.info("Could not load the custom CUDA kernel for RWKV5 attention.")
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        num_attention_heads = hidden_size // config.head_size
        self.num_attention_heads = num_attention_heads
        attention_hidden_size = (
            config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        )
        self.attention_hidden_size = attention_hidden_size

        self.time_decay = nn.Parameter(torch.empty(num_attention_heads, config.head_size))
        self.time_faaaa = nn.Parameter(torch.empty(num_attention_heads, config.head_size))
        self.time_mix_gate = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(hidden_size // config.head_size, hidden_size)

    def extract_key_value(self, hidden, state=None):
        # Mix hidden with the previous timestep to produce key, value, receptance
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        gate = hidden * self.time_mix_gate + shifted * (1 - self.time_mix_gate)

        key = self.key(key)
        value = self.value(value)
        receptance = self.receptance(receptance)
        gate = F.silu(self.gate(gate))

        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        return receptance, key, value, gate, state

    def forward(self, hidden, state=None, use_cache=False, seq_mode=True):
        receptance, key, value, gate, state = self.extract_key_value(hidden, state=state)
        layer_state = state[1][:, :, :, :, self.layer_id] if state is not None else None
        rwkv, layer_state = rwkv_linear_attention(
            hidden,
            self.time_decay,
            self.time_faaaa,
            receptance,
            key,
            value,
            gate,
            self.ln_x.weight,
            self.ln_x.bias,
            self.output.weight.t(),
            state=layer_state,
        )

        if layer_state is not None:
            state[1][:, :, :, :, self.layer_id] = layer_state

        return rwkv, state


class RwkvFeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else int((config.hidden_size * 3.5) // 32 * 32)
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[2][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[2][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance))

        if state is not None:
            state[2][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state


# copied from HuggingFace https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
class Rwkv5Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.attention = RwkvSelfAttention(config, layer_id)
        self.feed_forward = RwkvFeedForward(config, layer_id)

    def forward(self, hidden, state=None, use_cache=False, output_attentions=False, seq_mode=True):
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)
        attention, state = self.attention(self.ln1(hidden), state=state, use_cache=use_cache, seq_mode=seq_mode)
        hidden = hidden + attention

        feed_forward, state = self.feed_forward(self.ln2(hidden), state=state)
        hidden = hidden + feed_forward

        outputs = (hidden, state)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs


class Rwkv5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Rwkv5Config
    base_model_prefix = "rwkv"
    _no_split_modules = ["Rwkv5Block"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, RwkvSelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            num_attention_heads = hidden_size // module.config.num_attention_heads

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -6.0 + 5.0 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            tmp = torch.tensor(
                [
                    (1.0 - (i / (attention_hidden_size - 1.0))) * ratio_0_to_1 + 0.1 * ((i + 1) % 3 - 1)
                    for i in range(attention_hidden_size)
                ],
                dtype=module.time_faaaa.dtype,
                device=module.time_faaaa.device,
            )

            with torch.no_grad():
                module.time_decay.data = decay_speed.reshape(num_attention_heads, module.config.num_attention_heads)
                module.time_faaaa.data = tmp.reshape(num_attention_heads, module.config.num_attention_heads)
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)

                module.time_mix_value.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.time_mix_receptance.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
                module.time_mix_gate.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)

        elif isinstance(module, RwkvFeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)


# copied from HuggingFace https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
@dataclass
class Rwkv5Output(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# copied from HuggingFace https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
@dataclass
class Rwkv5CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


RWKV_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`Rwkv5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary. If `past_key_values` is used, only `input_ids` that do not have their
            past calculated should be passed as `input_ids`. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
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
    "The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV_START_DOCSTRING,
)
class Rwkv5Model(Rwkv5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv5Block(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)

        self.layers_are_rescaled = False
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Rwkv5Output,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Rwkv5Output]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # rwkv5 only support inference in huggingface.
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training == self.layers_are_rescaled and (
            self.embeddings.weight.dtype == torch.float16 or self.embeddings.weight.dtype == torch.bfloat16
        ):
            self._rescale_layers()

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and state is None:
            state = []
            num_attention_heads = self.config.hidden_size // self.config.num_attention_heads
            state_attn_x = torch.zeros(
                (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                dtype=inputs_embeds.dtype,
                requires_grad=False,
                device=inputs_embeds.device,
            ).contiguous()
            state_attn_kv = torch.zeros(
                (
                    inputs_embeds.size(0),
                    num_attention_heads,
                    self.config.hidden_size // num_attention_heads,
                    self.config.hidden_size // num_attention_heads,
                    self.config.num_hidden_layers,
                ),
                dtype=torch.float32,
                requires_grad=False,
                device=inputs_embeds.device,
            ).contiguous()
            state_ffn_x = torch.zeros(
                (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                dtype=inputs_embeds.dtype,
                requires_grad=False,
                device=inputs_embeds.device,
            ).contiguous()
            state.append(state_attn_x)
            state.append(state_attn_kv)
            state.append(state_ffn_x)

        seq_mode = inputs_embeds.shape[1] > 1
        hidden_states = inputs_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.blocks):
            hidden_states, state, attentions = block(
                hidden_states, state=state, use_cache=use_cache, output_attentions=output_attentions, seq_mode=seq_mode
            )
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, state, all_hidden_states, all_self_attentions)

        return Rwkv5Output(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,  # None
            attentions=all_self_attentions,  # None
        )

    def _rescale_layers(self):
        # Layers should be rescaled for inference only.
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    else:
                        block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.div_(2 ** int(block_id // self.config.rescale_every))

        self.layers_are_rescaled = not self.training

    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        r"""
        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will
        be quantized again.
        """
        if not is_bitsandbytes_available():
            raise ImportError("Please install bitsandbytes to use this method.")
        import bitsandbytes as bnb

        dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)

        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))

        # re-quantize the model:
        # we need to put it first on CPU then back to the device
        # this will create an overhead :/
        # We set requires_grad=False as we cannot compute gradients on top of 4bit parameters anyway and to avoid
        # bugs with bnb
        quant_weight = bnb.nn.Params4bit(dequant_weights.to("cpu"), requires_grad=False).to(dequant_weights.device)
        setattr(target_layer, "weight", quant_weight)


# copied from HuggingFace https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py
@add_start_docstrings(
    """
    The RWKV5 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    RWKV_START_DOCSTRING,
)
class Rwkv5ForCausalLM(Rwkv5PreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = Rwkv5Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Rwkv5CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Rwkv5CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = rwkv_outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Rwkv5CausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )
