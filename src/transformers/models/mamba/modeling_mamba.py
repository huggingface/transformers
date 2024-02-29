# coding=utf-8
# Copyright 2024 Tri Dao, Albert Gu and HuggingFace Inc. team.
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
"""PyTorch MAMBA model."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig


logger = logging.get_logger(__name__)

if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    logger.warning_once(
        " `mamba_ssm` is not installed in your environnement. Make sure to install it following `src/transformers/kernels/mamba/Makefile`"
    )
    selective_state_update, selective_scan_fn = None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    logger.warning_once(
        " `causal_conv1d` is not installed in your environnement. Make sure to install it: `src/transformers/kernels/mamba/Makefile`"
    )
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = not any((selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update)) is None

_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m"
_CONFIG_FOR_DOC = "MambaConfig"

MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "state-spaces/mamba-130m",
    "state-spaces/mamba-370m",
    "state-spaces/mamba-790m",
    "state-spaces/mamba-1.4b",
    "state-spaces/mamba-2.8b",
    "state-spaces/mamba-2.8b-slimpj",
]  # See all Mamba models at https://huggingface.co/models?filter=mamba


class MambaMixer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.expand = config.expand
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # THe core is to load them, compute the discrete states, then write the updates state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(
            torch.log(A)
        )  # TODO this parameter should be kept in float32. We don't have support for that I think

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.intermediate_size))  # Keep in fp32
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, inference_params=None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if inference_params is not None and inference_params.seqlen_offset > 0:
            conv_state = inference_params.conv_states[self.layer_idx]
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1), conv_state, conv_weights, self.conv1d.bias, self.activation
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
            inference_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        x_dbl = self.x_proj(hidden_states.transpose(1, 2))  # TODO find a better name for this one
        time_step, B, C = torch.split(x_dbl, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

        # 3.b. discretize time_step, B and C: zero-order hold from (B,L,D) to  (B,L,D,N)
        A = -torch.exp(self.A_log.float())

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        ssm_state = inference_params.ssm_states[self.layer_idx]
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        if inference_params is not None and inference_params.seqlen_offset > 0:
            y = selective_state_update(
                ssm_state,
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            y, last_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if last_state is not None:
                inference_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        selected_states = self.out_proj(y.transpose(1, 2))
        return selected_states


    def slow_forward(self, hidden_states, inference_params=None):
        """

         Compute ∆ A B C D, the state space parameters.
             A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
             ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
                                          and is why Mamba is called **selective** state spaces)

        Args:
            hidden_states:
            inference_params:

        Returns:

        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if inference_params.seqlen_offset > 0:
            conv_state = inference_params.conv_states[self.layer_idx]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states[:, :, 0]

            hidden_states = self.act(torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1) + self.conv1d.bias)
            hidden_states = hidden_states.unsqueeze(-1)

        else:
            inference_params.conv_states[self.layer_idx] = (
                nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        x_dbl = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(x_dbl, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
        discrete_time_step = self.dt_proj(time_step)
        A = -torch.exp(self.A_log.float())  # (intermediate_size, ssm_state_size)

        # 3.b. discretize time_step, B and C: zero-order hold from (B,L,D) to  (B,L,D,N)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)
        #                [batch_size, d, l, 1]     X        [1, d, 1, n] ->  [batch_size, d, l, n]
        dA = torch.exp(discrete_time_step[:, :, :, None] * A[None, :, None, :])
        #     [batch_size, d, l, 1]   [b, d, l, 1]  ->  [batch_size, d, l, 1]  X  [batch_size, 1, l, n] -> [batch_size, d, l, n]
        deltaB_u = (discrete_time_step[:, :, :, None] * hidden_states[:, :, :, None]) * B[:, None, :, :]

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        ssm_state = inference_params.ssm_states[self.layer_idx]
        ys = []
        for i in range(seq_len):
            ssm_state = (ssm_state * dA[:, :, i, :] + deltaB_u[:, :, i, :])
            #    [b, d, n]   X  [b, n] -> [b, d]
            y = torch.matmul(ssm_state, C[:, i, :].unsqueeze(-1))
            ys.append(y[:, :, 0])
        y = torch.stack(ys, dim=-1)  # shape (b, l, d)
        y = y + (hidden_states * self.D[None, :, None])
        y = y * self.act(gate)  # (B D)
        # 4. Final linear projection
        selected_states = self.out_proj(y.transpose(1, 2))

        inference_params.ssm_states[self.layer_idx] = ssm_state

        return selected_states

    def forward(self, hidden_states, inference_params=None):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, inference_params)
        return self.slow_forward(hidden_states, inference_params)


class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.seqlen_offset = 0

        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, inference_params=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = residual.to(torch.float32) + hidden_states
        return hidden_states


class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = "mamba"
    _no_split_modules = ["MambaBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            pass
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        # TODO make sure we properly init
        # self.A_log._no_weight_decay = True
        # self.D._no_weight_decay = True
        #
        # # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # dt = torch.exp(
        #     torch.rand(self.intermediate_size, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        #     + math.log(dt_min)
        # ).clamp(min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # inv_dt = dt + torch.log(-torch.expm1(-dt))
        # with torch.no_grad():
        #     self.dt_proj.bias.copy_(inv_dt)
        # # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # self.dt_proj.bias._no_reinit = True

        # if isinstance(module, nn.Linear):
        #     if module.bias is not None:
        #         if not getattr(module.bias, "_no_reinit", False):
        #             nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     nn.init.normal_(module.weight, std=initializer_range)
        #
        # if rescale_prenorm_residual:
        #     # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #     #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #     #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #     #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #     #
        #     # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        #     for name, p in module.named_parameters():
        #         if name in ["out_proj.weight", "fc2.weight"]:
        #             # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
        #             # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
        #             # We need to reinit p since this code could be called multiple times
        #             # Having just p *= scale would repeatedly scale it down
        #             nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        #             with torch.no_grad():
        #                 p /= math.sqrt(n_residuals_per_layer * n_layer)


@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        inference_params (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model states weights after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    inference_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        inference_params (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    inference_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


MAMBA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MambaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MAMBA_INPUTS_DOCSTRING = r"""
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
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        inference_params (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MAMBA Model transformer outputting raw hidden-states without any specific head on top.",
    MAMBA_START_DOCSTRING,
)
class MambaModel(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        inference_params: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if inference_params is None:
            inference_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
                )
                use_cache = False

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, inference_params
                )
            else:
                hidden_states = mixer_block(hidden_states, inference_params=inference_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            inference_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, inference_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            inference_params=inference_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    The MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MAMBA_START_DOCSTRING,
)
class MambaForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        model_kwargs["inference_params"] = outputs["inference_params"]
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, inference_params=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if inference_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and inference_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["inference_params"] = inference_params
        return model_inputs

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inference_params: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            inference_params=inference_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

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
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            inference_params=mamba_outputs.inference_params,
            hidden_states=mamba_outputs.hidden_states,
        )
