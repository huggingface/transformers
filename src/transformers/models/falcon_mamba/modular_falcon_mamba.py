# Copyright 2024 Tri Dao, Albert Gu, Technological Innovation Institute and HuggingFace Inc. team.
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
"""PyTorch FALCONMAMBA model."""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...utils import auto_docstring, logging
from ..llama.modeling_llama import LlamaRMSNorm
from ..mamba.configuration_mamba import MambaConfig
from ..mamba.modeling_mamba import (
    MambaBlock,
    MambaCausalLMOutput,
    MambaForCausalLM,
    MambaMixer,
    MambaModel,
    MambaOutput,
    MambaPreTrainedModel,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    mamba_inner_fn,
    mamba_selective_scan,
    mamba_selective_state_update,
)
from ..nanochat.modeling_nanochat import NanoChatRMSNorm


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="tiiuae/falcon-mamba-7b")
@strict
class FalconMambaConfig(MambaConfig):
    r"""
    expand (`int`, *optional*, defaults to 2):
        Expanding factor used to determine the intermediate size.
    conv_kernel (`int`, *optional*, defaults to 4):
        Size of the convolution kernel.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the convolution layer of the mixer block.
    residual_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
    rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
        Whether or not to rescale `out_proj` weights when initializing.
    use_falcon_mambapy (`bool`, *optional*, defaults to `False`):
        This argument corresponds to `use_mambapy` in MambaConfig.
        Determines the fallback strategy during training if the CUDA-based official implementation of Mamba is not available. If `True`, the mamba.py implementation is used. If `False`, the naive and slower implementation is used. Consider switching to the naive version if memory is limited.
    use_associative_scan (`bool`, *optional*, defaults to `True`):
        Whether to use PyTorch's `torch._higher_order_ops.associative_scan` for the parallel scan instead of the naive
        sequential implementation. The associative scan is only active during `torch.compile` tracing and
        requires torch >= 2.9.0. Both paths are tested to produce numerically identical results (see
        `test_associative_scan_matches_sequential`). Set to `False` to fall back to the sequential loop.
    mixer_rms_eps (`float`, *optional*, defaults to 1e-06):
        The RMS norm epsilon value that is used in the Mixer RMS norm for B, C and dt states.

    Example:

    ```python
    >>> from transformers import FalconMambaConfig, FalconMambaModel

    >>> # Initializing a FalconMamba configuration
    >>> configuration = FalconMambaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FalconMambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    use_falcon_mambapy: bool = False
    use_associative_scan: bool = True
    mixer_rms_eps: float = 1e-6

    @property
    def layer_types(self):
        return ["linear_attention"] * self.num_hidden_layers


class FalconMambaWeightlessRMSNorm(NanoChatRMSNorm):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__(eps)
        # Dummy weights that are not used (only for imitating on kernels path)
        self.register_buffer("weight", torch.ones(hidden_size, requires_grad=False), persistent=False)


class FalconMambaMixer(MambaMixer):
    def __init__(self, config: FalconMambaConfig, layer_idx: int, initialize_mixer_weights: bool = True):
        super().__init__(config, layer_idx, initialize_mixer_weights)
        self.dt_layernorm = FalconMambaWeightlessRMSNorm(self.intermediate_size, eps=config.mixer_rms_eps)
        self.b_layernorm = FalconMambaWeightlessRMSNorm(self.ssm_state_size, eps=config.mixer_rms_eps)
        self.c_layernorm = FalconMambaWeightlessRMSNorm(self.ssm_state_size, eps=config.mixer_rms_eps)
        self.rms_eps = config.mixer_rms_eps

    @torch.no_grad()
    def init_falcon_mamba_weights(self):
        super().init_falcon_mamba_weights()
        init.ones_(self.dt_layernorm.weight)
        init.ones_(self.b_layernorm.weight)
        init.ones_(self.c_layernorm.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        if self.training and cache_params is None:
            fused_output = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                # Key difference: norms on B, C, and dt
                b_rms_weight=self.b_layernorm.weight,
                c_rms_weight=self.c_layernorm.weight,
                dt_rms_weight=self.dt_layernorm.weight,
                b_c_dt_rms_eps=self.rms_eps,
            )

            # Only kernels can use this shortcircuit, fallback to normal torch otherwise
            if fused_output is not None:
                return fused_output

        hidden_states_B_C, gate = projected_states.chunk(2, dim=1)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # 2. Convolution sequence transformation
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )
        else:
            if cache_params is not None:
                hidden_states_B_C = cache_params.update_conv_state(
                    hidden_states_B_C,
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )

            hidden_states_B_C = causal_conv1d_fn(
                hidden_states_B_C,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            if cache_params is not None:
                hidden_states_B_C = hidden_states_B_C[:, :, -seq_len:]

        # 3. SSM transformation
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C.transpose(1, 2), attention_mask)
        time_step, B, C = torch.split(
            self.x_proj(hidden_states_B_C),
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        # Key difference: Additional norms on B, C, and dt
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        # In case the model has been quantized, we need a hack to properly call the `nn.Linear` module
        # at the price of a small overhead.
        if hasattr(self.config, "_is_quantized"):
            time_step = (self.dt_proj(time_step) - self.dt_proj.bias).transpose(1, 2)
        else:
            time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        time_proj_bias = self.dt_proj.bias.float() if self.dt_proj.bias is not None else None

        # Recurrent form
        if use_precomputed_states and seq_len == 1:
            scan_output = mamba_selective_state_update(
                recurrent_state,
                hidden_states_B_C.transpose(1, 2)[..., 0],
                time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                z=gate[..., 0],
                dt_bias=time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)

        # Full sequence form
        else:
            output_final_state = cache_params is not None
            scan_result = mamba_selective_scan(
                hidden_states_B_C.transpose(1, 2),
                time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                D=self.D.float(),
                z=gate,
                delta_bias=time_proj_bias,
                delta_softplus=True,
                return_last_state=output_final_state,
                # TODO: rename to normal mambapy
                use_mambapy=self.use_falcon_mambapy,
                use_associative_scan=self.use_associative_scan,
            )

            if output_final_state:
                scan_output, final_state = scan_result
                cache_params.update_recurrent_state(final_state, self.layer_idx)
            else:
                scan_output = scan_result

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2).to(dtype))
        return contextualized_states


class FalconMambaRMSNorm(LlamaRMSNorm):
    pass


class FalconMambaBlock(MambaBlock):
    pass


@auto_docstring
class FalconMambaPreTrainedModel(MambaPreTrainedModel):
    pass


class FalconMambaOutput(MambaOutput):
    pass


class FalconMambaCausalLMOutput(MambaCausalLMOutput):
    pass


class FalconMambaModel(MambaModel, FalconMambaPreTrainedModel):
    def __init__(self, config):
        FalconMambaPreTrainedModel.__init__(self, config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FalconMambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False
        self.norm_f = FalconMambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        raise AttributeError("Not needed for FalconMamba")


class FalconMambaForCausalLM(MambaForCausalLM):
    pass


__all__ = [
    "FalconMambaForCausalLM",
    "FalconMambaModel",
    "FalconMambaPreTrainedModel",
    "FalconMambaConfig",
]
