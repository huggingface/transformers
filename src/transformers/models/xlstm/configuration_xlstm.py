# Copyright 2025 NXAI GmbH. All rights reserved.
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

"""xLSTM configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, is_xlstm_available


if is_xlstm_available():
    from xlstm.xlstm_large.model import (
        BackendModeType,
        ChunkwiseKernelType,
        DtypeType,
        SequenceKernelType,
        StepKernelType,
        WeightModeType,
        round_up_to_next_multiple_of,
        xLSTMLargeConfig,
    )

    external_xlstm = True
else:
    from typing import Literal

    BackendModeType = Literal["train", "train_with_padding", "inference"]
    ChunkwiseKernelType = Literal[
        "chunkwise--native_autograd",
        "parallel--native_autograd",
    ]
    DtypeType = Literal["float32", "bfloat16", "float16"]
    SequenceKernelType = Literal["native_sequence__native"]
    StepKernelType = Literal["native"]
    WeightModeType = Literal["single", "fused"]

    def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
        """Rounds up x to the next multiple of multiple_of."""
        return int(((x + multiple_of - 1) // multiple_of) * multiple_of)

    external_xlstm = False


@auto_docstring(checkpoint="NX-AI/xLSTM-7b")
@strict
class xLSTMConfig(PreTrainedConfig):
    r"""
    num_blocks (int, optional, *optional*, defaults to 32):
        Number of blocks of the xLSTM model, use num_hidden_layers if None.
    num_heads (int, optional, *optional*, defaults to 8):
        Number of heads for the xLSTM Layer/Cell.
    use_bias (bool, optional, *optional*, defaults to `False`):
        Whether to use biases in the xLSTM model.
    norm_reduction_force_float32 (bool, optional, *optional*, defaults to `True`):
        Whether to force the float32 norm reduction op to be done in fp32 precision.
    add_out_norm (bool, optional, *optional*, defaults to `True`):
        Whether to add an output norm after the blocks before the LMHead.
    qk_dim_factor (float, optional, *optional*, defaults to 0.5):
        Scale factor for the query and key dimension.
    v_dim_factor (float, optional, *optional*, defaults to 1.0):
        Scale factor for the value dimension.
    chunkwise_kernel (ChunkwiseKernelType, optional, *optional*, defaults to `"chunkwise--native_autograd"`):
        Kernel type for chunkwise processing mode.
    sequence_kernel (SequenceKernelType, optional, *optional*, defaults to `"native_sequence__native"`):
        Kernel type for sequence processing mode.
    step_kernel (StepKernelType, optional, *optional*, defaults to `"native"`):
        Kernel type for step processing mode.
    mode (BackendModeType, optional, *optional*, defaults to `"inference"`):
        Operation mode (inference is needed for generation).
    chunk_size (int, optional, *optional*, defaults to 64):
        Internal chunk size.
    return_last_states (bool, optional, *optional*, defaults to `True`):
        If to return the last states / cache internally. Needed as True for generation.
    autocast_kernel_dtype (DtypeType, optional, *optional*, defaults to `"bfloat16"`):
        Kernel dtype for the states.
    inference_state_dtype (DtypeType, optional, *optional*, defaults to `"float32"`):
        Kernel dtype for states in inference.
    ffn_proj_factor (float, optional, *optional*, defaults to 2.667):
        Size factor of the post-up projection gated Feed Forward network.
    ffn_round_up_to_multiple_of (int, optional, *optional*, defaults to 64):
        Size factor round value of the post-up projection gated Feed Forward network.
    gate_soft_cap (float, optional, *optional*, defaults to 15.0):
        Gate soft cap scale.
    output_logit_soft_cap (float, optional, *optional*, defaults to 30.0):
        Output logit soft cap scale.
    weight_mode (`Literal`, *optional*, defaults to `"single"`):
        Whether parallel linear layers are separated or fused (single).
    max_inference_chunksize (int, optional, *optional*, defaults to 16384):
        Limit the chunk size for inference to save memory.

    Example:

    ```python
    >>> from transformers import xLSTMConfig, xLSTMModel

    >>> # Initializing a xLSTM configuration
    >>> configuration = xLSTMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = xLSTMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlstm"

    vocab_size: int = 50304
    hidden_size: int = 4096
    embedding_dim: int | None = None
    num_hidden_layers: int = 32
    num_blocks: int | None = None
    num_heads: int = 8
    use_bias: bool = False
    norm_reduction_force_float32: bool = True
    tie_word_embeddings: bool = False
    add_out_norm: bool = True
    norm_eps: float = 1e-6
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd"
    sequence_kernel: SequenceKernelType = "native_sequence__native"
    step_kernel: StepKernelType = "native"
    mode: BackendModeType = "inference"
    chunk_size: int = 64
    return_last_states: bool = True
    autocast_kernel_dtype: DtypeType = "bfloat16"
    eps: float = 1e-6
    inference_state_dtype: DtypeType = "float32"
    ffn_proj_factor: float = 2.667
    ffn_round_up_to_multiple_of: int = 64
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    weight_mode: WeightModeType = "single"
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_inference_chunksize: int = 16384

    def __post_init__(self, **kwargs):
        self.hidden_size = self.hidden_size if self.hidden_size is not None else self.embedding_dim
        self.embedding_dim = self.embedding_dim if self.embedding_dim is not None else self.hidden_size
        self.num_hidden_layers = self.num_hidden_layers if self.num_hidden_layers is not None else self.num_blocks
        self.num_blocks = self.num_blocks if self.num_blocks is not None else self.num_hidden_layers
        super().__post_init__(**kwargs)

    @property
    def qk_dim(self):
        return round_up_to_next_multiple_of(
            self.hidden_size * self.qk_dim_factor,
            multiple_of=64,
        )

    @property
    def v_dim(self):
        return round_up_to_next_multiple_of(
            self.hidden_size * self.v_dim_factor,
            multiple_of=64,
        )

    @property
    def qk_head_dim(self):
        return self.qk_dim // self.num_heads

    @property
    def v_head_dim(self):
        return self.v_dim // self.num_heads

    def to_xlstm_block_config(self):
        if external_xlstm:
            return xLSTMLargeConfig(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                num_blocks=self.num_hidden_layers,
                num_heads=self.num_heads,
                use_bias=self.use_bias,
                add_out_norm=self.add_out_norm,
                norm_eps=self.norm_eps,
                norm_reduction_force_float32=self.norm_reduction_force_float32,
                # mlstm_layer
                qk_dim_factor=self.qk_dim_factor,
                v_dim_factor=self.v_dim_factor,
                # mlstm backend
                chunkwise_kernel=self.chunkwise_kernel,
                sequence_kernel=self.sequence_kernel,
                step_kernel=self.step_kernel,
                mode=self.mode,
                chunk_size=self.chunk_size,
                return_last_states=self.return_last_states,
                autocast_kernel_dtype=self.autocast_kernel_dtype,
                eps=self.eps,
                inference_state_dtype=self.inference_state_dtype,
                # feedforward
                ffn_proj_factor=self.ffn_proj_factor,
                ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
                # capping
                gate_soft_cap=self.gate_soft_cap,
                output_logit_soft_cap=self.output_logit_soft_cap,
                weight_mode=self.weight_mode,
            )
        else:
            return self


__all__ = ["xLSTMConfig"]
