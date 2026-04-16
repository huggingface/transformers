# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PhoneticXeus model configuration"""

import functools
import operator

from ...configuration_utils import PretrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="changelinglab/PhoneticXeus")
class PhoneticXeusConfig(PretrainedConfig):
    r"""
    vocab_size (`int`, *optional*, defaults to 428):
        Vocabulary size of the PhoneticXeus model (IPA phoneme inventory).
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the encoder layers.
    num_hidden_layers (`int`, *optional*, defaults to 19):
        Number of E-Branchformer encoder layers.
    num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the encoder.
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the feed-forward layers in the encoder.
    hidden_act (`str`, *optional*, defaults to `"swish"`):
        The non-linear activation function in the feed-forward layers.
    hidden_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for fully connected layers in the encoder.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    feat_proj_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for the feature projection layer.
    final_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the final projection layer of [`PhoneticXeusForCTC`].
    layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability during training.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation of the truncated normal initializer for weight initialization.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        Epsilon for layer normalization.
    normalize_audio (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the raw audio waveform before the CNN feature encoder.
    feat_extract_norm (`str`, *optional*, defaults to `"layer"`):
        The norm to be applied to 1D convolutional layers in the feature encoder. One of `"group"` for group
        normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
        convolutional layers.
    feat_extract_activation (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function in the 1D convolutional layers of the feature extractor.
    conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
        Number of input and output channels of each 1D convolutional layer in the feature encoder.
    conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
        Stride of each 1D convolutional layer in the feature encoder.
    conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`):
        Kernel size of each 1D convolutional layer in the feature encoder.
    conv_bias (`bool`, *optional*, defaults to `True`):
        Whether the 1D convolutional layers have a bias.
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Kernel size of the convolutional positional embeddings layer.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of the convolutional positional embeddings layer.
    conv_pos_weight_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply weight normalization to the convolutional positional embedding layer.
    cgmlp_linear_units (`int`, *optional*, defaults to 4096):
        Hidden dimensionality of the ConvolutionalGatingMLP in each E-Branchformer layer.
    cgmlp_conv_kernel (`int`, *optional*, defaults to 31):
        Kernel size of the depthwise convolution in the Convolutional Spatial Gating Unit (CSGU).
    use_linear_after_conv (`bool`, *optional*, defaults to `False`):
        Whether to apply a linear layer after the depthwise convolution in CSGU.
    gate_activation (`str`, *optional*, defaults to `"identity"`):
        Activation function for gating in CSGU.
    merge_conv_kernel (`int`, *optional*, defaults to 31):
        Kernel size of the depthwise convolution used to merge the two branches in each E-Branchformer layer.
    use_ffn (`bool`, *optional*, defaults to `True`):
        Whether to use feed-forward layers in each E-Branchformer layer.
    macaron_ffn (`bool`, *optional*, defaults to `True`):
        Whether to use macaron-style pre-branch feed-forward layer (half-step residual).
    interctc_layer_idx (`tuple[int]` or `list[int]`, *optional*, defaults to `(4, 8, 12)`):
        Layer indices (1-based) at which intermediate CTC self-conditioning is applied. At each specified layer,
        the encoder output is projected through the CTC head, softmaxed, and fed back via a conditioning layer.
    interctc_use_conditioning (`bool`, *optional*, defaults to `True`):
        Whether to enable intermediate CTC self-conditioning in the encoder.
    ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`.
    ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`.

    Example:

    ```python
    >>> from transformers import PhoneticXeusConfig, PhoneticXeusModel

    >>> configuration = PhoneticXeusConfig()
    >>> model = PhoneticXeusModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "phoneticxeus"

    vocab_size: int = 428
    hidden_size: int = 1024
    num_hidden_layers: int = 19
    num_attention_heads: int = 8
    intermediate_size: int = 4096

    hidden_act: str = "swish"
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    feat_proj_dropout: float | int = 0.0
    final_dropout: float | int = 0.1
    layerdrop: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5

    normalize_audio: bool = True
    feat_extract_norm: str = "layer"
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = True

    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    conv_pos_weight_norm: bool = True

    cgmlp_linear_units: int = 4096
    cgmlp_conv_kernel: int = 31
    use_linear_after_conv: bool = False
    gate_activation: str = "identity"
    merge_conv_kernel: int = 31
    use_ffn: bool = True
    macaron_ffn: bool = True

    interctc_layer_idx: list[int] | tuple[int, ...] = (4, 8, 12)
    interctc_use_conditioning: bool = True

    ctc_loss_reduction: str = "sum"
    ctc_zero_infinity: bool = True

    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        self.num_feat_extract_layers = len(self.conv_dim)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)


__all__ = ["PhoneticXeusConfig"]
