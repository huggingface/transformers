# Copyright 2021 The Fairseq Authors, Microsoft Research, and The HuggingFace Inc. team. All rights reserved.
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
"""WavLM model configuration"""

import functools
import operator
from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class WavLMConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WavLMModel`]. It is used to instantiate an WavLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the WavLM
    [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32):
            Vocabulary size of the WavLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`WavLMModel`]. Vocabulary size of the model. Defines the different tokens
            that can be represented by the *inputs_ids* passed to the forward method of [`WavLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        final_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the final projection layer of [`WavLMForCTC`].
        layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more
            details.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        feat_extract_norm (`str`, *optional*, defaults to `"group"`):
            The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
            normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the feature encoder.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
            of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
        conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
            length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        do_stable_layer_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is
            True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is
            False` corresponds to applying layer norm after the attention layer.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://huggingface.co/papers/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Probability of each feature vector along the time axis to be chosen as the start of the vector span to be
            masked. Approximately `mask_time_prob * sequence_length // mask_time_length` feature vectors will be masked
            along the time axis. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Probability of each feature vector along the feature axis to be chosen as the start of the vector span to
            be masked. Approximately `mask_time_prob * hidden_size // mask_time_length` feature vectors will be masked
            along the time axis. This is only relevant if `apply_spec_augment is True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        num_codevectors_per_group (`int`, *optional*, defaults to 320):
            Number of entries in each quantization codebook (group).
        num_codevector_groups (`int`, *optional*, defaults to 2):
            Number of codevector groups for product codevector quantization.
        contrastive_logits_temperature (`float`, *optional*, defaults to 0.1):
            The temperature *kappa* in the contrastive loss.
        num_negatives (`int`, *optional*, defaults to 100):
            Number of negative samples for the contrastive loss.
        codevector_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the quantized feature vectors.
        proj_codevector_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the final projection of both the quantized and the transformer features.
        diversity_loss_weight (`int`, *optional*, defaults to 0.1):
            The weight of the codebook diversity loss component.
        ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`WavLMForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`WavLMForCTC`].
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`WavLMForSequenceClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification.
        tdnn_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 1500)`):
            A tuple of integers defining the number of output channels of each 1D convolutional layer in the *TDNN*
            module of the *XVector* model. The length of *tdnn_dim* defines the number of *TDNN* layers.
        tdnn_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 3, 3, 1, 1)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the *TDNN* module of the
            *XVector* model. The length of *tdnn_kernel* has to match the length of *tdnn_dim*.
        tdnn_dilation (`tuple[int]` or `list[int]`, *optional*, defaults to `(1, 2, 3, 1, 1)`):
            A tuple of integers defining the dilation factor of each 1D convolutional layer in *TDNN* module of the
            *XVector* model. The length of *tdnn_dilation* has to match the length of *tdnn_dim*.
        xvector_output_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the *XVector* embedding vectors.
        add_adapter (`bool`, *optional*, defaults to `False`):
            Whether a convolutional network should be stacked on top of the Wav2Vec2 Encoder. Can be very useful for
            warm-starting Wav2Vec2 for SpeechEncoderDecoder models.
        adapter_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adapter_stride (`int`, *optional*, defaults to 2):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        num_adapter_layers (`int`, *optional*, defaults to 3):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        output_hidden_size (`int`, *optional*):
            Dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
            if `add_adapter is True`.

    Example:

    ```python

    ```

    Example:

    ```python
    >>> from transformers import WavLMConfig, WavLMModel

    >>> # Initializing a WavLM facebook/wavlm-base-960h style configuration
    >>> configuration = WavLMConfig()

    >>> # Initializing a model (with random weights) from the facebook/wavlm-base-960h style configuration
    >>> model = WavLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "wavlm"

    vocab_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    feat_proj_dropout: float | int = 0.0
    final_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    feat_extract_norm: str = "group"
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    num_buckets: int = 320
    max_bucket_distance: int = 800
    do_stable_layer_norm: bool = False
    apply_spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    num_codevectors_per_group: int = 320
    num_codevector_groups: int = 2
    contrastive_logits_temperature: float = 0.1
    num_negatives: int = 100
    codevector_dim: int = 256
    proj_codevector_dim: int = 256
    diversity_loss_weight: float = 0.1
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = False
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    tdnn_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 1500)
    tdnn_kernel: list[int] | tuple[int, ...] = (5, 3, 3, 1, 1)
    tdnn_dilation: list[int] | tuple[int, ...] = (1, 2, 3, 1, 1)
    xvector_output_dim: int = 512
    num_ctc_classes: int = 80
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2
    add_adapter: bool = False
    adapter_kernel_size: int = 3
    adapter_stride: int = 2
    num_adapter_layers: int = 3
    output_hidden_size: int | None = None

    def __post_init__(self, **kwargs):
        self.num_feat_extract_layers = len(self.conv_dim)
        self.output_hidden_size = self.output_hidden_size or self.hidden_size
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
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


__all__ = ["WavLMConfig"]
