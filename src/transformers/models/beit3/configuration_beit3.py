# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.from transformers import PretrainedConfig
from transformers import PretrainedConfig


BEIT3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/beit-base-patch16-224-pt22k": (
        "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/resolve/main/config.json"
    ),
    # See all BEiT models at https://huggingface.co/models?filter=beit
}


class Beit3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Beit3Model`]. It is used to instantiate an BEiT3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BEiT3
    [microsoft/beit3-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit3-base-patch16-224-pt22k)
    architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 64010):
            Vocabulary size of the BEiT3 model. Defines the number of different image tokens that can be used during
            pre-training.
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
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to use BERT-style absolute position embeddings.
        use_relative_position_bias (`bool`, *optional*, defaults to `False`):
            Whether to use T5-style relative position embeddings in the self-attention layers.
        use_shared_relative_position_bias (`bool`, *optional*, defaults to `False`):
            Whether to use the same relative position embeddings across all self-attention layers of the Transformer.
        layer_scale_init_value (`float`, *optional*, defaults to 0.1):
            Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
            CLS token, before applying the classification head.
        out_indices (`List[int]`, *optional*, defaults to `[3, 5, 7, 11]`):
            Indices of the feature maps to use for semantic segmentation.
        pool_scales (`Tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):
            Pooling scales used in Pooling Pyramid Module applied on the last feature map.
        use_auxiliary_head (`bool`, *optional*, defaults to `True`):
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
            Weight of the cross-entropy loss of the auxiliary head.
        auxiliary_channels (`int`, *optional*, defaults to 256):
            Number of channels to use in the auxiliary head.
        auxiliary_num_convs (`int`, *optional*, defaults to 1):
            Number of convolutional layers to use in the auxiliary head.
        auxiliary_concat_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the output of the auxiliary head with the input before the classification layer.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example:

    ```python
    >>> from transformers import BeitConfig, BeitModel

    >>> # Initializing a BEiT3 beit3-base-patch16-224-pt22k style configuration
    >>> configuration = Beit3Config()

    >>> # Initializing a model (with random weights) from the beit3-base-patch16-224-pt22k style configuration
    >>> model = Beit3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "beit3"

    def __init__(
        self,
        embed_dim=768,
        num_attention_heads=12,
        hidden_size=3072,
        layers=12,
        encoder_normalize_before=False,
        normalize_before=False,
        activation_fn="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        subln=True,
        max_source_positions=1024,
        layernorm_eps=1e-5,
        vocab_size=64010,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_labels=2,
        initializer_range=0.02,
        label_smoothing=0.1,
        logit_scale_init_value=2.65926,
        layer_norm_eps=1e-05,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layers = layers
        self.normalize_before = normalize_before
        self.encoder_normalize_before = encoder_normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.subln = subln
        self.max_source_positions = max_source_positions
        self.layernorm_eps = layernorm_eps
        self.initializer_range = initializer_range
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.logit_scale_init_value = logit_scale_init_value
        self.layer_norm_eps = layer_norm_eps
        if self.subln:
            self.normalize_before = True
            self.deepnorm = False
