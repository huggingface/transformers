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

"""

Example of config from OmniASR-CTC-300M model

Wav2Vec2AsrConfig(
    encoder_config=Wav2Vec2EncoderConfig(
        model_dim=1024,
        max_seq_len=4096,
        feature_dim=512,
        use_fbank=False,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
        feature_extractor_bias=True,
        feature_extractor_layer_norm_convs=True,
        feature_grad_scale=0.1,
        num_fbank_channels=0,
        fbank_stride=0,
        sample_fbank_every_k=0,
        pos_encoder_type='conv',
        pos_encoder_depth=1,
        pos_conv_kernel_size=128,
        num_pos_conv_groups=16,
        use_conformer=False,
        num_encoder_layers=24,
        num_encoder_attn_heads=16,
        ffn_inner_dim=4096,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        ffn_inner_dropout_p=0.1,
        layer_drop_p=0.1,
        norm_order=<TransformerNormOrder.PRE: 1>,
        depthwise_conv_kernel_size=0
    ),
    target_vocab_size=10288,
    final_dropout_p=0.0,
    use_masking=False,
    temporal_mask_span_len=10,
    max_temporal_mask_prob=0.0,
    min_num_temporal_mask_spans=2,
    spatial_mask_span_len=64,
    max_spatial_mask_prob=0.0,
    min_num_spatial_mask_spans=
)


Example of 300m model config from OmniASR-LLM-300M v2 model

Wav2Vec2LlamaConfig(
    wav2vec2_asr_config=Wav2Vec2AsrConfig(
        encoder_config=Wav2Vec2EncoderConfig(
            model_dim=1024,
            max_seq_len=4096,
            feature_dim=512,
            use_fbank=False,
            first_pass_dropout_p=0.0,
            layer_norm_features=False,
            feature_extractor_layer_descs=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
            feature_extractor_bias=True,
            feature_extractor_layer_norm_convs=True,
            feature_grad_scale=0.1,
            num_fbank_channels=0,
            fbank_stride=0,
            sample_fbank_every_k=0,
            pos_encoder_type='conv',
            pos_encoder_depth=1,
            pos_conv_kernel_size=128,
            num_pos_conv_groups=16,
            use_conformer=False,
            num_encoder_layers=24,
            num_encoder_attn_heads=16,
            ffn_inner_dim=4096,
            dropout_p=0.0,
            attn_dropout_p=0.0,
            ffn_inner_dropout_p=0.1,
            layer_drop_p=0.1,
            norm_order=<TransformerNormOrder.PRE: 1>,
            depthwise_conv_kernel_size=0),
        target_vocab_size=9812,
        final_dropout_p=0.0,
        use_masking=False,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.0,
        min_num_temporal_mask_spans=2,
        spatial_mask_span_len=64,
        max_spatial_mask_prob=0.0,
        min_num_spatial_mask_spans=2),
    llama_config=LLaMAConfig(
        model_dim=4096,
        max_seq_len=8192,
        vocab_size=10288,
        pad_idx=1,
        tied_embeddings=False,
        num_layers=12,
        num_attn_heads=8,
        num_key_value_heads=8,
        ffn_inner_dim=4096,
        ffn_inner_dim_scale=0.6666666666666666,
        ffn_inner_dim_multiplier=1.0,
        ffn_inner_dim_multiple_of=256,
        rope_theta=10000.0,
        use_scaled_rope=False,
        rope_scale=LLaMARoPEScaleConfig(
            factor=8.0,
            frequency_factors=(1.0, 4.0),
            original_context_length=8192),
        dropout_p=0.1,
        init_std=None,
        init_std_scale='layer',
        shard_embed_dim=True),
    beam_search_config=Wav2Vec2LlamaBeamSearchConfig(
        nbest=5,
        length_norm=False,
        compression_window=100,
        compression_threshold=4.0),
    streaming_config=Wav2Vec2LlamaStreamingConfig(
        is_streaming=False,
        segment_secs=15.0,
        sample_rate=16000,
        n_context_segments=1,
        text_tokenizer='',
        min_audio_ms=25),
    encoder_stacking=1,
    frozen_encoder=1,
    lang_embeddings_p=0.5,
    language_column_name='lang',
    context_text_only=False,
    n_special_tokens=1,
    unk_idx=3,
    bos_idx=0,
    eos_idx=2,
    pad_idx=1,
    boh_idx=None,
    eoh_idx=None,
    model_type=<ModelType.LLM_ASR_LID: 2>,
    n_context_examples=0)


"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="bezzam/omniasr-ctc-300m-v2")
@strict
class OmniASREncoderConfig(PreTrainedConfig):
    r"""
    conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
        A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
        feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
    conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
        length of *conv_kernel* defines the number of convolutional layers and has to match the length of
        *conv_dim*.
    conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
        A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
        of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
    conv_bias (`bool`, *optional*, defaults to `True`):
        Whether the 1D convolutional layers have a bias.
    feat_extract_norm (`str`, *optional*, defaults to `"layer"`):
        The norm to be applied to 1D convolutional layers in the feature encoder. One of `"group"` for group
        normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
        convolutional layers.
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Number of convolutional positional embeddings. Defines the kernel size of the 1D convolutional positional
        embeddings layer.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of the 1D convolutional positional embeddings layer.
    hidden_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the encoder.
    layerdrop (`float`, *optional*, defaults to 0.1):
        The LayerDrop probability. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556) for more
        details.
    feat_proj_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for the output of the feature encoder.
    activation_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for activations inside the feed-forward layer.
    feat_extract_activation (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function in the 1D convolutional layers of the feature encoder.
    apply_spec_augment (`bool`, *optional*, defaults to `False`):
        Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder.
    mask_time_length (`int`, *optional*, defaults to 10):
        Length of vector span along the time axis.
    mask_time_prob (`float`, *optional*, defaults to 0.0):
        Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. Only relevant
        if `apply_spec_augment=True`.
    mask_time_min_masks (`int`, *optional*, defaults to 2):
        The minimum number of masks of length `mask_time_length` generated along the time axis.
    mask_feature_length (`int`, *optional*, defaults to 64):
        Length of vector span along the feature axis.
    mask_feature_prob (`float`, *optional*, defaults to 0.0):
        Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. Only
        relevant if `apply_spec_augment=True`.
    mask_feature_min_masks (`int`, *optional*, defaults to 2):
        The minimum number of masks of length `mask_feature_length` generated along the feature axis.

    Example:

    ```python
    >>> from transformers import OmniASREncoderConfig, OmniASRSpeechEncoder

    >>> # Initializing an OmniASR encoder configuration
    >>> configuration = OmniASREncoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRSpeechEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_encoder"

    hidden_size: int = 1024
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True
    feat_extract_norm: str = "layer"
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    intermediate_size: int = 4096
    attention_dropout: float | int = 0.0
    hidden_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    feat_proj_dropout: float | int = 0.0
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    feat_extract_activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    apply_spec_augment: bool = False
    mask_time_length: int = 10
    mask_time_prob: float | int = 0.0
    mask_time_min_masks: int = 2
    mask_feature_length: int = 64
    mask_feature_prob: float | int = 0.0
    mask_feature_min_masks: int = 2

    def __post_init__(self, **kwargs):
        self.num_feat_extract_layers = len(self.conv_dim)
        self.do_stable_layer_norm = False
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


@auto_docstring(checkpoint="bezzam/omniasr-ctc-300m-v2")
@strict
class OmniASRCTCConfig(PreTrainedConfig):
    r"""
    encoder_config (`Union[dict, OmniASREncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.
    unk_token_id (`int`, *optional*, defaults to 3):
        The id of the *unknown* token.
    ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
        instance of [`OmniASRForCTC`].
    ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`OmniASRForCTC`].

    Example:

    ```python
    >>> from transformers import OmniASRForCTC, OmniASRCTCConfig

    >>> # Initializing an OmniASR-CTC configuration
    >>> configuration = OmniASRCTCConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRForCTC(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_ctc"
    sub_configs = {"encoder_config": OmniASREncoderConfig}

    encoder_config: dict | PreTrainedConfig | None = None
    vocab_size: int = 10288
    unk_token_id: int = 3
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = False
    # TODO check token ids, took from Wav2Vec2
    bos_token_id: int | None = 0
    pad_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = OmniASREncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = OmniASREncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)

    @classmethod
    def from_encoder_config(cls, encoder_config: OmniASREncoderConfig, **kwargs):
        r"""
        Instantiate a [`OmniASRCTCConfig`] (or a derived class) from omniASR encoder model configuration.

        Returns:
            [`OmniASRCTCConfig`]: An instance of a configuration object
        """

        return cls(encoder_config=encoder_config.to_dict(), **kwargs)

    @property
    def hidden_size(self):
        return self.encoder_config.hidden_size


@auto_docstring(checkpoint="bezzam/omniasr-llm-300m-v2")
@strict
class OmniASRLLMConfig(PreTrainedConfig):
    r"""
    encoder_stacking (`int`, *optional*, defaults to 1):
        Number of consecutive encoder frames stacked together before being projected to the text decoder. Used by
        the Zero-Shot variant, see
        https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/model.py#L1024
    num_language_embeddings (`int`, *optional*, defaults to 1694):
        Number of language embeddings in the language embedding table.
    unk_token_id (`int`, *optional*, defaults to 3):
        The id of the *unknown* token.
    num_special_tokens (`int`, *optional*, defaults to 1):
        Number of special tokens in the vocabulary.
    language_embedding_probability (`float`, *optional*, defaults to 0.5):
        Probability of using the language embedding during training.
    language_token_id (`int`, *optional*, defaults to 9218):
        The id of the language token.

    Example:

    ```python
    >>> from transformers import OmniASRForConditionalGeneration, OmniASRLLMConfig

    >>> # Initializing an OmniASR-LLM configuration
    >>> configuration = OmniASRLLMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OmniASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "omniasr_llm"
    sub_configs = {"audio_config": OmniASREncoderConfig, "text_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 10288,
        "hidden_size": 4096,
        "num_hidden_layers": 12,
        "num_key_value_heads": 8,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-05,
        "intermediate_size": 2816,
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    encoder_stacking: int = 1
    num_language_embeddings: int = 1694
    unk_token_id: int = 3
    num_special_tokens: int = 1
    language_embedding_probability: float | int = 0.5
    language_token_id: int = 9218
    bos_token_id: int | None = 0
    pad_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config = OmniASREncoderConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = OmniASREncoderConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)

        self.initializer_range = self.audio_config.initializer_range
        super().__post_init__(**kwargs)


__all__ = ["OmniASRCTCConfig", "OmniASREncoderConfig", "OmniASRLLMConfig"]
