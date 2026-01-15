# coding=utf-8
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
    min_num_spatial_mask_spans=2
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


from typing import Union

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class OmniASREncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASREncoderConfig`]. It is used to instantiate
    a `OmniASREncoder` model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    TODO
    - docstrings for each parameter
    - rename to Transformer convention

    """

    model_type = "omniasr_encoder"

    def __init__(
        self,
        max_seq_len=4096, 
        feature_dim=512, 
        use_fbank=False, 
        layer_norm_features=False, 
        feature_grad_scale=0.1, 
        num_fbank_channels=0, 
        fbank_stride=0, 
        sample_fbank_every_k=0, 
        pos_encoder_depth=1, 
        use_conformer=False, 
        depthwise_conv_kernel_size=0,
        # NOTE: adapted to Transformer convention
        hidden_size=1024, 
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        layer_norm_pre=True,
        feat_extract_norm="layer", 
        num_attention_heads=16,
        num_hidden_layers=24,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        intermediate_size=4096,
        position_embeddings_type="conv",

        first_pass_dropout_p=0.0, # TODO not used
        attention_dropout=0.0,
        hidden_dropout=0.1,
        layerdrop=0.1, 
        final_dropout=0.0, 
        feat_proj_dropout=0.0,
        activation_dropout=0.1,

        # NOTE: added to be compatible with Wav2Vec2 modeling
        initializer_range=0.02,
        feat_extract_activation="gelu",
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        add_adapter=False,
        use_intermediate_ffn_before_adapter=False,  # TODO remove?
        # TODO keep spec agument params?
        apply_spec_augment=False, 
        mask_time_length=10, 
        mask_time_prob=0.0, 
        mask_time_min_masks=2, 
        mask_feature_length=64, 
        mask_feature_prob=0.0, 
        mask_feature_min_masks=2,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.use_fbank = use_fbank
        self.first_pass_dropout_p = first_pass_dropout_p
        self.layer_norm_features = layer_norm_features
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.feat_extract_norm = feat_extract_norm
        self.feature_grad_scale = feature_grad_scale
        self.num_fbank_channels = num_fbank_channels
        self.fbank_stride = fbank_stride
        self.sample_fbank_every_k = sample_fbank_every_k
        self.position_embeddings_type = position_embeddings_type
        self.pos_encoder_depth = pos_encoder_depth
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.use_conformer = use_conformer
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.final_dropout = final_dropout
        self.hidden_dropout = hidden_dropout
        self.layerdrop = layerdrop
        # Whether layer normalization is applied at the beginning of each layer or after each layer's residuation connection: https://github.com/facebookresearch/fairseq2/blob/a510a839e007d2b036185b7b4ca76074d287c67e/src/fairseq2/models/transformer/norm_order.py#L12
        self.layer_norm_pre = layer_norm_pre
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        
        self.layer_norm_eps = layer_norm_eps
        self.feat_proj_dropout = feat_proj_dropout
        self.activation_dropout = activation_dropout
        self.feat_extract_activation = feat_extract_activation
        self.hidden_act = hidden_act
        self.add_adapter=add_adapter
        if use_intermediate_ffn_before_adapter and not add_adapter:
            raise ValueError("`use_intermediate_ffn_before_adapter` is `True` but `add_adapter` is `False`.")
        self.use_intermediate_ffn_before_adapter = use_intermediate_ffn_before_adapter
        
        self.initializer_range = initializer_range

        # SpecAugment parameters
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_length = mask_time_length
        self.mask_time_prob = mask_time_prob
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_length = mask_feature_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_min_masks = mask_feature_min_masks

        super().__init__(**kwargs)

    @property
    def num_feat_extract_layers(self):
        return len(self.conv_dim)


class OmniASRCTCConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASRForCTC`]. It is used to instantiate a
    OmniASR-CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """

    model_type = "omniasr_ctc"
    sub_configs = {"encoder_config": OmniASREncoderConfig}

    def __init__(
        self,
        encoder_config=None,
        vocab_size=10288,
        # TODO check token ids, took from Wav2Vec2
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        **kwargs,
    ):
        
        if isinstance(encoder_config, dict):
            encoder_config = OmniASREncoderConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = OmniASREncoderConfig()
        self.encoder_config = encoder_config

        self.vocab_size = vocab_size
        self.initializer_range = self.encoder_config.initializer_range
        self.unk_token_id = unk_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

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


class OmniASRLLMConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASRForConditionalGeneration`]. It is used to
    instantiate an OmniASRForConditionalGeneration model according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

        
    TODO
    - docstrings for each parameter

    TODO encoder_stacking used by Zero-Shot variant
    https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/model.py#L1024
    """

    model_type = "omniasr_llm"
    sub_configs = {"encoder_config": OmniASREncoderConfig, "text_config": AutoConfig}

    # TODO change to omniasr vals
    # from other repo: https://github.com/harikc456/wav2vec2_llama_hf/blob/6153f04a7d3357d49601323fc1f7f4364bce6735/convert_to_hf.py#L237
    # TODO default to 7bv2?
    """
    LLaMAConfig(model_dim=4096, max_seq_len=8192, vocab_size=10288, pad_idx=1, tied_embeddings=False, num_layers=12,            
        num_attn_heads=8, num_key_value_heads=8, ffn_inner_dim=4096, ffn_inner_dim_scale=0.6666666666666666,                        
        ffn_inner_dim_multiplier=1.0, ffn_inner_dim_multiple_of=256, rope_theta=10000.0, use_scaled_rope=False,                     
        rope_scale=LLaMARoPEScaleConfig(factor=8.0, frequency_factors=(1.0, 4.0), original_context_length=8192), dropout_p=0.1,     
        init_std=None, init_std_scale='layer', shard_embed_dim=True)  
    """
    _default_text_config_kwargs = {
        "vocab_size": 10288,
        "hidden_size": 4096,
        "num_hidden_layers": 12,
        "num_key_value_heads": 8,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-05,
        
        # "max_position_embeddings": 8192,
        "intermediate_size": 2816,
    }

    def __init__(
        self,
        encoder_config=None,
        text_config=None,
        encoder_stacking=1,
        num_lang_embeddings=1694,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        num_special_tokens=1,
        **kwargs,
    ):
        
        if isinstance(encoder_config, dict):
            encoder_config = OmniASREncoderConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = OmniASREncoderConfig()
        self.encoder_config = encoder_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)
        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.initializer_range = self.encoder_config.initializer_range
        self.unk_token_id = unk_token_id
        self.encoder_stacking = encoder_stacking
        self.num_lang_embeddings = num_lang_embeddings
        self.num_special_tokens = num_special_tokens

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )



__all__ = ["OmniASRCTCConfig", "OmniASRLLMConfig", "OmniASREncoderConfig"]