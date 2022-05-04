# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
""" Jukebox configuration"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging


logger = logging.get_logger(__name__)

JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/jukebox-dummy": "https://huggingface.co/huggingface/jukebox-dummy/resolve/main/config.json",
}


class JukeboxConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxModel`] or a [`TFJukeboxModel`]. It is
    used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the GPT-2
    [small](https://huggingface.co/jukebox) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    The downsampling and stride are used to determine downsampling of the input sequence. For example, downsamoling =
    (5,3), and strides = (2, 2) will downsample the audio by 2**5 = 32 to get the first level of codes, and 2**8 = 256
    to get the second level codes. This is mostly true for training the top level prior and the upsamplers.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JukeboxModel`] or [`TFJukeboxModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`JukeboxDoubleHeadsModel`] and
            [`TFJukeboxDoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`JukeboxDoubleHeadsModel`] and
            [`TFJukeboxDoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`JukeboxDoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`JukeboxDoubleHeadsModel`] and
            [`TFJukeboxDoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`JukeboxDoubleHeadsModel`] and
            [`TFJukeboxDoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import JukeboxModel, JukeboxConfig

    >>> # Initializing a Jukebox configuration
    >>> configuration = JukeboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = JukeboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "jukebox"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        emb_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        # Global paranmeters
        sr=16000,
        sample_length=None,
        sample_length_in_seconds=10,
        y_bins=(120, 4111),
        use_nonrelative_specloss=True,
        copy_input=False,
        resid_dropout=0.0,
        # MLP parameters
        mlp_init_scale=0.02,
        # Attention layer parameters
        attn_dropout=0.0,
        attn_init_scale=1.0,
        # transformer parameters
        activation_function="gelu_new",
        sample_hop_length=30000,
        hop_length=256,
        multispec_loss_n_fft=(2048, 1024, 512),
        multispec_loss_hop_length=(240, 120, 50),
        multispec_loss_window_size=(1200, 600, 240),
        vq_vae_levels=3,
        vq_vae_downs_t=(3, 2, 2),
        vq_vae_strides_t=(2, 2, 2),
        vq_vae_emmbedding_width=2048,
        vq_vae_codebook_dimension=2048,
        vq_vae_width=64,
        vq_vae_depth=4,
        vq_vae_m_conv=1,
        vq_vae_dilation_growth_rate=3,
        vq_vae_dilation_cycle=1,
        vq_vae_multipliers=(2, 1, 1),
        vq_vae_lmu=0.99,  # for the ema?
        vq_vae_commit=0.02,
        spectral=0.0,
        multispectral=1.0,
        # vq_vae_loss_fn = 'l1',
        vq_vae_conv_block_depth=4,
        vq_vae_conv_block_width=64,
        vq_vae_reverse_decoder_dilation=1,
        # parameters always false/useless at inference
        nb_priors=3,
        spread=None,
        prime_spread=None,
        zero_out=False,
        res_scale=False,
        pos_init=False,
        cond_zero_out=False,
        # args for the priors, 3 priors
        n_ctx=8192,
        t_bins=128,
        # l_bins=512,
        downs_t=(3, 2, 2),
        strides_t=(2, 2, 2),
        single_enc_dec=False,
        labels=False,
        merged_decoder=[True, False, False],
        priors_width=[4096, 2048, 1024],
        l_bins=256,
        width=[4800, 1920, 128],
        depth=[79, 72, 72],
        n_heads=[8, 1, 1],
        use_tokens=[False, True, True],
        n_tokens=[512, 512, 512],
        attn_order=[10, 2, 2],
        blocks=128,
        c_res=1,
        init_scale=[0.7, 1, 1],
        prime_width=[1280, 128, 128],
        prime_depth=[18, 3, 3],
        cond_depth=[3, 16, 16],
        cond_width=[128, 1024, 1024],
        cond_dilation_growth_rate=[1, 3, 3],
        cond_dilation_cycle=[None, 8, 8],
        cond_c_res=[0, 1, 1],
        cond_res_scale=False,
        prime_cond_c_res=[0, 1, 1],
        prime_heads=4,
        prime_attn_order=2,
        prime_blocks=32,
        prime_init_scale=[0.1, 0.4, 0.4],
        prime_c_res=1,
        prime_loss_fraction=[0.4, 0.0, 0.0],
        min_duration=23.8,
        max_duration=600.0,
        fp16_params=True,
        alignment_layer=68,
        alignment_head=2,
        m_attn=0.25,
        n_vocab=80,
        cond_m_conv=1,
        max_bow_genre_size=5,  # this should only be in the tokenizer
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.emb_dropout = emb_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.max_bow_genre_size = max_bow_genre_size
        self.cond_m_conv = cond_m_conv
        self.n_vocab = n_vocab
        self.sr = sr
        self.sample_length = sample_length
        self.sample_length_in_seconds = sample_length_in_seconds
        self.y_bins = y_bins
        self.use_nonrelative_specloss = use_nonrelative_specloss
        self.copy_input = copy_input
        self.resid_dropout = resid_dropout
        self.mlp_init_scale = mlp_init_scale
        self.attn_dropout = attn_dropout
        self.attn_init_scale = attn_init_scale

        self.activation_function = activation_function
        self.sample_hop_length = sample_hop_length
        self.hop_length = hop_length
        self.multispec_loss_n_fft = multispec_loss_n_fft

        self.multispec_loss_hop_length = multispec_loss_hop_length

        self.multispec_loss_window_size = multispec_loss_window_size

        self.vq_vae_levels = vq_vae_levels
        self.vq_vae_downs_t = vq_vae_downs_t

        self.vq_vae_strides_t = vq_vae_strides_t

        self.vq_vae_emmbedding_width = vq_vae_emmbedding_width
        self.vq_vae_codebook_dimension = vq_vae_codebook_dimension
        self.vq_vae_width = vq_vae_width
        self.vq_vae_depth = vq_vae_depth
        self.vq_vae_m_conv = vq_vae_m_conv
        self.vq_vae_dilation_growth_rate = vq_vae_dilation_growth_rate
        self.vq_vae_dilation_cycle = vq_vae_dilation_cycle
        self.vq_vae_multipliers = vq_vae_multipliers

        self.vq_vae_lmu = vq_vae_lmu

        self.vq_vae_commit = vq_vae_commit
        self.spectral = spectral
        self.multispectral = multispectral

        self.vq_vae_conv_block_depth = vq_vae_conv_block_depth
        self.vq_vae_conv_block_width = vq_vae_conv_block_width
        self.vq_vae_reverse_decoder_dilation = vq_vae_reverse_decoder_dilation

        self.nb_priors = nb_priors
        self.spread = spread
        self.prime_spread = prime_spread
        self.zero_out = zero_out
        self.res_scale = res_scale
        self.pos_init = pos_init
        self.cond_zero_out = cond_zero_out
        self.n_ctx = n_ctx
        self.t_bins = t_bins
        self.l_bins = l_bins
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.single_enc_dec = single_enc_dec
        self.labels = labels
        self.merged_decoder = merged_decoder
        self.priors_width = priors_width
        self.width = width
        self.depth = depth
        self.n_heads = n_heads
        self.use_tokens = use_tokens
        self.n_tokens = n_tokens
        self.attn_order = attn_order
        self.blocks = blocks
        self.c_res = c_res
        self.init_scale = init_scale
        self.prime_width = prime_width
        self.prime_depth = prime_depth
        self.cond_depth = cond_depth
        self.cond_width = cond_width
        self.cond_dilation_growth_rate = cond_dilation_growth_rate
        self.cond_dilation_cycle = cond_dilation_cycle
        self.cond_c_res = cond_c_res
        self.cond_res_scale = cond_res_scale
        self.prime_cond_c_res = prime_cond_c_res
        self.prime_heads = prime_heads
        self.prime_attn_order = prime_attn_order
        self.prime_blocks = prime_blocks
        self.prime_init_scale = prime_init_scale
        self.prime_c_res = prime_c_res
        self.prime_loss_fraction = prime_loss_fraction
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.fp16_params = fp16_params
        self.alignment_layer = alignment_layer
        self.alignment_head = alignment_head
        self.m_attn = m_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class JukeboxOnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13
