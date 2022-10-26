# coding=utf-8
# Copyright 2020, The SwitchTransformers Authors and HuggingFace Inc.
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
""" Switch Transformers model configuration"""
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging


logger = logging.get_logger(__name__)

SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ybelkada/switch_transformers-base": (
        "https://huggingface.co/ybelkada/switch_transformers-base/resolve/main/config.json"
    ),
}


class SwitchTransformersConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SwitchTransformersModel`]. It is used to
    instantiate a SwitchTransformers model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    SwitchTransformers [ybelkada/switch_transformers-base](https://huggingface.co/ybelkada/switch_transformers-base)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the SwitchTransformers model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`SwitchTransformersModel`] or
            [`FlaxSwitchTransformersModel`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
        expert_capacity (`int`, *optional*, defaults to 1):
            Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
            Transformer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of dense hidden layers in the Transformer encoder layer.
        num_sparse_encoder_layers (`int`, *optional*, defaults to 6):
            Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.
        num_decoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_sparse_decoder_layers (`int`, *optional*, defaults to 12):
            Number of sparse (MoE) dense hidden layers in the Transformer decoder layer.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_experts (`int`, *optional*, defaults to 8):
            Number of experts for each SwitchTransformer layer.
        router_type (`str`, *optional*, defaults to `tokens_masked`):
            Router type - choice between `tokens_masked` and `tokens_scatter`, `experts_masked`.
        router_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the router.
        router_jitter_noise (`float`, *optional*, defaults to 0.1):
            Amount of noise to add to the router.
        router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
            Whether to ignore padding tokens when routing.
        router_dtype (`str`, *optional*, default to `float32`):
            The `dtype` used for the routers. It is preferable to keep the `dtype` to `float32` as specified in the
            "selective precision" discussion in https://arxiv.org/abs/2101.03961.
        batch_prioritized_routing (`bool`, *optional*, defaults to `False`):
            Whether to use batch prioritized routing.
        add_router_probs (`bool`, *optional*, defaults to `False`):
            Whether to output router probabilities to compute router auxiliary loss.
        num_selected_experts (`int`, *optional*, defaults to 2):
            Number of experts to select for each token.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. SwitchTransformersv1.1
            uses the `"gated-gelu"` feed forward projection. Original SwitchTransformers uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = "switch_transformers"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_sparse_encoder_layers=3,
        num_decoder_layers=12,
        num_sparse_decoder_layers=3,
        num_heads=12,
        num_experts=64,
        expert_capacity=1,
        router_type="tokens_masked",
        router_bias=False,
        router_jitter_noise=0.01,
        router_dtype="float32",
        num_selected_experts=2,
        router_ignore_padding_tokens=False,
        batch_prioritized_routing=False,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        add_router_probs=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff

        self.num_sparse_encoder_layers = num_sparse_encoder_layers

        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_sparse_decoder_layers = num_sparse_decoder_layers

        # This tells us, each how many encoder layer we'll have to set a sparse layer.
        if self.num_sparse_encoder_layers > 0:
            self.encoder_sparse_step = self.num_layers // self.num_sparse_encoder_layers
        else:
            self.encoder_sparse_step = self.num_layers  # HACK: this will create 0 sparse layers

        # This tells us, each how many encoder layer we'll have to set a sparse layer.
        if self.num_sparse_decoder_layers > 0:
            self.decoder_sparse_step = self.num_decoder_layers // self.num_sparse_decoder_layers
        else:
            self.decoder_sparse_step = self.num_decoder_layers  # HACK: this will create 0 sparse layers

        self.num_heads = num_heads
        self.router_type = router_type
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        self.router_dtype = router_dtype

        if router_dtype not in ["float16", "float32", "bfloat16"]:
            raise ValueError("""Please select a correct `router_dtype` from ["float16", "float32", "bfloat16"].""")

        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.batch_prioritized_routing = batch_prioritized_routing

        self.num_selected_experts = num_selected_experts

        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.add_router_probs = add_router_probs

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class SwitchTransformersOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13
