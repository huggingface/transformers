# coding=utf-8
# Copyright 2022 The Metaseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" OPT model configuration"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging


logger = logging.get_logger(__name__)

OPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
}


class OPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OPTModel`]. It is used to instantiate a OPT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OPT
    [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50272):
            Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OPTModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        do_layer_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to perform layer normalization before the attention block.
        word_embed_proj_dim (`int`, *optional*):
            `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
            `hidden_size`.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more
            details.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import OPTModel, OPTConfig

    >>> # Initializing a OPT facebook/opt-large style configuration
    >>> configuration = OPTConfig()

    >>> # Initializing a model from the facebook/opt-large style configuration
    >>> model = OPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before

        # Note that the only purpose of `_remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = _remove_final_layer_norm


class OPTOnnxConfig(OnnxConfigWithPast):
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
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13
