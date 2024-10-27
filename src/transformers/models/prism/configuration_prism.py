# coding=utf-8
# Copyright 2024 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Prism model configuration"""

from collections import OrderedDict
from typing import Any, Mapping, Optional

from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging


logger = logging.get_logger(__name__)


class PrismConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `PrismModel`. It is used to instantiate a
    Prism model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Prism architecture as described in the paper.

    Args:
        vocab_size (`int`, *optional*, defaults to 65400):
            Vocabulary size of the Prism model.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
        encoder_layers (`int`, *optional*, defaults to 8):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 12288):
            Dimensionality of the feed-forward layer in the encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 8):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 12288):
            Dimensionality of the feed-forward layer in the decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is an encoder-decoder architecture.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the encoder and decoder.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and decoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        decoder_start_token_id (`int`, *optional*, defaults to 2):
            The token ID that indicates the start of the decoding sequence.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale the embeddings by the square root of the `d_model`.
        pad_token_id (`int`, *optional*, defaults to 1):
            The token ID used for padding sequences.
        bos_token_id (`int`, *optional*, defaults to 0):
            The token ID for the beginning of the sequence.
        eos_token_id (`int`, *optional*, defaults to 2):
            The token ID for the end of the sequence.

    Example:

    ```python
    >>> from transformers import PrismConfig, PrismModel

    >>> # Initializing a Prism facebook/prism style configuration
    >>> configuration = PrismConfig()

    >>> # Initializing a model (with random weights) from the facebook/prism style configuration
    >>> model = PrismModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "prism"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=65400,
        max_position_embeddings=1024,
        encoder_layers=8,
        encoder_ffn_dim=12288,
        encoder_attention_heads=20,
        decoder_layers=8,
        decoder_ffn_dim=12288,
        decoder_attention_heads=20,
        encoder_layerdrop=0.00,
        decoder_layerdrop=0.00,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=1280,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        scale_embedding=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


class PrismOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
            ]
        )

        if self.use_past:
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
        return common_inputs

    # Copied from BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering
    # A better name would be _generate_dummy_inputs_for_encoder_and_decoder because sequence classification and question
    # answering are not supported for Prism, but this name is preserved to be able to check that the copy matches what
    # was done for BART so that it can be updated if need be.
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # Copied from OnnxConfig.generate_dummy_inputs
        # Did not use super(OnnxConfigWithPast, self).generate_dummy_inputs for code clarity.
        # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # Generate dummy inputs according to compute batch and sequence
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs

    # Copied from transformers.models.bart.configuration_bart.BartOnnxConfig._generate_dummy_inputs_for_default_and_seq2seq_lm
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # Generate decoder inputs
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            decoder_past_length = decoder_seq_length + 3
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )

            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            common_inputs["past_key_values"] = []
            # If the number of encoder and decoder layers are present in the model configuration, both are considered
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            # TODO: test this.
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        return common_inputs

    generate_dummy_inputs = _generate_dummy_inputs_for_default_and_seq2seq_lm
