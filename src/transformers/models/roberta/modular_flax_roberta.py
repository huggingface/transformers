# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
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
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roberta import RobertaConfig
from ..bert.modeling_flax_bert import (
    FlaxBertEmbeddings,
    FlaxBertSelfAttention,
    FlaxBertSelfOutput,
    FlaxBertAttention,
    FlaxBertIntermediate,
    FlaxBertOutput,
    FlaxBertLayer,
    FlaxBertLayerCollection,
    FlaxBertEncoder,
    FlaxBertPooler,
    FlaxBertPreTrainedModel,
    FlaxBertModule,
    FlaxBertForMultipleChoiceModule,
    FlaxBertForTokenClassificationModule,
    FlaxBertForQuestionAnsweringModule,
)


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "FacebookAI/roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

remat = nn_partitioning.remat


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx).astype("i4")

    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    return incremental_indices.astype("i4") + padding_idx


ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = None # Will be inherited from mdeling_flax_bert.py


class FlaxRobertaEmbeddings(FlaxBertEmbeddings):
    pass


class FlaxRobertaSelfAttention(FlaxBertSelfAttention):
    pass


class FlaxRobertaSelfOutput(FlaxBertSelfOutput):
    pass

class FlaxRobertaAttention(FlaxBertAttention):
    config: RobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        super().setup()
        self.self = FlaxRobertaSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)


class FlaxRobertaIntermediate(FlaxBertIntermediate):
    pass


class FlaxRobertaOutput(FlaxBertOutput):
    pass


class FlaxRobertaLayer(FlaxBertLayer):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        super().setup()
        self.attention = FlaxRobertaAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)


class FlaxRobertaLayerCollection(FlaxBertLayerCollection):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        if self.gradient_checkpointing:
            FlaxRobertaCheckpointLayer = remat(FlaxRobertaLayer, static_argnums=(5, 6, 7))
            self.layers = [
                FlaxRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            self.layers = [
                FlaxRobertaLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

class FlaxRobertaEncoder(FlaxBertEncoder):

    def setup(self):
        super().setup()
        self.layer = FlaxRobertaLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

class FlaxRobertaPooler(FlaxBertPooler):
    pass

class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN["gelu"](hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class FlaxRobertaPreTrainedModel(FlaxBertPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.ones_like(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
            )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        if self.config.add_cross_attention:
            # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed
            # down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be
            # changed by FlaxRobertaAttention module
            if past_key_values:
                inputs["cache"] = past_key_values
                mutable = ["cache"]
            else:
                mutable = False

            outputs = self.module.apply(
                inputs,
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                token_type_ids=jnp.array(token_type_ids, dtype="i4"),
                position_ids=jnp.array(position_ids, dtype="i4"),
                head_mask=jnp.array(head_mask, dtype="i4"),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=not train,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                rngs=rngs,
                mutable=mutable,
            )

            # add updated cache to model output
            if past_key_values is not None and return_dict:
                outputs, past_key_values = outputs
                outputs["past_key_values"] = unfreeze(past_key_values["cache"])
                return outputs
            elif past_key_values is not None and not return_dict:
                outputs, past_key_values = outputs
                outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        else:
            outputs = self.module.apply(
                inputs,
                jnp.array(input_ids, dtype="i4"),
                jnp.array(attention_mask, dtype="i4"),
                token_type_ids=jnp.array(token_type_ids, dtype="i4"),
                position_ids=jnp.array(position_ids, dtype="i4"),
                head_mask=jnp.array(head_mask, dtype="i4"),
                deterministic=not train,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                rngs=rngs,
            )

        return outputs


class FlaxRobertaModule(FlaxBertModule):

    def setup(self):
        super().setup()
        self.embeddings = FlaxRobertaEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.pooler = FlaxRobertaPooler(self.config, dtype=self.dtype)


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaModule


append_call_sample_docstring(FlaxRobertaModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class FlaxRobertaForMaskedLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMaskedLMModule


append_call_sample_docstring(
    FlaxRobertaForMaskedLM,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)


class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.classifier = FlaxRobertaClassificationHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForSequenceClassificationModule


append_call_sample_docstring(
    FlaxRobertaForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRobertaForMultipleChoiceModule(FlaxBertForMultipleChoiceModule):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        super().setup()
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMultipleChoiceModule


overwrite_call_docstring(
    FlaxRobertaForMultipleChoice, ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxRobertaForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRobertaForTokenClassificationModule(FlaxBertForTokenClassificationModule):

    def setup(self):
        super().setup()
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )


@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForTokenClassificationModule


append_call_sample_docstring(
    FlaxRobertaForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRobertaForQuestionAnsweringModule(FlaxBertForQuestionAnsweringModule):

    def setup(self):
        super().setup()
        self.roberta = FlaxRobertaModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )


@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxRobertaForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxRobertaForCausalLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.roberta = FlaxRobertaModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForCausalLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyway.
        # Thus, we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(
    FlaxRobertaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
