# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
from configuration_sentence_transform import SentenceTransformerCLIPConfig
from flax.core.frozen_dict import FrozenDict
from transformers import FLAX_MODEL_MAPPING
from transformers.file_utils import ModelOutput
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)


class FlaxSentenceCLIPModule(nn.Module):
    config: SentenceTransformerCLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        text_config = self.config.text_config
        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        text_module = FLAX_MODEL_MAPPING[self.config.text_config.__class__].module_class

        self.text_model = text_module(text_config, dtype=self.dtype)

        self.text_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02, dtype=self.dtype),
            use_bias=False,
        )
        self.logit_scale = self.param("logit_scale", jax.nn.initializers.ones, [])

    def __call__(
        self,
        input1_ids=None,
        input2_ids=None,
        attention1_mask=None,
        attention2_mask=None,
        position1_ids=None,
        position2_ids=None,
        token_type1_ids=None,
        token_type2_ids=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        text1_outputs = self.text_model(
            input_ids=input1_ids,
            attention_mask=attention1_mask,
            token_type_ids=token_type1_ids,
            position_ids=position1_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text2_outputs = self.text_model(
            input_ids=input2_ids,
            attention_mask=attention2_mask,
            token_type_ids=token_type2_ids,
            position_ids=position2_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text1_embeds = text1_outputs[1]
        text1_embeds = self.text_projection(text1_embeds)

        text2_embeds = text2_outputs[1]
        text2_embeds = self.text_projection(text2_embeds)

        # normalized features
        text1_embeds = text1_embeds / jnp.linalg.norm(text1_embeds, axis=-1, keepdims=True)
        text2_embeds = text2_embeds / jnp.linalg.norm(text2_embeds, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text1 = jnp.matmul(text1_embeds, text2_embeds.T) * logit_scale
        logits_per_text2 = logits_per_text1.T

        if not return_dict:
            return (logits_per_text2, logits_per_text1, text1_embeds, text1_embeds, text1_outputs, text2_outputs)

        return SentenceEncoderOutput(
            logits_per_text1=logits_per_text1,
            logits_per_text2=logits_per_text2,
            text1_embeds=text1_embeds,
            text2_embeds=text2_embeds,
            text1_model_output=text1_outputs,
            text2_model_output=text2_outputs,
        )


@flax.struct.dataclass
class SentenceEncoderOutput(ModelOutput):
    logits_per_text1: jax_xla.DeviceArray = None
    logits_per_text2: jax_xla.DeviceArray = None
    text1_embeds: jax_xla.DeviceArray = None
    text2_embeds: jax_xla.DeviceArray = None
    text1_model_output: FlaxBaseModelOutputWithPooling = None
    text2_model_output: FlaxBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class FlaxSentenceEncoderCLIPModel(FlaxPreTrainedModel):
    config_class = SentenceTransformerCLIPConfig
    module_class = FlaxSentenceCLIPModule

    def __init__(
        self,
        config: SentenceTransformerCLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        if input_shape is None:
            input_shape = ((1, 1),)

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensor
        input1_ids = jnp.zeros(input_shape[0], dtype="i4")
        input2_ids = jnp.zeros(input_shape[1], dtype="i4")
        position1_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input1_ids).shape[-1]), input_shape[0])
        position2_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input2_ids).shape[-1]), input_shape[0])
        token_type1_ids = jnp.ones_like(input1_ids)
        token_type2_ids = jnp.ones_like(input2_ids)
        attention1_mask = jnp.ones_like(input1_ids)
        attention2_mask = jnp.ones_like(input2_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input1_ids, input2_ids, attention1_mask, attention2_mask,
                                position1_ids, position2_ids, token_type1_ids, token_type2_ids)["params"]

    def __call__(
        self,
        input1_ids,
        input2_ids,
        attention1_mask=None,
        attention2_mask=None,
        position1_ids=None,
        position2_ids=None,
        token_type1_ids=None,
        token_type2_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if position1_ids is None:
            position1_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input1_ids).shape[-1]), input1_ids.shape)

        if position2_ids is None:
            position2_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input2_ids).shape[-1]), input2_ids.shape)

        if token_type1_ids is None:
            token_type1_ids = jnp.zeros_like(input1_ids)

        if token_type2_ids is None:
            token_type2_ids = jnp.zeros_like(input2_ids)

        if attention1_mask is None:
            attention1_mask = jnp.ones_like(input1_ids)

        if attention2_mask is None:
            attention2_mask = jnp.ones_like(input2_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input1_ids, dtype="i4"),
            jnp.array(input2_ids, dtype="i4"),
            jnp.array(attention1_mask, dtype="i4"),
            jnp.array(attention2_mask, dtype="i4"),
            jnp.array(position1_ids, dtype="i4"),
            jnp.array(position2_ids, dtype="i4"),
            jnp.array(token_type1_ids, dtype="i4"),
            jnp.array(token_type2_ids, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):
        r"""
        Args:
            input_ids (:obj:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.PreTrainedTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__

        Returns:
            text_features (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, output_dim`): The text embeddings
            obtained by applying the projection layer to the pooled output of text model.
        """
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, input_ids, attention_mask, position_ids, token_type_ids, deterministic):
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                deterministic=deterministic,
            )
            pooled_output = text_outputs[1]
            text_features = module.text_projection(pooled_output)
            return text_features

        return self.module.apply(
            {"params": self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    @classmethod
    def from_text_pretrained(
        cls,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxPreTrainedModel:
        """
        Params:
            text_model_name_or_path (:obj: `str`, `optional`):
                Information necessary to initiate the text model. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.FlaxPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch checkpoint folder` (e.g, ``./pt_model``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch checkpoint in
                      a Flax model using the provided conversion scripts and loading the Flax model afterwards.

            vision_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the vision model. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.FlaxPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch checkpoint folder` (e.g, ``./pt_model``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch checkpoint in
                      a Flax model using the provided conversion scripts and loading the Flax model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.

            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`).

                - To update the text configuration, use the prefix `text_` for each configuration parameter.
                - To update the vision configuration, use the prefix `vision_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import FlaxHybridCLIP
            >>> # initialize a model from pretrained BERT and CLIP models. Note that the projection layers will be randomly initialized.
            >>> # If using CLIP's vision model the vision projection layer will be initialized using pre-trained weights
            >>> model = FlaxHybridCLIP.from_text_vision_pretrained('bert-base-uncased', 'openai/clip-vit-base-patch32')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert-clip")
            >>> # load fine-tuned model
            >>> model = FlaxHybridCLIP.from_pretrained("./bert-clip")
        """

        kwargs_text = {
            argument[len("text_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the text and vision model
        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            assert (
                text_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
            from transformers import FlaxAutoModel

            if "config" not in kwargs_text:
                from transformers import AutoConfig

                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = FlaxAutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = SentenceTransformerCLIPConfig.from_text_configs(text_model.config, **kwargs)

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)

        model.params["text_model"] = text_model.params

        return model
