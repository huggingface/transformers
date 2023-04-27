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

import flax.linen as nn
import jax
import jax.numpy as jnp
from configuration_hybrid_clip import HybridCLIPConfig
from flax.core.frozen_dict import FrozenDict

from transformers import FLAX_MODEL_MAPPING, FlaxCLIPVisionModel
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.models.clip.modeling_flax_clip import FlaxCLIPOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class FlaxHybridCLIPModule(nn.Module):
    config: HybridCLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        text_config = self.config.text_config
        vision_config = self.config.vision_config

        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        text_module = FLAX_MODEL_MAPPING[self.config.text_config.__class__].module_class
        vision_module = FLAX_MODEL_MAPPING.get(self.config.vision_config.__class__, FlaxCLIPVisionModel).module_class

        self.text_model = text_module(text_config, dtype=self.dtype)
        self.vision_model = vision_module(vision_config, dtype=self.dtype)

        self.visual_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.text_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.logit_scale = self.param("logit_scale", jax.nn.initializers.ones, [])

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)

        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class FlaxHybridCLIP(FlaxPreTrainedModel):
    config_class = HybridCLIPConfig
    module_class = FlaxHybridCLIPModule

    def __init__(
        self,
        config: HybridCLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = ((1, 1), (1, config.vision_config.image_size, config.vision_config.image_size, 3))

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        input_ids = jnp.zeros(input_shape[0], dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        token_type_ids = jnp.ones_like(input_ids)
        attention_mask = jnp.ones_like(input_ids)

        pixel_values = jax.random.normal(rng, input_shape[1])

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids, token_type_ids)["params"]

    def __call__(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
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

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(pixel_values, dtype=jnp.float32),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
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
        params: dict = None,
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
            text_features (:obj:`jnp.ndarray` of shape :obj:`(batch_size, output_dim`): The text embeddings
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
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    def get_image_features(
        self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train=False
    ):
        r"""
        Args:
            pixel_values (:obj:`numpy.ndarray` of shape :obj:`(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using :class:`~transformers.ImageFeatureExtractionMixin`. See
                :meth:`transformers.ImageFeatureExtractionMixin.__call__` for details.

        Returns:
            image_features (:obj:`jnp.ndarray` of shape :obj:`(batch_size, output_dim`): The image embeddings
            obtained by applying the projection layer to the pooled output of vision model.
        """

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            pooled_output = vision_outputs[1]  # pooled_output
            image_features = module.visual_projection(pooled_output)
            return image_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    @classmethod
    def from_text_vision_pretrained(
        cls,
        text_model_name_or_path: str = None,
        vision_model_name_or_path: str = None,
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

        kwargs_vision = {
            argument[len("vision_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_")
        }

        # remove text, vision kwargs from kwargs
        for key in kwargs_text.keys():
            del kwargs["text_" + key]
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]

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

        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            assert (
                vision_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
            from transformers import FlaxAutoModel

            if "config" not in kwargs_vision:
                from transformers import AutoConfig

                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)
                kwargs_vision["config"] = vision_config

            vision_model = FlaxAutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = HybridCLIPConfig.from_text_vision_configs(text_model.config, vision_model.config, **kwargs)

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)

        if vision_config.model_type == "clip":
            model.params["vision_model"]["vision_model"] = vision_model.params["vision_model"]
            model.params["visual_projection"]["kernel"] = vision_model.params["visual_projection"]["kernel"]
        else:
            model.params["vision_model"] = vision_model.params

        model.params["text_model"] = text_model.params

        return model
