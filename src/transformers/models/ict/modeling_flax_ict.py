# coding=utf-8
# Copyright 2023 The Google Flax Team Authors and The HuggingFace Inc. team.
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
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_ict import ICTConfig


ICT_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`ICTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

ICT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTPatchEmbeddings with ViT->ICT
class FlaxICTPatchEmbeddings(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTEmbeddings with ViT->ICT
class FlaxICTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, 1, self.config.hidden_size),
        )
        self.patch_embeddings = FlaxICTPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(self.config.initializer_range**2, "fan_in", "truncated_normal"),
            (1, num_patches + 1, self.config.hidden_size),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]

        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfAttention with ViT->ICT
class FlaxICTSelfAttention(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfOutput with ViT->ICT
class FlaxICTSelfOutput(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTAttention with ViT->ICT
class FlaxICTAttention(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxICTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxICTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTIntermediate with ViT->ICT
class FlaxICTIntermediate(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTOutput with ViT->ICT
class FlaxICTOutput(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + attention_output
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTLayer with ViT->ICT
class FlaxICTLayer(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxICTAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxICTIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxICTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ICT, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # first residual connection
        attention_output = attention_output + hidden_states

        # in ICT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(attention_output)

        hidden_states = self.intermediate(layer_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTLayerCollection with ViT->ICT
class FlaxICTLayerCollection(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxICTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTEncoder with ViT->ICT
class FlaxICTEncoder(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxICTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTPooler with ViT->ICT
class FlaxICTPooler(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTPreTrainedModel with VIT->ICT,ViT->ICT,vit->ict
class FlaxICTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ICTConfig
    base_model_prefix = "ict"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ICTConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(ICT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
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

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTModule with ViT->ICT
class FlaxICTModule(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxICTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxICTEncoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxICTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare ICT Model transformer outputting raw hidden-states without any specific head on top.",
    ICT_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTModel with ViT->ICT
class FlaxICTModel(FlaxICTPreTrainedModel):
    module_class = FlaxICTModule


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxICTModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("sheonhan/image-completion-transformer")
    >>> model = FlaxICTModel.from_pretrained("sheonhan/image-completion-transformer")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxICTModel, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxICTModel, output_type=FlaxBaseModelOutputWithPooling, config_class=ICTConfig)


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTForImageClassificationModule with ViT->ICT,vit->ict
class FlaxICTForImageClassificationModule(nn.Module):
    config: ICTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.ict = FlaxICTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ict(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states[:, 0, :])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    ICT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    ICT_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTForImageClassification with ViT->ICT
class FlaxICTForImageClassification(FlaxICTPreTrainedModel):
    module_class = FlaxICTForImageClassificationModule


FLAX_VISION_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxICTForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("google/ict-base-patch16-224")
    >>> model = FlaxICTForImageClassification.from_pretrained("google/ict-base-patch16-224")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
    ```
"""

overwrite_call_docstring(FlaxICTForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)
append_replace_return_docstrings(
    FlaxICTForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=ICTConfig
)
