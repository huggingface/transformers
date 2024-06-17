# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from .feature_extraction_utils import BatchFeature
from .tokenization_utils_base import BatchEncoding
from .utils import logging


logger = logging.get_logger(__name__)


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU. It is
    meant as a workaround for the [following issue](https://github.com/tensorflow/tensorflow/issues/55682), and will be
    removed after it gets fixed. The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that
    `softmax(x) = softmax(x + c)` (see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html).

    Args:
        logits (`tf.Tensor`):
            Must be one of the following types: half, float32, float64.
        axis (`int`, *optional*):
            The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`, *optional*):
            A name for the operation.

    Returns:
        `tf.Tensor`:
            A Tensor. Has the same type and shape as logits.
    """
    # TODO: When the issue linked above gets sorted, add a check on TF version here and use the original function if
    # it has the fix. After we drop the support for unfixed versions, remove this function.
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


def functional_layernorm(inputs, weight, bias, epsilon=1e-5, axis=-1):
    # This is a very simplified functional layernorm, designed to duplicate
    # the functionality of PyTorch nn.functional.layer_norm when this is needed to port
    # models in Transformers.

    if weight.shape.rank != 1 or bias.shape.rank != 1 or not isinstance(axis, int):
        raise NotImplementedError("Only 1D weight and bias tensors are supported for now, with only a single axis.")

    # Get mean and variance on the axis to be normalized
    mean, variance = tf.nn.moments(inputs, axes=[axis], keepdims=True)

    if axis != -1:
        # Reshape scale and weight to have the same rank as inputs, but with 1 dimensions
        # on every dimension except axis
        shape = [1] * inputs.shape.rank
        shape[axis] = shape_list(inputs)[axis]
        weight = tf.reshape(weight, shape)
        bias = tf.reshape(bias, shape)

    # Compute layer normalization using the batch_normalization
    # function.
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=bias,
        scale=weight,
        variance_epsilon=epsilon,
    )
    return outputs


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale: float = None
):
    """TF equivalent for torch's nn.functional.scaled_dot_product_attention"""
    if dropout_p != 0.0:
        raise ValueError(
            "Dropout is not supported in this implementation - file an issue "
            "with Transformers and ping @Rocketknight1 if you need it for a port!"
        )
    if is_causal and attn_mask is not None:
        raise ValueError("You cannot specify an attn_mask and is_causal at the same time!")
    if is_causal:
        attn_mask = tf.ones((tf.shape(query)[-2], tf.shape(key)[-2]), dtype=tf.int32)
        attn_mask = tf.experimental.numpy.tril(attn_mask, k=0)
    if attn_mask is not None and (attn_mask.dtype.is_integer or attn_mask.dtype.is_bool):
        # Convert boolean mask to a negative logit bias
        attn_mask = tf.where(attn_mask > 0, tf.cast(0.0, query.dtype), tf.cast(-1000.0, query.dtype))
    logits = tf.einsum("...qd, ...kd -> ...qk", query, key)
    if scale is None:
        scale = tf.cast(tf.shape(key)[-1], logits.dtype) ** -0.5
    logits *= scale  # scale by 1/sqrt(key_dim)
    if attn_mask is not None:
        logits += attn_mask
    probs = tf.nn.softmax(logits)
    return probs @ value


def flatten(input, start_dim=0, end_dim=-1):
    # Replicates the behavior of torch.flatten in TF

    # If end_dim or start_dim is negative, count them from the end
    if end_dim < 0:
        end_dim += input.shape.rank
    if start_dim < 0:
        start_dim += input.shape.rank

    if start_dim == end_dim:
        return input

    in_shape = tf.shape(input)
    flattened_dim = tf.math.reduce_prod(in_shape[start_dim : end_dim + 1])
    out_shape = tf.concat([in_shape[:start_dim], [flattened_dim], in_shape[end_dim + 1 :]], axis=0)
    return tf.reshape(input, out_shape)


def invert_attention_mask(encoder_attention_mask: tf.Tensor) -> tf.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `tf.Tensor`: The inverted attention mask.
    """
    if not isinstance(encoder_attention_mask, tf.Tensor):
        encoder_attention_mask = tf.convert_to_tensor(encoder_attention_mask)  # Catches stray NumPy inputs
    if encoder_attention_mask.shape.rank == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.shape.rank == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = (
        tf.cast(1, encoder_attention_mask.dtype) - encoder_extended_attention_mask
    ) * encoder_extended_attention_mask.dtype.min

    return encoder_extended_attention_mask


def check_embeddings_within_bounds(tensor: tf.Tensor, embed_dim: int, tensor_name: str = "input_ids") -> None:
    """
    `tf.gather`, on which TF embedding layers are based, won't check positive out of bound indices on GPU, returning
    zeros instead. This function adds a check against that dangerous silent behavior.

    Args:
        tensor (`tf.Tensor`): The tensor of indices to check.
        embed_dim (`int`): The embedding dimension.
        tensor_name (`str`, *optional*): The name of the tensor to use in the error message.
    """
    tf.debugging.assert_less(
        tensor,
        tf.cast(embed_dim, dtype=tensor.dtype),
        message=(
            f"The maximum value of {tensor_name} ({tf.math.reduce_max(tensor)}) must be smaller than the embedding "
            f"layer's input dimension ({embed_dim}). The likely cause is some problem at tokenization time."
        ),
    )


def save_attributes_to_hdf5_group(group, name, data):
    """Saves attributes (data) of the specified name into the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not able to store data larger than
    HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to save.
        data: Attributes data to store.

    Raises:
      RuntimeError: If any single attribute is too large to be saved.

    Copied from Keras to Transformers to avoid versioning issues.
    """
    HDF5_OBJECT_HEADER_LIMIT = 64512
    # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
    # because in that case even chunking the array would not make the saving
    # possible.
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

    # Expecting this to never be true.
    if bad_attributes:
        raise RuntimeError(
            "The following attributes cannot be saved to HDF5 file because "
            f"they are larger than {HDF5_OBJECT_HEADER_LIMIT} "
            f"bytes: {bad_attributes}"
        )

    data_npy = np.asarray(data)

    num_chunks = 1
    chunked_data = np.array_split(data_npy, num_chunks)

    # This will never loop forever thanks to the test above.
    while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)

    if num_chunks > 1:
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs["%s%d" % (name, chunk_id)] = chunk_data
    else:
        group.attrs[name] = data


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not able to store data larger than
    HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    Returns:
        data: Attributes data.

    Copied from Keras to Transformers to avoid versioning issues.
    """
    if name in group.attrs:
        data = [n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while "%s%d" % (name, chunk_id) in group.attrs:
            data.extend(
                [n.decode("utf8") if hasattr(n, "decode") else n for n in group.attrs["%s%d" % (name, chunk_id)]]
            )
            chunk_id += 1
    return data


def expand_1d(data):
    """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s.
    Copied from Keras to here to avoid versioning issues."""

    def _expand_single_1d_tensor(t):
        if isinstance(t, tf.Tensor) and t.shape.rank == 1:
            return tf.expand_dims(t, axis=-1)
        return t

    return tf.nest.map_structure(_expand_single_1d_tensor, data)


def convert_batch_encoding(*args, **kwargs):
    # Convert HF BatchEncoding/BatchFeature objects in the inputs to dicts that Keras understands
    if args and isinstance(args[0], (BatchEncoding, BatchFeature)):
        args = list(args)
        args[0] = dict(args[0])
    elif "x" in kwargs and isinstance(kwargs["x"], (BatchEncoding, BatchFeature)):
        kwargs["x"] = dict(kwargs["x"])
    return args, kwargs
