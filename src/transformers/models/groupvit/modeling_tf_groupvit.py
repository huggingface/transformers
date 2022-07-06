# coding=utf-8
# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
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
""" TF 2.0 GroupViT model."""


import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling

# Public API
from ...modeling_tf_utils import (
    DUMMY_INPUTS,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"

GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/groupvit-gcc-yfcc",
    # See all GroupViT models at https://huggingface.co/models?filter=groupvit
]


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# Copied from transformers.models.clip.modeling_tf_clip.clip_loss with clip->groupvit
def groupvit_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


def hard_softmax(logits: tf.Tensor, dim: int):
    """
    Reference: https://gist.github.com/ariG23498/b9eca9a73fc9d93884fb2f59c4a303fb
    """
    y_soft = stable_softmax(logits, dim)
    # Straight through.
    index = tf.argmax(y_soft, dim)
    y_hard = tf.one_hot(
        index,
        depth=shape_list(logits)[dim],
        axis=dim,
    )
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft

    return ret


def gumbel_softmax(logits: tf.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> tf.Tensor:
    gumbel_dist = tfp.distributions.Gumbel(0.0, 1.0)
    gumbels = gumbel_dist.sample(shape_list(logits))

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = stable_softmax(gumbels, dim)

    if hard:
        # Straight through.
        index = tf.argmax(y_soft, dim)
        y_hard = tf.one_hot(
            index,
            depth=shape_list(logits)[dim],
            axis=dim,
        )
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def resize_attention_map(attentions: tf.Tensor, height: int, width: int, align_corners: Optional[bool] = False) -> tf.Tensor:
    """
    Args:
        attentions (`tf.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]
        height (`int`): height of the output attention map
        width (`int`): width of the output attention map
        align_corners (`bool`, *optional*): the `align_corner` argument for `nn.functional.interpolate`.

    Returns:
        `torch.Tensor`: resized attention map of shape [batch_size, groups, height, width]
    """

    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = shape_list(attentions)[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = shape_list(attentions)[2] // feat_height

    batch_size = shape_list(attentions)[0]
    groups = shape_list(attentions)[1]  # number of group token
    # [batch_size, groups, height*width, groups] -> [batch_size, groups, height, width]
    attentions = tf.reshape(attentions, (batch_size, groups, feat_height, feat_width))
    attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
    attentions = tf.image.resize(
        attentions, size=(height, width), method="bilinear",
    )
    attentions = tf.transpose(attentions, perm=(0, 3, 1, 2))
    return attentions


def get_grouping_from_attentions(attentions: Tuple[tf.Tensor], hw_shape: int) -> tf.Tensor:
    """
    Args:
        attentions (`tuple(tf.Tensor)`: tuple of attention maps returned by `TFGroupViTVisionTransformer`
        hw_shape (`tuple(int)`): height and width of the output attention map
    Returns:
        `tf.Tensor`: the attention map of shape [batch_size, groups, height, width]
    """

    attn_maps = []
    
    prev_attn_masks = None
    for attn_masks in attentions:
        # [batch_size, num_groups, height x width] -> [batch_size, height x width, num_groups]
        attn_masks = tf.transpose(attn_masks, perm=(0, 2, 1))
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            prev_attn_masks = prev_attn_masks @ attn_masks
        # [batch_size, heightxwidth, num_groups] -> [batch_size, num_groups, heightxwidth] -> [batch_size, num_groups, height, width]
        cur_attn_map = resize_attention_map(
            tf.transpose(prev_attn_masks, perm=(0, 2, 1)), *hw_shape
        )
        attn_maps.append(cur_attn_map)

    # [batch_size, num_groups, height, width]
    final_grouping = attn_maps[-1]

    return final_grouping

