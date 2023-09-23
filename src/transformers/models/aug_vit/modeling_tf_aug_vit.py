# coding=utf-8
# Copyright 2022 Tensorgirl and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 AugViT model. """



import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...utils import (
    DUMMY_INPUTS,
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from ...modeling_tf_utils import (
    TFPreTrainedModel,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_aug_vit import AugViTConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "tensorgirl/TFaugvit"
_CONFIG_FOR_DOC = "AugViTConfig"

TF_AUG_VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tensorgirl/TFaugvit",
    # See all AugViT models at https://huggingface.co/models?filter=aug_vit
]


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
import numpy as np


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
def gelu(x):

    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

class PreNorm(Layer):
    def __init__(self,fn,name):
        super(PreNorm, self).__init__(name=name)
        self.norm = nn.LayerNormalization(name=f'{name}/layernorm')
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)


class MLP(Layer):
    def __init__(self, dim, hidden_dim, name,dropout=0.0):
        super(MLP, self).__init__(name=name)
        self.net = Sequential([
            nn.Dense(units=hidden_dim,activation=gelu,name=f'{name}/den1'),

            nn.Dropout(rate=dropout,name=f'{name}/drop1'),
            nn.Dense(units=dim,name=f'{name}/den2'),
            nn.Dropout(rate=dropout,name=f'{name}/drop2')
        ],name=f'{name}/seq1')

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, name,heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__(name=name)
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(name=f'{name}/soft')
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False,name=f'{name}/den1')

        if project_out:
            self.to_out = [
                nn.Dense(units=dim,name=f'{name}/den2'),
                nn.Dropout(rate=dropout,name=f'{name}/drop1')
            ]
        else:
            self.to_out = []
        self.to_out = Sequential(self.to_out,name=f'{name}/seq')

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        # x = tf.matmul(attn, v)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, name,dropout=0.0):
        super(Transformer, self).__init__(True,name)

        self.layers = []

        for i in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,name=f'{name}/att{i}'),name=f'{name}preno{i}'),
                PreNorm(nn.Dense(dim,activation=gelu,name=f'{name}/den{i}'),name=f'{name}preno1{i}'),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout,name=f'{name}/mlp{i}'),name=f'{name}preno2{i}'),
                PreNorm(nn.Dense(dim,activation=gelu,name=f'{name}/den2{i}'),name=f'{name}preno3{i}'),
            ])


    def call(self, x, training=True):
        for attn,aug_attn, mlp, augs in self.layers:
            x = attn(x, training=training) + x + aug_attn(x, training=training)
            x = mlp(x, training=training) + x + augs(x, training=training)
        return x
        
@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class AUGViT(Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,name='augvit',
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0):

        super(AUGViT, self).__init__(name=name)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.patch_den=    nn.Dense(units=dim,name='patchden')
        

        self.pos_embedding = AddPositionEmbs(name="Transformer/posembed_input")
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]),name='cls',trainable=True)
        self.dropout = nn.Dropout(rate=emb_dropout,name='drop')
        # self.pos_embedding = tf.Variable(initial_value=tf.random_normal_initializer(stddev=0.06)(
        #         shape=(1, num_patches + 1, dim)),name='pos_emb',trainable=True)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout,name='trans')

        self.pool = pool

        self.mlp_head = Sequential([
            nn.LayerNormalization(name='layernorm'),
            nn.Dense(units=num_classes,name='dense12')
        ], name='mlp_head')
                     
    def call(self, img, training=True, **kwargs):
        x = self.patch_embedding(img)
        x = self.patch_den(x)
        b, n, d = x.shape
        # print(x.shape)
        cls_tokens = tf.cast(
            tf.broadcast_to(self.cls_token, [b, 1, d]),
            dtype=x.dtype,
        )
        x = tf.concat([cls_tokens, x], axis=1)
        # print(x.shape,cls_tokens.shape )
        x= self.pos_embedding(x)
        
        # print(x.shape,pos.shape,self.pos_embedding.shape)
        x = self.dropout(x, training=training)
        # print(x.shape)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x




from typing import Dict, Optional, Tuple, Union

class TFAugViTForImageClassification(TFPreTrainedModel):
    config_class = AugViTConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = AUGViT(
            image_size = config.image_size,
            patch_size = config.patch_size,
            num_classes = config.num_classes,
            dim = config.dim,
            depth = config.depth,
            heads = config.heads,
            mlp_dim = config.mlp_dim,
            dropout = config.dropout,
            emb_dropout =config.emb_dropout
        )

    def call(self, pixel_values: tf.Tensor | None = None,
        output_hidden_states: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs):
        inp = pixel_values['pixel_values']
        if inp.shape[-1]!=3:
            inp = tf.transpose(inp,[0,2,3,1])
        logits = self.model(inp)
        return logits
