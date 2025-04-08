# This code was adapted from https://github.com/lucidrains/flamingo-pytorch licensed under the MIT License.
#
# MIT License
#
# Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""

Generic interface to various configurations of the Perceiver Resampler, that simply takes in a series of (potentially
time-indexed) contextual embeddings, and "resamples" (compresses) them down to a pre-specified number of latents! Note
that the Perceiver in general resamples based solely off the *long-range* context; there's a nice opportunity here to
prime the Perceiver Resampler with say a single layer's worth of language embeddings (the target domain), and use that
to softly "retrieve & compress" what we need --> this would be a novel contribution we should explore.

References:
    - DeepMind's Flamingo: https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model
    - Code borrowed w/ love from: https://github.com/lucidrains/flamingo-pytorch

"""

from typing import Optional, Tuple

import tensorflow as tf

from ...modeling_tf_utils import shape_list
from .configuration_idefics import IdeficsConfig


class TFIdeficsPerceiverResampler(tf.keras.layers.Layer):
    def __init__(
        self, config: IdeficsConfig, embed_dim: int, depth: int, n_heads: int, head_dim: int, n_latents: int, **kwargs
    ) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. :param embed_dim: Dimensionality of embeddings being fed
        to the Perceiver Resampler (also dimensionality of latent embeddings *returned* by the Perceiver Resampler.
        Could be e.g., VIT embed_dim, ResNet pool dim, and so on.

        Args:
            config (`IdeficsConfig`): config object
            embed_dim (`int`): The size of each embedding vector
            depth (`int`): Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
            n_heads (`int`): Number of heads in each Transformer block (for multi-headed self-attention).
            head_dim (`int`): Dimensionality of each head projection in the Transformer block.
            n_latents (`int`):
                Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).

        """
        super().__init__(**kwargs)
        self.embed_dim, self.n_heads, self.head_dim, self.n_latents = embed_dim, n_heads, head_dim, n_latents
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver

        self.intermediate_dim = (
            self.embed_dim * 4
            if not hasattr(config.vision_config, "embed_dim")
            else config.vision_config.embed_dim * 4
        )
        # Create Transformer Blocks
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                [
                    TFIdeficsPerceiverAttention(
                        self.embed_dim, self.n_heads, self.head_dim, self.qk_layer_norms, name=f"blocks.{i}.0"
                    ),
                    TFIdeficsMLP(self.intermediate_dim, config, name=f"blocks.{i}.1"),
                ]
            )

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    def build(self, input_shape):
        # Create Latents for Perceiver
        self.latents = self.add_weight(
            shape=(self.n_latents, self.embed_dim), initializer="random_normal", trainable=True, name="latents"
        )
        super().build(input_shape)

    def call(self, context: tf.Tensor) -> tf.Tensor:
        """Resample arbitrary length context & *compress* down to self.n_latents latent embeddings"""
        # tf.repeat(self.latents, "seq embed -> bsz seq embed", bsz=context.shape[0])
        latents = tf.expand_dims(self.latents, axis=0)
        latents = tf.tile(latents, [tf.shape(context)[0], 1, 1])
        # Feed through Perceiver Attention blocks...
        for attn, ff in self.blocks:
            latents = attn(context, latents) + latents
            latents = ff(latents) + latents
        return self.layer_norm(latents)


class TFIdeficsPerceiverAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, n_heads: int, head_dim: int, qk_layer_norms: bool, **kwargs) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__(**kwargs)
        self.embed_dim, self.n_heads, self.head_dim = embed_dim, n_heads, head_dim
        self.qk_layer_norms = qk_layer_norms
        # Normalization & Scaling
        self.context_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="context_layer_norm")
        self.latents_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="latents_layer_norm")
        if self.qk_layer_norms:
            self.q_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="q_layer_norm")
            self.k_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="k_layer_norm")

        self.qk_scale = self.head_dim**-0.5

        # Q, K, V Projection (no bias -- detail from Perceiver/Flamingo Papers).
        self.q_proj = tf.keras.layers.Dense(self.n_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.n_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.n_heads * self.head_dim, use_bias=False, name="v_proj")

        self.output_proj = tf.keras.layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def call(self, context: tf.Tensor, latents: tf.Tensor) -> tf.Tensor:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            context (`tf.Tensor`):
                Tensor of shape `[bsz, seq, embed_dim]` representing long-form context to resample.
            latents (`tf.Tensor`):
                Tensor of shape `[bsz, n_latents, embed_dim]` representing fixed length latents to compress to.

        Returns:
            `tf.Tensor`: Tensor of shape `[bsz, n_latents, embed_dim]` representing attention over latents w/ cross
            from context.
        """
        context = self.context_layer_norm(context)
        latents = self.latents_layer_norm(latents)
        batch_size, seq_length, embed_dim = shape_list(context)

        # Query, Key, Value Projections --> Note that in Flamingo, latents are *concatenated* with context prior to attn!
        #   Note: This results in queries w/ `seq = n_latents`, and keys, values with `seq = len(context) + n_latents`
        q = self.q_proj(latents)
        k = self.k_proj(tf.concat([context, latents], axis=-2))
        v = self.v_proj(tf.concat([context, latents], axis=-2))

        # Multiheaded Self-Attention w/ stable softmax (subtract per-row max -- `amax` -- before softmax call)
        #   =>> `attn` should be a 2D matrix of shape [n_latents x (context + n_latents)]
        q, k, v = [
            tf.transpose(tf.reshape(x, (batch_size, x.shape[1], self.n_heads, self.head_dim)), perm=[0, 2, 1, 3])
            for x in (q, k, v)
        ]

        if self.qk_layer_norms:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)

        scores = tf.einsum("... i d, ... j d -> ... i j", q * self.qk_scale, k)
        stabilized_scores = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
        attn = tf.nn.softmax(stabilized_scores, axis=-1)

        # Attend & project back to output...
        resampled = tf.einsum("... i j, ... j d -> ... i d", attn, v)
        return self.output_proj(
            tf.reshape(tf.transpose(resampled, perm=[0, 2, 1, 3]), (batch_size, -1, self.n_heads * self.head_dim))
        )


class TFIdeficsMLP(tf.keras.layers.Layer):
    def __init__(self, intermediate_size, config: IdeficsConfig, **kwargs):
        """Simple MLP block with intermediate_size and embedding size"""
        super().__init__(**kwargs)
        self.embed_dim = config.vision_config.embed_dim
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln")
        self.fc = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="fc")
        self.act = tf.keras.layers.ReLU(name="act")
        self.c_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=False, name="c_proj")

    def call(self, hidden_states: Optional[Tuple[tf.Tensor]]) -> tf.Tensor:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        return hidden_states
