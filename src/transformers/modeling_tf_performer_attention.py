from typing import Optional, Union
import logging
import math
import numpy as np
import random
import tensorflow as tf


from .configuration_performer_attention import PerformerAttentionConfig
from .modeling_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer
)

KERNEL_CALLABLES = {
    'cosh': lambda x, h: tf.concat((tf.exp(h + x), tf.exp(h - x)), dim=-1),
    'exp': lambda x, h: tf.exp(h + x),  # Default
    'elu': lambda x: tf.nn.elu(x) + 1,
    'relu': tf.nn.relu
}

SHORT_SEQUENCE_BEHAVIOR_CALLABLES = {
    'use_softmax_eval_only': lambda L, M, training: False if training else L < 2.0 * M,
    'use_softmax_eval_and_train': lambda L, M, training: L < 2.0 * M, 
    'never_use_softmax': lambda L, M, training: False
}


class TFPerformerAttention(tf.keras.layers.Layer):
    def __init__(self, config: Optional[Union[dict, PerformerAttentionConfig]] = None, **kwargs):
        super().__init__()
        
        if config is not None:
            # config can either be a dictionary or a PerformerAttentionConfig object
            if not isinstance(config, dict):
                config = config.__dict__
            
            # Just copy over all the parameters
            self.__dict__.update(config)
        else:
            # Make sure we have all the default values filled in
            config = PerformerAttentionConfig(**kwargs)
            kwargs = config.__dict__
        
        # kwargs take precedence over the default values that might be stored in the config object
        self.__dict__.update(kwargs)
        
        if self.num_heads is None or self.d_model is None:
            raise ValueError("PerformerAttention: num_heads and d_model must be non-None")
        
        self.dropout = tf.keras.layers.Dropout(rate=self.attention_dropout)
        self.calls_since_last_redraw = 0
        self.random_features = None
        
        behavior = self.short_sequence_behavior
        if not behavior:
            behavior = 'never_use_softmax' if self.kernel_type == 'relu' else 'use_softmax_eval_only'
            self.should_fallback_to_softmax = SHORT_SEQUENCE_BEHAVIOR_CALLABLES[behavior]
        
        elif self.kernel_type == 'relu' and behavior != 'never_use_softmax':
            raise ValueError(f"PerformerAttention: short_sequence_behavior = {behavior} cannot be combined with the relu "
                             "kernel type")
        
        elif isinstance(behavior, str):
            self.should_fallback_to_softmax = SHORT_SEQUENCE_BEHAVIOR_CALLABLES[behavior]
        elif callable(behavior):
            self.should_fallback_to_softmax = behavior
        else:
            raise ValueError("PerformerAttention: short_sequence_behavior must be either str or Callable")
        
        self.kernel_fn = KERNEL_CALLABLES[self.kernel_type]

        assert self.d_model % self.num_heads == 0
        
        if self.use_qkv_linear_layers:
            self.q_lin = tf.keras.layers.Dense(units=self.d_model)
            self.k_lin = tf.keras.layers.Dense(units=self.d_model)
            self.v_lin = tf.keras.layers.Dense(units=self.d_model)
        
        self.out_lin = tf.keras.layers.Dense(units=self.d_model)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.d_model // self.num_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        if self.use_qkv_linear_layers:
            self.q_lin = prune_linear_layer(self.q_lin, index)
            self.k_lin = prune_linear_layer(self.k_lin, index)
            self.v_lin = prune_linear_layer(self.v_lin, index)
        
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.num_heads = self.num_heads - len(heads)
        self.d_model = attention_head_size * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def redraw_features_now(self):
        self._generate_feature_matrix()
        
        if self.training and self.redraw_verbose:
            logging.info("PerformerAttention: Just redrew random features.")
        
        self.calls_since_last_redraw = 0

    def call(self, query, key, value, mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: tf.tensor(bs, num_heads, seq_length, seq_length) Attention weights context: tf.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.d_model, 'Dimensions do not match: %s input vs %s configured' % (dim, self.d_model)
        # assert key.size() == value.size()

        dim_per_head = self.d_model // self.num_heads
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            new_shape = tf.concat((x.shape[:-1], tf.constant([self.num_heads, dim_per_head])), axis=0)
            return tf.transpose(tf.reshape(x, new_shape), perm=[0, 2, 1, 3])
        
        if self.use_qkv_linear_layers:
            q = self.q_lin(query)
            k = self.k_lin(key)
            v = self.v_lin(value)
        else:
            q, k, v = query, key, value
        
        # (bs, num_heads, q_length, dim_per_head)
        q, k, v = (shape(x) for x in (q, k, v))
        
        # If the sequence length is short enough that FAVOR+ would use considerably more time and/or memory than just
        # using softmax attention, use softmax. This works because FAVOR+ is an unbiased estimator of softmax attention.
        m = round(dim_per_head * math.log(dim_per_head))  # m is the number of random features
        if self.should_fallback_to_softmax(q_length, m, self.training):
            scores = q @ tf.linalg.matrix_transpose(k) / (dim ** 0.5)
            
            if mask is not None:
                mask = tf.reshape((mask == 0), mask_reshp)  # .expand_as(scores)  # (bs, num_heads, q_length, k_length)
                scores -= 1e9 * tf.cast(mask, q.dtype)  # (bs, num_heads, q_length, k_length)
            
            attn_map = tf.nn.softmax(scores, dim=-1)
            attn_map = self.dropout(attn_map)  # (bs, num_heads, q_length, k_length)
            return self._finalize_attention_output(attn_map @ v, head_mask, attn_map)
        
        # When we're using FAVOR+ we can't output the attention matrix
        if output_attentions:
            raise ValueError("TFPerformerAttention: Can't output attention maps when using FAVOR+ linear attention.")
        
        self._redraw_features_if_needed()
        
        # Get the transformed values of Q and K
        q_prime, k_prime = self.get_projected_queries_and_keys(q, k)
        return self.compute_attention_with_projected_queries_and_keys(q_prime, k_prime, v, mask, head_mask)
    
    # Turns Q into Q', K into K'
    def get_projected_queries_and_keys(self, q, k):
        # Broadcast the feature matrix across the batch dimension
        # new_shape = list(q.shape)
        # new_shape[-2] = self.random_features.shape[-2]
        W_t = tf.linalg.matrix_transpose(self.random_features)  # .expand(new_shape)
        
        # Instead of dividing the product QK^T by sqrt(d), we divide Q and K by the 4th root of d.
        q = q / (self.d_model ** 0.25)
        k = k / (self.d_model ** 0.25)
        
        projected_q = q @ W_t
        projected_k = k @ W_t
        
        # Special logic for kernels that attempt to approximate softmax
        if self.kernel_type in ('cosh', 'exp'):
            # The h(x) function is defined in Lemma 1 in Choromanski et al. pg. 4 as exp(-||x||**2 / 2). For numerical
            # stability we leverage the fact that exp(x)*exp(y) = exp(x + y) here and delay computing the exp().
            h_of_q = -tf.reduce_sum(q ** 2, dim=-1, keepdim=True) / 2
            h_of_k = -tf.reduce_sum(k ** 2, dim=-1, keepdim=True) / 2
            
            # Compute the numerical stabilizer that we subtract from the input to exp(). For some reason the original
            # Jax implementation uses different types of stabilizers for queries vs. keys, and we follow that here.
            q_stabilizer = tf.math.reduce_max(h_of_q, axis=-1, keepdims=True)
            
            # This is just a scalar
            k_stabilizer = tf.math.reduce_max(h_of_k)
            
            q_kernel_output = self.kernel_fn(projected_q - q_stabilizer, h_of_q)
            k_kernel_output = self.kernel_fn(projected_k - k_stabilizer, h_of_k)
            
            # By multiplying by 1/sqrt(m), we ensure the final matrix product will contain a factor of 1/m. This means
            # each row of Q'K'^T can be interpreted as an average over the exp(omega^T * q) * exp(omega^T * k) terms.
            normalizing_constant = (q_kernel_output.shape[-1] ** -0.5)
            
            q_prime = normalizing_constant * (q_kernel_output + self.kernel_epsilon)
            k_prime = normalizing_constant * (k_kernel_output + self.kernel_epsilon)
            return q_prime, k_prime
        
        # Generalized attention (ReLU, ELU...)
        else:
            return tuple(self.kernel_fn(x) + self.kernel_epsilon for x in (projected_q, projected_k))
    
    def compute_attention_with_projected_queries_and_keys(self, q_prime, k_prime, v, mask = None, head_mask = None):
        # Apply the padding mask to K'. Also applying it to Q' would be redundant.
        if mask is not None:
            k_prime *= tf.expand_dims(tf.expand_dims(mask, 1), -1)#.expand_as(k_prime)

        if self.causal:
            output = _causal_numerator(q_prime, k_prime, v)
        else:
            k_prime_t = tf.linalg.matrix_transpose(k_prime)
            output = q_prime @ (k_prime_t @ v)
        
        # Ensure that the output vectors are convex combinations of input vectors; that is,
        # the implied attention scores sum to 1
        if self.normalize_output:
            if self.causal:
                d = _causal_denominator
            else:
                # Equivalent to multiplying K'^T by a ones vector
                d = q_prime @ tf.expand_dims(tf.math.reduce_sum(k_prime), -1)
            
            # Avoid dividing by very small numbers
            d += 2 * self.normalization_stabilizer * (tf.abs(d) <= self.normalization_stabilizer)
            output /= d
        
        return self._finalize_attention_output(output, head_mask)
    
    def _finalize_attention_output(self, context, head_mask=None, att_map_to_output=None):
        def unshape(x):
            """ group heads """
            x = tf.transpose(context, perm=[0, 2, 1, 3])  # [...seq_len, num_heads, dim_per_head]
            new_last_dim = tf.constant(x.shape[-2] * x.shape[-1])  # Multiply num_heads * dim_per_head
            return tf.reshape(x, tf.concat((x.shape[:-2], new_last_dim), axis=0))
        
        # Mask heads if we want to
        if head_mask is not None:
            context = context * head_mask
            
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if att_map_to_output:
            return context, att_map_to_output
        else:
            return context,

    def _generate_feature_matrix(self):
        dim_per_head = self.d_model // self.num_heads
        num_rows = round(dim_per_head * math.log(dim_per_head))
        
        if not self.use_orthogonal_features:
            return tf.random.normal((num_rows, dim_per_head))
        
        def get_square_block(size):
            with tf.device('/CPU:0'):
                unstructured_block = tf.random.normal((size, size))
                orthog, r = tf.linalg.qr(unstructured_block)

            return tf.transpose(orthog)

        num_full_blocks = num_rows // dim_per_head
        block_list = [get_square_block(dim_per_head) for _ in range(num_full_blocks)]
        
        remaining_rows = num_rows - num_full_blocks * dim_per_head
        if remaining_rows > 0:
            q = get_square_block(dim_per_head)
            block_list.append(q[:remaining_rows])
        
        final_matrix = tf.concat(block_list)
        
        # This option yields SMREG
        if self.regularize_feature_norms:
            final_matrix *= dim_per_head ** 0.5
        else:
            # Hack to make the matrix columns have the norm we would expect them to have if they were sampled straight
            # from a Gaussian, instead of being all norm 1 since they went through QR decomposition
            multiplier = tf.random.normal((num_rows, dim_per_head)).norm(dim = 1)
            final_matrix = tf.linalg.diag(multiplier) @ final_matrix

        self.random_features = final_matrix
    
    def _redraw_features_if_needed(self):
        # We haven't created the projection matrix yet, let's create it
        if self.random_features is None:
            self._generate_feature_matrix()
        
        elif self.feature_redraw_interval is not None:
            if self.redraw_stochastically:
                # random.random() returns a float between 0.0 and 1.0, so this expression
                # evaluates to True with probability 1. / self.feature_redraw_interval
                if random.random() < 1. / self.feature_redraw_interval:
                    self.redraw_features_now()
            
            # It's time to redraw the projection matrix
            elif self.calls_since_last_redraw >= self.feature_redraw_interval:
                self.redraw_features_now()
        
            # Keep track of how many forward passes we do before we redraw again
            else:
                self.calls_since_last_redraw += 1

# An asymptotically fast way of making random orthogonal matrices- O(nlog(n)) vs. O(n^3) for Gram-Schmidt QR
def _create_products_of_givens_rotations(num_rows, seed):
    r"""Constructs a 2D-tensor which is a product of Givens random rotations.
    Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
    random rotation. The resulting tensor mimics a matrix taken uniformly at
    random form the orthogonal group.
    Args:
      num_rows: number of rows/columns of the resulting 2D-tensor.
      seed: random seed.
    Returns:
      The product of Givens random rotations.
    """
    q = np.eye(num_rows)   # Start with identity matrix
    np.random.seed(seed)

    # Each iteration, pick two random rows and a random angle between 0.0 and pi, and rotate
    num_rotations = num_rows * int(math.ceil(math.log(float(num_rows))))     # N log N rotations
    angles = np.random.uniform(0.0, math.pi, size=num_rotations)             # Random rotation angles
    row_pairs = np.random.choice(num_rows, (num_rotations, 2))
    row_pairs.sort(axis=1)  # Make sure the first row in each pair is the smaller of the two

    for n, random_angle in enumerate(angles):
        index_i, index_j = row_pairs[n]

        # This is for speed as well as accuracy- without this check the resulting matrices will not be valid rotation
        # matrices (their determinants will not be 1) since we will end up modifying the same row twice (see below)
        if index_i == index_j:
            continue

        row_i, row_j = q[index_i], q[index_j]

        new_row_i = math.cos(random_angle) * row_i + math.sin(random_angle) * row_j
        new_row_j = -math.sin(random_angle) * row_i + math.cos(random_angle) * row_j
        q[index_i], q[index_j] = new_row_i, new_row_j

    return tf.constant(q, dtype=tf.float32)

def _noncausal_numerator(q_prime, k_prime, v):
    """Computes not-normalized FAVOR noncausal attention AV.
    Args:
      q_prime: query_prime tensor of the shape [L,B,H,M].
      k_prime: key_prime tensor of the shape [L,B,H,M].
      v: value tensor of the shape [L,B,H,D].
    Returns:
      Not-normalized FAVOR noncausal attention AV.
    """
    kvs = tf.einsum("lbhm,lbhd->bhmd", k_prime, v)
    return tf.einsum("lbhm,bhmd->lbhd", q_prime, kvs)


def _noncausal_denominator(q_prime, k_prime):
    """Computes FAVOR normalizer in noncausal attention.
    Args:
      q_prime: query_prime tensor of the shape [L,B,H,M].
      k_prime: key_prime tensor of the shape [L,B,H,M].
    Returns:
      FAVOR normalizer in noncausal attention.
    """
    all_ones = tf.ones([k_prime.shape[0]])
    ks_sum = tf.einsum("lbhm,l->bhm", k_prime, all_ones)
    return tf.einsum("lbhm,bhm->lbh", q_prime, ks_sum)

# Custom gradient functions
@tf.custom_gradient
def _causal_numerator(q_prime, k_prime, v):
    """Computes not-normalized FAVOR causal attention using the prefix-sum method.
    Args:
      q_prime: query_prime tensor of the shape [L,B,H,M].
      k_prime: key_prime tensor of the shape [L,B,H,M].
      v: value tensor of the shape [L,B,H,D].
    Returns:
      Not-normalized FAVOR causal attention A_{masked}V.
    """
    result = []
    sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", k_prime[0], v[0]))

    for index in range(q_prime.shape[0]):
        sums = sums + tf.einsum("ijk,ijl->ijkl", k_prime[index], v[index])
        result.append(tf.einsum("ijkl,ijk->ijl", sums, q_prime[index])[None, Ellipsis])

    result = tf.concat(result, axis=0)

    # Function called later by TensorFlow to compute the gradient, if needed
    def grad(res_grad):
        grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", k_prime[0], v[0]))
        gr_sums = sums
        q_grads = []
        k_grads = []
        v_grads = []

        for i in range(q_prime.shape[0] - 1, -1, -1):
            q_grads.append(
                tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[i])[None, Ellipsis])
            grads = grads + tf.einsum("ijk,ijl->ijkl", q_prime[i], res_grad[i])
            k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, v[i])[None, Ellipsis])
            v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, k_prime[i])[None, Ellipsis])
            gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", k_prime[i], v[i])

        q_grads = tf.concat(q_grads[::-1], axis=0)
        k_grads = tf.concat(k_grads[::-1], axis=0)
        v_grads = tf.concat(v_grads[::-1], axis=0)
        return q_grads, k_grads, v_grads

    return result, grad

@tf.custom_gradient
def _causal_denominator(q_prime, k_prime):
    """Computes FAVOR normalizer in causal attention using the prefix-sum method.
    Args:
      q_prime: query_prime tensor of the shape [L,B,H,M].
      k_prime: key_prime tensor of the shape [L,B,H,M].
    Returns:
      FAVOR normalizer in causal attention.
    """

    result = []
    sums = tf.zeros_like(k_prime[0])

    for index in range(q_prime.shape[0]):
        sums = sums + k_prime[index]
        result.append(tf.reduce_sum(q_prime[index] * sums, axis=2)[None, Ellipsis])

    result = tf.concat(result, axis=0)

    # Function called later by TensorFlow to compute the gradient, if needed
    def grad(res_grad):
        k_grad = tf.zeros_like(k_prime[0])
        gr_sums = sums

        q_grads = []
        k_grads = []

        for i in range(q_prime.shape[0] - 1, -1, -1):

            q_grads.append(
                tf.einsum("ijk,ij->ijk", gr_sums, res_grad[i])[None, Ellipsis])
            k_grad = k_grad + tf.einsum("ijk,ij->ijk", q_prime[i], res_grad[i])
            k_grads.append(k_grad[None, Ellipsis])
            gr_sums = gr_sums - k_prime[i]

        q_grads = tf.concat(q_grads[::-1], axis=0)
        k_grads = tf.concat(k_grads[::-1], axis=0)

        return q_grads, k_grads

    return result, grad
