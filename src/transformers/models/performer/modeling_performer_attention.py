from itertools import count
from torch import nn
from typing import Callable, Optional, Union
import logging
import math
import random
import torch
import torch.nn.functional as F

from .configuration_performer_attention import PerformerAttentionConfig, PerformerKernel, OrthogonalFeatureAlgorithm
from ...modeling_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer
)

KERNEL_CALLABLES = {
    PerformerKernel.cosh: lambda x, h: torch.cat((torch.exp(h + x), torch.exp(h - x)), dim=-1),
    PerformerKernel.exp: lambda x, h: torch.exp(h + x),  # Default
    PerformerKernel.elu: lambda x: F.elu(x) + 1,
    PerformerKernel.relu: F.relu
}


def resolve_enum(enum_class, value):
    return enum_class[value] if isinstance(value, str) else value


class PerformerAttention(nn.Module):
    causal_numerator_fn = None  # Either refers to _headwise_causal_numerator or the fast_transformers CUDA kernel

    def __init__(self, config: Optional[Union[dict, PerformerAttentionConfig]] = None, **kwargs):
        super().__init__()

        if isinstance(config, dict):
            config = PerformerAttentionConfig(**config)
        else:
            config = config or PerformerAttentionConfig()

        # kwargs take precedence over the default values that might be stored in the config object
        for k, v in kwargs.items():
            assert hasattr(config, k), f"'{k}' is an invalid config parameter"
            setattr(config, k, v)

        self.__dict__.update(config.__dict__)

        assert self.num_heads and self.d_model, "Num_heads and d_model must be non-None"
        assert self.d_model % self.num_heads == 0, "Num_heads must divide d_model evenly"
        assert self.d_model > self.num_heads, "Number of dimensions per head must be greater than 1"

        self.dropout = nn.Dropout(p=self.attention_dropout)
        self.calls_since_last_redraw = 0

        self.orthogonal_feature_algorithm = resolve_enum(OrthogonalFeatureAlgorithm, self.orthogonal_feature_algorithm)
        if self.orthogonal_feature_algorithm == OrthogonalFeatureAlgorithm.auto:
            self.orthogonal_feature_algorithm = OrthogonalFeatureAlgorithm.kacs

        self.random_feature_chain = None

        # Create the feature matrix up front if we don't need to know what the batch dimension is;
        # otherwise, lazily create it on the first forward pass
        self.random_features = None
        if not self.use_thick_features:
            self._generate_feature_matrix(batch_size=1, device=None)

            # This is needed because apparently DistilBertModel deepcopies its layers on initialization for some
            # reason, which weirdly involves pickling them, and generators can't be pickled. So we'll just burn in
            # another Markov chain on the first redraw if needed.
            self.random_feature_chain = None

        if isinstance(self.kernel_type, Callable):
            self.kernel_fn = self.kernel_type   # Allow for custom kernel types
        else:
            self.kernel_type = resolve_enum(PerformerKernel, self.kernel_type)
            self.kernel_fn = KERNEL_CALLABLES[self.kernel_type]

        #if self.use_linear_layers:
        #    for name in self.linear_layer_names:
        #        setattr(self, name, nn.Linear(self.d_model, self.d_model))

        self.pruned_heads = set()

        if self.causal:
            # Try to load the custom CUDA kernel for fast causal attention if available
            if not self.causal_numerator_fn:
                try:
                    from fast_transformers.causal_product import CausalDotProduct

                    def cuda_causal_numerator(queries, keys_t, values):
                        return CausalDotProduct.apply(queries, keys_t.transpose(-2, -1), values)

                    self.causal_numerator_fn = cuda_causal_numerator
                except ImportError:
                    CausalDotProduct = None

                    logger = logging.getLogger()
                    logger.info("Failed to load custom CUDA kernel for fast causal attention from the fast_transformers"
                                " library. Falling back on a ~2x slower alternative. For the faster algorithm, type"
                                " `pip install pytorch-fast-transformers` on the command line.")

                    self.causal_numerator_fn = _headwise_causal_numerator

            # Recurrent state, taken from 'Transformers are RNNs' Katharopoulos et al. 2020 paper
            if self.use_recurrent_decoding:
                self.s = None   # Numerator
                self.z = None   # Denominator
        else:
            assert not self.use_recurrent_decoding

    def forward(self, query, key, value, mask=None, output_attentions=False, position_bias=None):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, num_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, _, _ = query.shape

        assert not output_attentions, "Can't output attention maps when using Performer attention."
        if self.use_recurrent_decoding:
            assert q_length == 1, "When use_recurrent_decoding == True, we only input and output one token at a time."

        self._redraw_features_if_needed(bs, query.device)

        # Get the transformed values of Q and K
        q_prime, k_prime = self.get_projected_queries_and_keys(query, key)
        return self.compute_attention_with_projected_queries_and_keys(q_prime, k_prime, value, mask, position_bias)

    def get_projected_queries_and_keys(self, q, k):
        """
        Turns Q into Q' and K into K' by multiplying them by the random feature tensor.
        Parameters:
            q: torch.tensor(bs, seq_length, dim)
            k: torch.tensor(bs, seq_length, dim)

        Returns:
            q_prime: torch.tensor(bs, seq_length, num_features)
            k_prime: torch.tensor(bs, seq_length, num_features)
        """
        # Instead of dividing the product QK^T by sqrt(d), we divide Q and K by the 4th root of d.
        q = q / (self.d_model ** 0.25)
        k = k / (self.d_model ** 0.25)

        projected_q = q @ self.random_features
        projected_k = k @ self.random_features

        # Special logic for kernels that attempt to approximate softmax
        if self.kernel_type in (PerformerKernel.cosh, PerformerKernel.exp):
            # The h(x) function is defined in Lemma 1 in Choromanski et al. pg. 4 as exp(-||x||**2 / 2). For numerical
            # stability we leverage the fact that exp(x)*exp(y) = exp(x + y) here and delay computing the exp().
            h_of_q = -torch.sum(q ** 2, dim=-1, keepdim=True) / 2
            h_of_k = -torch.sum(k ** 2, dim=-1, keepdim=True) / 2

            # Compute the numerical stabilizer that we subtract from the input to exp(). For some reason the original
            # Jax implementation uses different types of stabilizers for queries vs. keys, and we follow that here.
            # This is a workaround for very slow performance of torch.max(dim=N) on PyTorch 1.4 and earlier;
            # see this GitHub discussion: https://github.com/pytorch/pytorch/issues/36900
            q_indices = h_of_q.argmax(-1).unsqueeze(-1)
            q_stabilizer = h_of_q.gather(-1, q_indices)  # Note this is a (d_model, 1) matrix that gets broadcasted

            # This is just a scalar
            k_stabilizer = torch.max(h_of_k)

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
            return (self.kernel_fn(x) + self.kernel_epsilon for x in (projected_q, projected_k))

    def compute_attention_with_projected_queries_and_keys(self, q_prime, k_prime, v, mask=None, position_bias=None):
        """
        Computes the attention output given Q' and K' from the above get_projected_queries_and_keys method.
        Parameters:
            q_prime: torch.tensor(bs, seq_length, num_features)
            k_prime: torch.tensor(bs, seq_length, num_features)
            v: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            V': torch.tensor(bs, seq_length, dim)
        """
        # Apply the padding mask to K'. Also applying it to Q' would be redundant.
        if mask is not None:
            k_prime *= mask

        k_prime_t = k_prime.transpose(-2, -1)
        output = self._numerator_for_projected_queries_and_keys(q_prime, k_prime_t, v)

        if position_bias is not None:
            # position_bias: (bs, n_heads, q_length, q_length)
            add_pos = position_bias @ v
            output += add_pos

        if self.normalize_output:
            output /= self._denominator_for_projected_queries_and_keys(q_prime, k_prime_t)

        return output

    def _numerator_for_projected_queries_and_keys(self, q_prime, k_prime_t, v):
        # Noncausal
        if not self.causal:
            return q_prime @ (k_prime_t @ v)

        # Causal, during training
        if not self.use_recurrent_decoding:
            return self.causal_numerator_fn(q_prime, k_prime_t, v)

        # Causal, at inference time- recurrent autoregressive decoding
        s_delta = k_prime_t @ v
        self.s = s_delta if self.s is None else self.s + s_delta

        return q_prime @ self.s

    def _denominator_for_projected_queries_and_keys(self, q_prime, k_prime_t):
        # Noncausal
        if not self.causal:
            denom = q_prime @ k_prime_t.sum(dim=-1, keepdim=True)   # Sum over positions

        # Causal, during training
        elif not self.use_recurrent_decoding:
            prefix_sums = k_prime_t.cumsum(dim=-1)                  # Cumsum over positions
            denom = torch.einsum("bhlm,bhml->bhl", q_prime, prefix_sums)
            denom.unsqueeze_(-1)

        # Causal, at inference time- recurrent autoregressive decoding
        else:
            self.z = k_prime_t if self.z is None else self.z + k_prime_t    # Incrementally sum over positions
            denom = q_prime @ self.z

        # Avoid dividing by very small numbers
        return denom + 2 * self.normalization_stabilizer * (torch.abs(denom) <= self.normalization_stabilizer)

    def redraw_features_now(self):
        """
        Immediately redraws the random features.
        """
        device = self.random_features.device
        batch = self.random_features.shape[0]
        self._generate_feature_matrix(batch, device)

        if self.training and self.redraw_verbose:
            logging.getLogger().info("PerformerAttention: Just redrew random features.")

        self.calls_since_last_redraw = 1

    def reset_recurrent_state(self):
        """
        Resets the recurrent state kept by the model when use_recurrent_decoding == True
        """
        self.s = None
        self.z = None

    def _generate_feature_matrix(self, batch_size, device):
        dim_per_head = self.d_model // self.num_heads
        num_rows = self.num_random_features or round(dim_per_head * math.log(dim_per_head))
        batch = batch_size if self.use_thick_features else 1

        # Just return a random Gaussian matrix
        if not self.use_orthogonal_features:
            output_tensor = torch.randn(batch, num_rows, dim_per_head, device=device)

        # Return an orthogonal matrix
        else:
            # Use a Kac's random walk Markov chain to speed up successive redraws
            if self.orthogonal_feature_algorithm == OrthogonalFeatureAlgorithm.kacs:
                if not self.random_feature_chain:
                    self.random_feature_chain = _get_orthogonal_feature_chain(batch, num_rows, dim_per_head, device)

                output_tensor = next(self.random_feature_chain)

            # Do QR decomposition on random Gaussian blocks
            else:
                total_num_blocks = int(math.ceil(num_rows / dim_per_head))
                extra_rows = total_num_blocks * dim_per_head - num_rows

                blocks = [_get_square_orthogonal_block_qr(batch, dim_per_head, device) for _ in range(total_num_blocks)]
                if extra_rows > 0:
                    blocks[-1] = blocks[-1][:, extra_rows:]

                output_tensor = torch.cat(blocks, dim=1)

            # This option yields SMREG
            if self.regularize_feature_norms:
                output_tensor *= dim_per_head ** 0.5
            else:
                # Hack to make the matrix columns have the norm we would expect them to have if they were sampled
                # straight from a Gaussian, instead of being all norm 1
                multiplier = torch.randn(batch, num_rows, dim_per_head, device=device).norm(dim=-1)
                output_tensor = torch.diag_embed(multiplier) @ output_tensor

        # Add an attention head dimension
        output_tensor.unsqueeze_(1)
        new_shape = list(output_tensor.shape)
        new_shape[0] = batch  # This is redundant if use_thick_features == True, but not otherwise
        new_shape[1] = self.num_heads
        output_tensor = output_tensor.expand(*new_shape).transpose(-2, -1).clone()

        if self.random_features is None:
            self.random_features = torch.nn.Parameter(output_tensor, requires_grad=False)
        else:
            self.random_features.data = output_tensor

    def _redraw_features_if_needed(self, batch, device):
        # We haven't created the projection matrix yet, let's create it
        if self.random_features is None or batch != self.random_features.shape[0]:
            self._generate_feature_matrix(batch, device)

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

# Not currently used
def _get_square_orthogonal_block_qr(batch, size, device=None):
    unstructured_block = torch.randn(batch, size, size, device=device)
    q, r = torch.qr(unstructured_block, some=True)
    return q.transpose(-2, -1)


def _get_orthogonal_feature_chain(batch, num_rows, dim_per_head, device=None):
    # The algorithm requires an even number of rows, so round up to the nearest even number
    rows_per_block = dim_per_head + dim_per_head % 2

    total_num_blocks = int(math.ceil(num_rows / rows_per_block))
    extra_rows = total_num_blocks * rows_per_block - num_rows

    block_chains = [_get_kacs_random_walk_chain(batch, rows_per_block, device) for _ in range(total_num_blocks)]

    while True:
        blocks = [next(chain)[:, :, :dim_per_head] for chain in block_chains]
        if extra_rows > 0:
            blocks[-1] = blocks[-1][:, extra_rows:, :dim_per_head]

        yield torch.cat(blocks, dim=1)


# Not ideal but all we can do until https://github.com/pytorch/pytorch/issues/42502 gets implemented
def _batch_randperm(batch, n, dtype=torch.int64, device=None):
    out_tensor = torch.empty(batch, n, dtype=dtype, device=device)
    for i in range(batch):
        torch.randperm(n, out=out_tensor[i], dtype=dtype, device=device)

    return out_tensor


# Parallelized version of Kac's random walk, a Markov chain that quickly generates random orthogonal matrices. Samples
# are autocorrelated with a mixing time of roughly (2 log n), but this should actually be good for training stability.
# Each sample is generated in O(n^2) time, after 2 log n burn-in steps.
@torch.no_grad()
def _get_kacs_random_walk_chain(batch, num_rows, device=None):
    # Start with identity matrix
    block = torch.eye(num_rows, device=device)
    block = block.expand(batch, num_rows, num_rows)
    block.unsqueeze_(2)

    burnin_steps = 2 * int(math.ceil(math.log(num_rows)))
    for n in count():
        # Compute the cosines and sines of the rotations up front
        angles = math.pi * torch.rand(batch, num_rows // 2, 1, device=device)
        cosines, sines = torch.cos(angles), torch.sin(angles)

        # Group the matrix into random, non-overlapping pairs of rows. Because these pairs are non-overlapping, we can
        # perform each set of rotations in parallel.
        shuffled_rows = _batch_randperm(batch, num_rows, device=device).view(batch, -1, 1, 1)
        random_row_pairs = block.gather(1, shuffled_rows.expand_as(block)).view(batch, -1, 2, num_rows)

        rows1, rows2 = random_row_pairs[:, :, 0], random_row_pairs[:, :, 1]
        new_rows1 = cosines * rows1 + sines * rows2
        new_rows2 = -sines * rows1 + cosines * rows2

        random_row_pairs[:, :, 0], random_row_pairs[:, :, 1] = new_rows1, new_rows2

        block = random_row_pairs.view(batch, -1, 1, num_rows)  # Ungroup all the rows again

        # Only yield the block after we've completed 2 log(n) burn-in iterations
        if n >= burnin_steps:
            # Account for accumulated numerical error- the norm tends to drift upward
            if n % 1000 == 0:
                block /= block.norm(dim=-1, keepdim=True)

            yield block.squeeze(-2)


def _headwise_causal_numerator(q_prime, k_prime_t, v):
    results = []

    # Iterate over the attention heads to avoid allocating a very large tensor
    for head in range(q_prime.shape[1]):
        # Outer products- a sorta biggish tensor
        outer_prods = torch.einsum('bml,bld->blmd', k_prime_t[:, head], v[:, head])
        prefix_sums = outer_prods.cumsum(dim=1)

        query_prods = torch.einsum('blmd,blm->bld', prefix_sums, q_prime[:, head])
        results.append(query_prods.unsqueeze(1))

    return torch.cat(results, dim=1)
