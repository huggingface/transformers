from torch import nn
from typing import Callable, Optional, Union
import numpy as np

from .modeling_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer
)

KERNEL_CALLABLES = {
    'exp': lambda x, h, stabilizer: torch.exp(h + x - stabilizer),
    'cosh': lambda x, h, stabilizer: torch.cat((torch.exp(h + x - stabilizer), torch.exp(h - x - stabilizer)), dim=-1),
    'relu': lambda x, h, stabilizer: nn.ReLU()(x) + 1e-9 # Adding epsilon prevents dividing by zero when we normalize
}

SHORT_SEQUENCE_BEHAVIOR_CALLABLES = {
    'use_softmax_eval_only': lambda L, M, training: False if training else L < 2.0 * M,
    'use_softmax_eval_and_train': lambda L, M, training: L < 2.0 * M, 
    'never_use_softmax': lambda L, M, training: False
}

class PerformerAttentionConfig(object):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.PerformerAttention` module. It is used
    to define the behavior of a Performer/FAVOR+ attention module when it is initialized.
    
    Args:
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        feature_redraw_interval (:obj:`int`, `optional`, defaults to 1000):
            The number of forward passes after which the random feature matrix should be redrawn. If None, then the feature
            matrix is never redrawn. It is recommended to set this property to some value on the order of 1-5k while
            training in order to get the best model performance.
        kernel_type (:obj:`str`,  `optional`, defaults to :obj:`'exp'`):
            The type of kernel function to use for comparing the queries and keys. Possible options are :obj:`'exp'`,
            :obj:`'cosh'`, and :obj:`'relu'`. The :obj:`'cosh' option approximates softmax attention with a smaller
            variance than :obj:`'exp'`, but at the cost of using twice as many random features. :obj:`'relu'` may result
            in better performance than :obj:`'exp'` and :obj:`'cosh'` in certain circumstances, but it is not an unbiased
            estimator of softmax attention and thus should not be used with pretrained models that were pretrained with
            softmax attention.
        short_sequence_behavior (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`'use_softmax_eval_only'`):
            This parameter determines if and when the module should fall back to regular softmax attention. For relatively
            short sequences, softmax attention is often faster than FAVOR+. The default behavior is to always use FAVOR+
            regardless of sequence length while training, but to use softmax attention at test time when the sequence length
            is less than twice the number of random features FAVOR+ would use. Possible values are
            :obj:`'use_softmax_eval_only'`, :obj:`'use_softmax_eval_and_train'`, :obj:`'never_use_softmax'`. The option
            :obj:`'use_softmax_eval_and_train'` should probably only be used if the training set has a significant number of
            long sequences; otherwise, the model may not learn to deal with the random noise inherent in the FAVOR+
            algorithm. If a :obj:`Callable`, should take three arguments, :obj:`L`, :obj:`M`, and :obj:`training`, where
            :obj:`L` is the sequence length, :obj:`M` is the number of random features that FAVOR+ plans to use, and
            :obj:`training` is whether the module is in training mode, and return a :obj:`bool` which is True if softmax
            attention should be used, and False if FAVOR+ should be used.
        use_orthogonal_features (:obj:`bool`, `optional`, defaults to True):
            Whether to use strictly orthogonal random features, as opposed to features drawn from a standard Gaussian
            distribution. Orthogonal features result in outputs that more closely approximate softmax attention, but at
            the cost of doing QR decomposition on the CPU every time the features are redrawn. Best combined with a
            reasonably large value of :obj:`feature_redraw_interval` (1-5k).
        regularize_feature_norms (:obj:`bool`, `optional`, defaults to True):
            Whether to ensure that the random feature vectors have a norm of sqrt(`d`), where `d` is the dimensionality of
            each attention head.
        dim (:obj:`int`, `optional`):
            Dimensionality of the queries, keys, and values.
        n_heads (:obj:`int`, `optional`):
            Number of attention heads.
    """
    
    def __init__(
        self,
        attention_dropout: float = 0.1,
        feature_redraw_interval: int = 1000,
        kernel_type: str = 'exp',
        use_orthogonal_features: bool = True,
        regularize_feature_norms: bool = True,
        
        # Default determined in PerformerAttention.__init__()
        short_sequence_behavior: Optional[Union[str, Callable]] = None,
        
        # Optional here so the user doesn't have to set redundant parameters, but must be set by model before config is
        # passed to PerformerAttention.__init__()
        dim: Optional[int] = None,
        n_heads: Optional[int] = None
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.feature_redraw_interval = feature_redraw_interval
        self.kernel_type = kernel_type
        self.short_sequence_behavior = short_sequence_behavior
        self.use_orthogonal_features = use_orthogonal_features
        self.regularize_feature_norms = regularize_feature_norms
    
    def __repr__(self):
        return str(self.__dict__)


class PerformerAttention(nn.Module):
    def __init__(self, config: Optional[Union[dict, PerformerAttentionConfig]] = None, **kwargs):
        super().__init__()
        
        # Three possibilities: config == None, config is a dict, or config is a PerformerAttentionConfig object
        if not config:
            config = PerformerAttentionConfig(**kwargs)
        elif isinstance(config, dict):
            config = PerformerAttentionConfig(**config)
        else:
            for key, value in kwargs.items():
                setattr(config, key, value)
        
        try:
            self.n_heads = config.n_heads
            self.dim = config.dim
        except:
            raise ValueError("PerformerAttention: config.n_heads and config.dim must be non-None")
        
        self.dropout = nn.Dropout(p=config.attention_dropout)
        
        self.feature_redraw_interval = config.feature_redraw_interval
        self.use_orthogonal_features = config.use_orthogonal_features
        self.regularize_feature_norms = config.regularize_feature_norms
        self.register_buffer('calls_since_last_redraw', torch.tensor(0)) # Should be persistent
        
        behavior = config.short_sequence_behavior
        if not behavior:
            behavior = 'never_use_softmax' if config.kernel_type == 'relu' else 'use_softmax_eval_only'
            self.should_fallback_to_softmax = SHORT_SEQUENCE_BEHAVIOR_CALLABLES[behavior]
        
        elif config.kernel_type == 'relu' and behavior != 'never_use_softmax':
            raise ValueError(f"PerformerAttention: short_sequence_behavior = {behavior} cannot be combined with the relu "
                             "kernel type")
        
        elif isinstance(behavior, str):
            self.should_fallback_to_softmax = SHORT_SEQUENCE_BEHAVIOR_CALLABLES[behavior]
        elif callable(behavior):
            self.should_fallback_to_softmax = behavior
        else:
            raise ValueError("PerformerAttention: short_sequence_behavior must be either str or Callable")
        
        self.kernel_fn = KERNEL_CALLABLES[config.kernel_type]

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def redraw_features_now(self):
        device = self.random_features.device
        self._generate_feature_matrix(device)
        
        self.calls_since_last_redraw = torch.tensor(0)

    def forward(self, query, key, value, mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        
        # Instead of dividing the product QK^T by sqrt(d), we divide Q and K separately by the 4th root of d
        q = q / (dim ** 0.25)
        k = k / (dim ** 0.25)
        
        # If the sequence length is short enough that FAVOR+ would use considerably more time and/or memory than just
        # using softmax attention, use softmax. This works because FAVOR+ is an unbiased estimator of softmax attention.
        m = round(dim_per_head * np.log(dim_per_head)) # m is the number of random features
        if self.should_fallback_to_softmax(q_length, m):
            scores = q @ k.transpose(-2, -1)
            
            if mask is not None:
                mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
                scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)
            
            attn_map = nn.Softmax(dim=-1)(scores)
            attn_map = self.dropout(attn_map)  # (bs, n_heads, q_length, k_length)
            return self._finalize_attention_output(attn_map @ v, head_mask, output_attentions)
        
        # When we're using FAVOR+ we can't output the attention matrix
        if output_attentions:
            raise ValueError("PerformerAttention: Can't output attention maps when using FAVOR+ linear attention.")
        
        # We haven't created the projection matrix yet, let's create it
        if not hasattr(self, 'random_features'):
            self._generate_feature_matrix(q.device)
        
        # It's time to redraw the projection matrix
        elif exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            self.redraw_features_now()
        
        # Keep track of how many forward passes we do before we redraw again
        else:
            self.calls_since_last_redraw += 1
        
        # Broadcast the feature matrix across the batch dimension
        self.random_features.expand(q.shape[0], num_heads, m, dim_per_head)
        W_t = self.random_features.transpose(-2, -1)
        
        # By multiplying Q' and K' by 1/sqrt(m), we ensure the final matrix product will contain a factor of 1/m. This means
        # that each row of Q'K'^T can be interpreted as an average over the exp(omega^T * q) * exp(omega^T * k) terms.
        epsilon = 1e-4
        def phi(x, is_query):
            # The h(x) function, defined in Lemma 1 in Choromanski et al. pg. 4
            h_of_x = -torch.sum(x ** 2, dim=-1, keepdim=True) / 2
            
            projected_x = x @ W_t
            stabilizer = torch.max(h_of_x) if not is_query else torch.max(h_of_x, axis=-1, keepdim=True).values
            kernel_output = kernel_fn(projected_x, h_of_x, stabilizer)
            
            return (kernel_output.shape[-1] ** -0.5) * (kernel_output + epsilon)
        
        # Get the transformed values of Q and K
        q_prime, k_prime = phi(q, True), phi(k, False)
        
        # Now apply the padding mask to K'. Also applying it to Q' would be redundant.
        if mask is not None:
            k_prime *= mask.view(mask_reshp).expand_as(k_prime)
        
        # Equivalent to multiplying K'^T by a ones vector
        d = q_prime @ k_prime.sum(dim=-2).unsqueeze(-1)
        d += 2 * epsilon * (torch.abs(d) <= epsilon) # Avoid dividing by very small numbers
        
        k_prime_t = k_prime.transpose(-2, -1)
        context = q_prime @ (k_prime_t @ v) / d
        
        return self._finalize_attention_output(context, head_mask)
    
    def _finalize_attention_output(self, context, head_mask=None, output_attentions=False):
        # Mask heads if we want to
        if head_mask is not None:
            context = context * head_mask
        
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)

    def _generate_feature_matrix(self, device):
        dim_per_head = self.dim // self.n_heads
        num_rows = round(dim_per_head * np.log(dim_per_head))
        
        if not self.use_orthogonal_features:
            return torch.randn(num_rows, dim_per_head, device=device)
        
        def get_square_block(size):
            unstructured_block = torch.randn(size, size, device='cpu')
            q, r = torch.qr(unstructured_block, some = True)
            return q.t()
        
        num_full_blocks = num_rows // dim_per_head
        block_list = [get_square_block(dim_per_head) for _ in range(num_full_blocks)]
        
        remaining_rows = num_rows - num_full_blocks * dim_per_head
        if remaining_rows > 0:
            q = get_square_block(dim_per_head)
            block_list.append(q[:remaining_rows])
        
        final_matrix = torch.cat(block_list)
        
        # This option yields SMREG
        if self.regularize_feature_norms:
            final_matrix *= dim_per_head ** 0.5
        else:
            # Hack to make the matrix columns have the norm we would expect them to have if they were sampled straight
            # from a Gaussian, instead of being all norm 1 since they went through QR decomposition
            multiplier = torch.randn(num_rows, dim_per_head, device='cpu').norm(dim = 1)
            final_matrix = torch.diag(multiplier) @ final_matrix
        
        random_features = final_matrix.to(device)
        
        # Make sure this is persistent so that, if someone saves a model while training in between feature redraws and then
        # later loads it and continues training, the training will behave the same as if they hadn't paused training
        self.register_buffer('random_features', random_features)