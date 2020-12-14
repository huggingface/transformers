from dataclasses import dataclass
from typing import Callable, Optional, Union

@dataclass
class PerformerAttentionConfig:
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.PerformerAttention` module.
    It is used to define the behavior of a Performer/FAVOR+ attention module when it is initialized.
    
    Args:
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        causal (:obj:`bool`, `optional`, defaults to False):
            Whether to apply causal attention, where positions are prevented from attending to positions to ahead
            of themselves in the sequence, using the prefix-sum method.
        kernel_type (:obj:`str`,  `optional`, defaults to :obj:`'exp'`):
            The type of kernel function to use for comparing the queries and keys. Possible options are :obj:`'exp'`,
            :obj:`'cosh'`, and :obj:`'relu'`. The :obj:`'cosh'` option approximates softmax attention with a smaller
            variance than :obj:`'exp'`, but at the cost of using twice as many random features. :obj:`'relu'` may result
            in better performance than :obj:`'exp'` and :obj:`'cosh'` in certain circumstances, but it is not an
            unbiased estimator of softmax attention and thus should not be used with pretrained models that were
            pretrained with softmax attention.
        short_sequence_behavior (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`'use_softmax_eval_only'`):
            This parameter determines if and when the module should fall back to regular softmax attention. Softmax
            attention is generally faster than FAVOR+ when the sequence length is not significantly larger than the
            number of random features used— which is equal to round(d*log(d)), where d is the number of dimensions per
            attention head. The default behavior is to use FAVOR+ regardless of sequence length while training, but to
            use softmax attention at test time when the sequence length is less than twice the number of random
            features.
            Possible values for this parameter are :obj:`'use_softmax_eval_only'`, :obj:`'use_softmax_eval_and_train'`,
            :obj:`'never_use_softmax'`. The option :obj:`'use_softmax_eval_and_train'` should probably only be used if
            the training set has a significant number of long sequences; otherwise, the model may not learn to deal with
            the random noise inherent in the FAVOR+ algorithm.
        kernel_epsilon (:obj:`float`, `optional`, defaults to 1e-4):
            Stabilizer term added to the output of the kernel function to avoid dividing by very small numbers.
        normalize_output (:obj:`bool`, `optional`, defaults to True):
            Whether to ensure that the output vectors are convex combinations of the input vectors; that is, that the
            rows of the implicit attention map sum to 1.
        normalization_stabilizer (:obj:`float`, `optional`, defaults to 1e-6):
            Stabilizer term used when normalizing the output to avoid dividing by very small numbers.
        use_orthogonal_features (:obj:`bool`, `optional`, defaults to True):
            Whether to use strictly orthogonal random features, as opposed to features drawn from a standard Gaussian
            distribution. Orthogonal features result in outputs that more closely approximate softmax attention, but at
            the cost of doing QR decomposition on the CPU every time the features are redrawn. Best combined with a
            reasonably large value of :obj:`feature_redraw_interval` (1-5k).
        use_qkv_linear_layers (:obj:`bool`, `optional`, defaults to True):
            Whether to transform the Q, K, and V inputs with a Linear layer before applying attention. Setting this
            to False may be useful if you want to use PerformerAttention as one component of a more complex
            attention mechanism.
        regularize_feature_norms (:obj:`bool`, `optional`, defaults to False):
            Whether to ensure that the random feature vectors have a norm of sqrt(`d`), where `d` is the dimensionality
            of each attention head.
        feature_redraw_interval (:obj:`int`, `optional`, defaults to 1000):
            The number of forward passes after which the random feature matrix should be redrawn. If None, then the
            feature matrix is never redrawn. It is recommended to set this property to some value on the order of 1-5k
            while training in order to get the best model performance. When combined with :obj:`redraw_stochastically`,
            this parameter determines the expected value of the redraw interval, rather than the interval itself.
        redraw_stochastically (:obj:`bool`, `optional`, defaults to False):
            If true, PerformerAttention will redraw its random features each forward pass with a probability equal to
            (1 / :obj:`feature_redraw_interval`), instead of deterministically redrawing once every N passes. This could
            be desirable in large models to ensure that the attention layers don't all redraw their features at the same
            time.
        redraw_verbose (:obj:`bool`, `optional`, defaults to False):
            Whether to log a message when random features are redrawn during training.
        dim (:obj:`int`, `optional`):
            Dimensionality of the queries, keys, and values.
        num_heads (:obj:`int`, `optional`):
            Number of attention heads.
    """
    
    attention_dropout: float = 0.1
    causal: bool = False
    kernel_type: str = 'exp'
    
    # Default determined in PerformerAttention.__init__()
    short_sequence_behavior: Optional[Union[str, Callable]] = None
    
    kernel_epsilon: float = 1e-4
    normalize_output: bool = True
    normalization_stabilizer: float = 1e-6

    use_orthogonal_features: bool = True
    use_qkv_linear_layers: bool = True
    regularize_feature_norms: bool = True
    
    feature_redraw_interval: int = 1000
    redraw_stochastically: bool = False
    redraw_verbose: bool = False
    
    # Optional here so the user doesn't have to set redundant parameters, but must be set by model before config is
    # passed to PerformerAttention.__init__()
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
