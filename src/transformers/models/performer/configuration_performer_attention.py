from dataclasses import dataclass
from typing import Callable, Sequence, Optional, Union
from enum import Enum


PerformerKernel = Enum('PerformerKernel', ['cosh', 'exp', 'elu', 'relu'])
OrthogonalFeatureAlgorithm = Enum('OrthogonalFeatureAlgorithm', ['auto', 'kacs', 'qr'])


class PerformerAttentionConfig():
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.PerformerAttention` module.
    It is used to define the behavior of a Performer/FAVOR+ attention module when it is initialized.
    
    Args:
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        causal (:obj:`bool`, `optional`, defaults to False):
            Whether to apply causal attention, where positions are prevented from attending to positions to ahead
            of themselves in the sequence, using the prefix-sum method.
        kernel_type (:obj:`Enum(PerformerKernel)`, `optional`, defaults to :obj:`'exp'`):
            The type of kernel function to use for comparing the queries and keys. Possible options are :obj:`'exp'`,
            :obj:`'cosh'`, and :obj:`'relu'`. The :obj:`'cosh'` option approximates softmax attention with a smaller
            variance than :obj:`'exp'`, but at the cost of using twice as many random features. :obj:`'relu'` may result
            in better performance than :obj:`'exp'` and :obj:`'cosh'` in certain circumstances, but it is not an
            unbiased estimator of softmax attention and thus should not be used with pretrained models that were
            pretrained with softmax attention.
        kernel_epsilon (:obj:`float`, `optional`, defaults to 1e-4):
            Stabilizer term added to the output of the kernel function to avoid dividing by very small numbers.
        normalize_output (:obj:`bool`, `optional`, defaults to True):
            Whether to ensure that the output vectors are convex combinations of the input vectors; that is, that the
            rows of the implicit attention map sum to 1.
        normalization_stabilizer (:obj:`float`, `optional`, defaults to 1e-6):
            Stabilizer term used when normalizing the output to avoid dividing by very small numbers.
        num_random_features (:obj:`int`, `optional`, defaults to None):
            The dimensionality of the random feature vectors to use. When None, the dimensionality is set to
            D * log(D), where D is the dimensionality of each attention head.
        orthogonal_feature_algorithm (:obj:`Enum(OrthogonalFeatureAlgorithm)`, defaults to 'auto'):
            The algorithm to use for generating random orthogonal features. Possible values are 'kacs', which uses a
            Kac's random walk Markov chain; 'qr', which performs QR decomposition on a random Gaussian matrix at each
            redraw; and 'auto', which is equivalent to 'kacs' on PyTorch and 'qr' on TensorFlow, since the Kac's random
            walk algorithm is not supported on TensorFlow. Kac's is generally faster than QR, but successive samples
            are correlated with each other.
        use_recurrent_decoding (:obj:`bool`, `optional`, defaults to False):
            Whether to use recurrent autoregressive decoding, as described in the 'Transformers are RNNs' paper. If
            True, the PerformerAttention object will expect input tensors with a sequence length dimension of exactly 1,
            and will output tensors with sequence length of 1. It will retain a recurrent hidden state between forward
            passes that can be reset with the reset_recurrent_state() method.
        use_thick_features (:obj:`bool`, `optional`, defaults to False):
            Whether to generate a random feature tensor that has a batch dimension.
        use_orthogonal_features (:obj:`bool`, `optional`, defaults to True):
            Whether to use strictly orthogonal random features, as opposed to features drawn from a standard Gaussian
            distribution. Orthogonal features result in outputs that more closely approximate softmax attention, but at
            the cost of doing QR decomposition on the CPU every time the features are redrawn. Best combined with a
            reasonably large value of :obj:`feature_redraw_interval` (1-5k).
        use_linear_layers (:obj:`bool`, `optional`, defaults to True):
            Whether to transform the Q, K, and V inputs with a Linear layer before applying attention. Setting this
            to False may be useful if you want to use PerformerAttention as one component of a more complex
            attention mechanism.
        regularize_feature_norms (:obj:`bool`, `optional`, defaults to False):
            Whether to ensure that the random feature vectors have a norm of sqrt(`d`), where `d` is the dimensionality
            of each attention head.
        feature_redraw_interval (:obj:`int`, `optional`, defaults to 100):
            The number of forward passes after which the random feature matrix should be redrawn. If None, then the
            feature matrix is never redrawn. When combined with :obj:`redraw_stochastically`, this parameter determines
            the expected value of the redraw interval, rather than the interval itself.
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

    def __init__(
        self,
        attention_dropout = 0.1,
        kernel_type = PerformerKernel.exp,
        causal = False,
        use_recurrent_decoding = False,
        kernel_epsilon = 1e-4,
        normalize_output = True,
        normalization_stabilizer: float = 1e-6,
        use_linear_layers = True,
        linear_layer_names = ('q_linear', 'k_linear', 'v_linear', 'out_linear'),
        num_random_features = None,
        use_thick_features = False,
        regularize_feature_norms = True,
        use_orthogonal_features = True,
        orthogonal_feature_algorithm = OrthogonalFeatureAlgorithm.auto,
        feature_redraw_interval = 100,
        redraw_stochastically = False,
        redraw_verbose = False,
        d_model = None,
        num_heads = None,
        **kwargs
    ):

        self.attention_dropout = attention_dropout
        self.kernel_type: Union[str, Callable, PerformerKernel] = PerformerKernel.exp

        self.causal = causal
        self.use_recurrent_decoding = use_recurrent_decoding

        self.kernel_epsilon = kernel_epsilon
        self.normalize_output = kernel_epsilon
        self.normalization_stabilizer = normalization_stabilizer

        # The linear_layer_names parameter is needed to allow the PerformerAttention object to imitate the naming
        # convention of arbitrary attention modules, and therefore load weights from pretrained models. It can either have
        # 3 or 4 elements; if it has 3, then no output linear layer is used.
        self.use_linear_layers = use_linear_layers
        self.linear_layer_names = linear_layer_names

        self.num_random_features = num_random_features
        self.use_thick_features = use_thick_features
        self.regularize_feature_norms = regularize_feature_norms

        self.use_orthogonal_features = use_orthogonal_features
        self.orthogonal_feature_algorithm = orthogonal_feature_algorithm
        self.feature_redraw_interval = feature_redraw_interval
        self.redraw_stochastically = redraw_stochastically
        self.redraw_verbose = redraw_verbose

        # Optional here so the user doesn't have to set redundant parameters, but must be set by model before config is
        # passed to PerformerAttention.__init__()
        self.d_model = d_model
        self.num_heads = num_heads

    # Make enums JSON serializable
    def to_dict(self):
        return {k: v.name if isinstance(v, Enum) else v for k, v in self.__dict__.items()}
