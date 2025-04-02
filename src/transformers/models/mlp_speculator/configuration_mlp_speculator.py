from typing import List

from ...configuration_utils import PretrainedConfig



class MLPSpeculatorConfig(PretrainedConfig):
    model_type = "mlp_speculator"

    def __init__(
        self,
        vocab_size: int = 32000,
        emb_dim: int = 4096,
        inner_dim: int = 0,
        n_predict: int = 3,
        top_k_tokens_per_head: List[int] = [5, 4, 3],
        n_candidates: int = 5,
        tie_weights: bool = False,
        scale_input: bool = False,
        **kwargs
    ):
        """
        Initialize an MLPSpeculatorConfig

        Args:
            vocab_size: int
                the model vocab size
            emb_dim: int
                the model embedding dimension
            inner_dim: int
                the inner dimension of the model. If 0, will be the emb_dim.
            n_predict: int
                the number of lookaheads for the speculator
            top_k_tokens_per_head: List[int]
                Number of tokens to consider from each head when forming the candidate tree.
                For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
            n_candidates: int
                number of child candidates to create per sequence
            tie_weights : bool
                If true, use a single set of weights for every model head/stage after the first.
                The initial projection from the base model may have a different size, so that stays separate.
            scale_input: bool
                If true, apply an extra layernorm to the initial state vector input.
                Helps training dynamics, particularly when base model output has unusual scale.
        """
        assert len(top_k_tokens_per_head) == n_predict
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.n_predict = n_predict
        self.top_k_tokens_per_head = top_k_tokens_per_head
        self.n_candidates = n_candidates
        self.tie_weights = tie_weights
        self.scale_input = scale_input
        super().__init__(**kwargs)
