import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import PreTrainedModel
from .configuration_mlp_speculator import MLPSpeculatorConfig


class MLPSpeculatorLayerNorm(nn.Module):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value
         fits in the range of your encoding scheme
         (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale_and_shift : bool
        Include a learned scaling and shift term after normalization.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale_and_shift=True,
    ):
        super().__init__()
        self.elementwise_scale_and_shift = elementwise_scale_and_shift
        if self.elementwise_scale_and_shift:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale_and_shift:
            x = self.weight * x
            x = x + self.bias
        return x


class MLPSpeculator(nn.Module):
    """
    This is a simple MLP-based speculator that functions similarly to Medusa
    (https://arxiv.org/abs/2401.10774), ingesting context via the final embedding
    vector from the base model. However, this model also conditions on previously
    predicted tokens, similarly to an RNN, allowing it to generate better-quality n-grams.

    The architecture is as flat and simple as possible: for each prediction head,
    the current state vector is projected into a new latent space and added to the
    previous token's embedding. This sum goes through layernorm and activation, forming
    the new state vector. This state predicts the next token (or set of candidate tokens)
    for the current head, and then is passed on to the next.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of the input vector from the base model.
    inner_dim : int
        Latent dimensionality of the speculator model.
    vocab_size : int
        Number of entries in the tokenizer associated with the base model.
    n_predict : int
        Number of heads / number of tokens to guess ahead. Model size and speed scale with this value.
    tie_weights : bool
        If true, use a single set of weights for every model head/stage after the first.
        The initial projection from the base model may have a different size, so that stays separate.
    scale_input: bool
        If true, apply an extra layernorm to the initial state vector input.
        Helps training dynamics, particularly when base model output has unusual scale.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_predict = config.n_predict
        self.emb_dim = config.emb_dim
        self.inner_dim = config.inner_dim if config.inner_dim != 0 else config.emb_dim
        self.vocab_size = config.vocab_size
        self.tie_weights = config.tie_weights
        self.scale_input = config.scale_input

        self.emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.inner_dim) for _ in range(self.n_predict)])
        self.proj = nn.ModuleList(
            [
                nn.Linear((self.emb_dim if i == 0 else self.inner_dim), self.inner_dim, bias=False)
                for i in range(self.n_predict)
            ]
        )
        self.head = nn.ModuleList(
            [nn.Linear(self.inner_dim, self.vocab_size, bias=False) for _ in range(self.n_predict)]
        )
        self.ln = nn.ModuleList(
            [MLPSpeculatorLayerNorm(self.inner_dim, elementwise_scale_and_shift=True) for _ in range(self.n_predict)]
        )
        if self.scale_input:
            self.ln0 = MLPSpeculatorLayerNorm(self.emb_dim, elementwise_scale_and_shift=False)
        # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
        self.state_weight = 0.5 ** (0.5 / self.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.inner_dim / 2))
        self.activation = nn.GELU()

        # Handle weight tying as specified
        if self.tie_weights:
            assert self.n_predict > 1, "You cannot tie weights between stages when only 1 exists"
            for emb in self.emb:
                emb.weight = self.emb[0].weight

            for head in self.head:
                head.weight = self.head[0].weight

            for ln in self.ln:
                ln.weight = self.ln[0].weight
                ln.bias = self.ln[0].bias

            # Since first proj has different size, allow different initial proj from base into model
            for i in range(2, self.n_predict):
                self.proj[i].weight = self.proj[1].weight

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1 / math.sqrt(self.inner_dim))
            elif isinstance(m, MLPSpeculatorLayerNorm) and hasattr(m, "weight"):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def generate_suffixes(
        self,
        state: torch.Tensor,
        ind: torch.Tensor,
        topk: List[int] = [5, 4, 3],
        n: int = 5,
    ) -> torch.Tensor:
        """
        FOR INFERENCE
        Generate tree of candidate sequences.
        ...
        Args
        ----
        state : torch.Tensor
            Most recent embedding vector from the base model (pre-classification head).
            Expects size [b 1 d] where b is batch size and d is model width.
        ind : torch.Tensor
            Token indices of the base model's most recent predicted token(s).
            Expects size [b 1] where b is batch size.
        topk : List(int)
            Number of tokens to consider from each head when forming the candidate tree.
            For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
        n : int
            Given the final tree of prod(topk) candidates, return only the top n most confident.
        ...
        Output : torch.Tensor
            The tensor of most likely candidate sequences.
            Has size [b n self.n_predict], where b is batch size and n is provided above.
        """
        # k indicates # of candidates
        # h indicates # of generated tokens
        b = state.size(0)
        k = math.prod(topk)
        out = torch.empty(b, 1, k, self.n_predict, device=state.device).int()  # b 1 k h -> b k 1 h
        log_probs = torch.zeros(b, 1, k, device=state.device)  # b 1 k -> b k 1
        assert len(topk) == self.n_predict, (
            f"You must provide a topk number for each head ({self.n_predict} heads, {len(topk)} provided)"
        )
        if self.scale_input:
            state = self.ln0(state) / (2**0.5)
        for i in range(self.n_predict):
            # Project and predict
            z = self.emb[i](ind)  # b k d
            state = self.proj[i](state)
            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
            state = self.activation(self.ln[i](state))  # b k d
            probs = F.log_softmax(self.head[i](state), dim=2)  # b k v
            probs, preds = probs.topk(topk[i], dim=2)  # b k k'

            # Update candidate set with new predictions, repeating shared prefixes as needed
            out = out.view(b, preds.size(1) * preds.size(2), -1, self.n_predict)
            out[:, :, :, i] = preds.view(b, -1, 1)

            # Update state, log_probs and ind for new predictions
            state = state.unsqueeze(2).expand(-1, -1, topk[i], -1)  # b k k' d
            state = state.reshape(b, -1, state.size(3))  # b kk' d
            ind = preds.view(b, -1)  # b kk'
            log_probs = log_probs.view(b, probs.size(1) * probs.size(2), -1)
            log_probs = log_probs.add(probs.view(b, -1, 1))

        # Take only top n best guesses
        out = out.view(b, k, self.n_predict)
        log_probs = log_probs.view(b, k)
        best_guesses = log_probs.topk(n, dim=1)[1]  # b k
        return out.gather(1, best_guesses.unsqueeze(2).expand(-1, -1, self.n_predict))  # b n h

    def forward(
        self,
        state: torch.Tensor,
        inds: torch.Tensor,
    ) -> torch.Tensor:
        """
        FOR TRAINING
        A parallel forward pass on pre-existing ground-truth tokens in pretraining contexts.
        Produces self.n_predict predicted tokens for each token embedding in state.
        Inds requires self.n_predict extra tokens on the right to "simulate" recursive
        behavior for end positions.
        ...
        Args
        ----
        state : torch.Tensor
            Embedding vectors from the base model for a given sequence.
            Expects size [b n d] where b is batch size, n is seq len, and d is model width.
        inds : torch.Tensor
            Ground-truth token indices. inds[:,i] is the prediction coming from state[:,i]
            (or the legal fiction ground truth corresponding to that prediction).
            Expects size [b n+self.n_predict].
        ...
        Output : torch.Tensor
            Prediction logits at each position, for each head of the speculator.
            Has size [self.n_predict b n v] where v is vocab size.
        """
        out = []
        if self.scale_input:
            state = self.ln0(state) / (2**0.5)
        for i in range(self.n_predict):
            z = self.emb[i](inds[:, i : i + state.size(1)])  # b n d
            state = self.proj[i](state)
            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)
            state = self.activation(self.ln[i](state))  # b n d
            out.append(self.head[i](state))  # b n v
        return torch.stack(out, dim=0)  # h b n v


class MLPSpeculatorPreTrainedModel(PreTrainedModel):
    """
    Huggingface MLPSpeculator which provides loading/saving in huggingface
    """

    config_class = MLPSpeculatorConfig
    _no_split_modules = ["MLPSpeculator"]

    def __init__(self, config: MLPSpeculatorConfig, speculator: Optional[MLPSpeculator] = None):
        super().__init__(config)
        if speculator is None:
            self.speculator = MLPSpeculator(config)
            self.speculator.reset_parameters()
        else:
            self.speculator = speculator

    def generate_suffixes(
        self,
        state: torch.Tensor,
        ind: torch.Tensor,
        topk: List[int] = [5, 4, 3],
        n_candidates: int = 5,
    ) -> torch.Tensor:
        """
        FOR INFERENCE
        Generate tree of candidate sequences.
        ...
        Args
        ----
        state : torch.Tensor
            Most recent embedding vector from the base model (pre-classification head).
            Expects size [b 1 d] where b is batch size and d is model width.
        ind : torch.Tensor
            Token indices of the base model's most recent predicted token(s).
            Expects size [b 1] where b is batch size.
        topk : List(int)
            Number of tokens to consider from each head when forming the candidate tree.
            For each candidate branch in the tree, head n produces topk[n] additional sub-branches.
        n_candidates : int
            Given the final tree of prod(topk) candidates, return only the top n most confident.
        ...
        Output : torch.Tensor
            The tensor of most likely candidate sequences.
            Has size [b n self.n_predict], where b is batch size and n is provided above.
        """
        return self.speculator.generate_suffixes(state, ind, topk, n_candidates)

    def forward(
        self,
        state: torch.Tensor,
        inds: torch.Tensor,
    ) -> torch.Tensor:
        """
        FOR TRAINING
        A parallel forward pass on pre-existing ground-truth tokens in pretraining contexts.
        Produces self.n_predict predicted tokens for each token embedding in state.
        Inds requires self.n_predict extra tokens on the right to "simulate" recursive
        behavior for end positions.
        ...
        Args
        ----
        state : torch.Tensor
            Embedding vectors from the base model for a given sequence.
            Expects size [b n d] where b is batch size, n is seq len, and d is model width.
        inds : torch.Tensor
            Ground-truth token indices. inds[:,i] is the prediction coming from state[:,i]
            (or the legal fiction ground truth corresponding to that prediction).
            Expects size [b n+self.n_predict].
        ...
        Output : torch.Tensor
            Prediction logits at each position, for each head of the speculator.
            Has size [self.n_predict b n v] where v is vocab size.
        """
        return self.speculator(state, inds)

    def reset_parameters(self):
        self.speculator.reset_parameters()
