import torch
from ...configuration_utils import PretrainedConfig


ACT2FN = {
    "relu": torch.nn.functional.relu,
    "sigmoid": torch.nn.functional.sigmoid,
    "tanh": torch.nn.functional.tanh,
    "leaky_relu": torch.nn.functional.leaky_relu,
    "elu": torch.nn.functional.elu,
    "selu": torch.nn.functional.selu,
    "gelu": torch.nn.functional.gelu
}


class SigmaMoEConfiguration(PretrainedConfig):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        expert_size: int,
        top_k_experts: int,
        dropout: float,
        selection_mode: str,
        activation_after_topk: bool,
        activation: str,
        bias: bool,
        v_dim: int,
        sinkhorn_n_iters: int,
        expert_dropout: float,
        weight_std_scale: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.top_k_experts = top_k_experts
        self.dropout = dropout
        self.selection_mode = selection_mode
        self.activation_after_topk = activation_after_topk
        self.activation = ACT2FN[activation]
        self.bias = bias
        self.v_dim = v_dim
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.expert_dropout = expert_dropout
        self.weight_std_scale = weight_std_scale
