import torch
import logging

if torch.cuda.is_available():
    try:
        from .triton_src.moe_layer import MoE
    except ImportError:
        logging.warning("Could not import triton_src.moe_layer.MoE. Using cuda_src.moe_layer.MoE instead.")
        from .cuda_src.moe_layer import MoE
else:
    from .cpu_src.moe_layer import MoE


from .configuration_sigma_moe import SigmaMoEConfiguration, ACT2FN


class SigmaMoEDenseActDense(torch.nn.Module):
    def __init__(self, config: SigmaMoEConfiguration):
        super().__init__()
        self.wi = torch.nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.wo = torch.nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.act = ACT2FN[config.activation]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class SigmaMoEFeedForwardLayer(torch.nn.Module):
    def __init__(self, config: SigmaMoEConfiguration, is_sparse: bool):
        super().__init__()
        self.is_sparse = is_sparse
        self.config = config

        if self.is_sparse:
            self.ff = MoE(
                d_model=config.d_model,
                n_experts=config.n_experts,
                expert_size=config.expert_size,
                k=config.top_k_experts,
                dropout=config.dropout,
                selection_mode=config.selection_mode,
                activation_after_topk=config.activation_after_topk,
                activation=config.activation,
                bias=config.bias,
                v_dim=config.v_dim,
                sinkhorn_n_iters=config.sinkhorn_n_iters,
                expert_dropout=config.expert_dropout,
                weight_std_scale=config.weight_std_scale,
            )
        else:
            self.ff = SigmaMoEDenseActDense(config)

    def forward(self, hidden_states: torch.Tensor):
        # shape must be (bs, seq_len, d_model)
        if hidden_states.ndim == 2:
            _, d_model = hidden_states.shape
        else:
            _, _, d_model = hidden_states.shape
        assert d_model == self.config.d_model, f"Expected {self.config.d_model} but got {d_model}."

        if self.is_sparse:
            hidden_states, reg_loss = self.ff(hidden_states)
        else:
            hidden_states = self.ff(hidden_states)
            reg_loss = None
        return hidden_states, reg_loss
        


class SigmaMoETransformerLayer(torch.nn.Module):
    """
    This layer can be a decoder or an encoder layer.
    
    If we get hidden encoder states in the forward,
    and this is a decoder layer, we use cross attention.
    
    If we don't get hidden encoder states in the forward
    and this is a decoder layer, we don't do cross attention.

    This module is a single layer, meaning that it has one
    self attention (always) and one cross attention (if decoder
    and enc-dec architecture) and one SigmaMoEFeedForwardLayer.
    The module also has layer norms, dropout and the standard
    residual connections.
    """

    def __init__(self, config: SigmaMoEConfiguration, is_sparse: bool):
        super().__init__()