import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.utils import (add_code_sample_docstrings, add_start_docstrings, 
                                add_start_docstrings_to_model_forward, logging,)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_saffu import SAFFUConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "saffu"
_CONFIG_FOR_DOC = "SAFFUConfig"

# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class SAFFUPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SAFFUConfig
    load_tf_weights = None
    base_model_prefix = "saffu"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    # def _init_weights(self, module: nn.Module):
    #     """Initialize the weights."""
    #     if isinstance(module, nn.Linear):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

SAFFU_START_DOCSTRING = r"""
"""

SAFFU_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare SAFFU encoder/transformer outputting raw hidden-states without any specific head on top.",
    SAFFU_START_DOCSTRING,
)
class SAFFUModel(SAFFUPreTrainedModel):
    def __init__(self, config: PretrainedConfig, state_dict = {}):
        super().__init__(config)
        self._N = config._N
        self._r = config._r
        self._bits = config._bits
        self._hidden = config._hidden
        self._rho = config._block_size
        self._wave_encode = config._wave_encode
        self._r_heads = int(min([config._heads, self._r]))
        self._rho_heads = int(min([config._heads, self._rho]))
        if state_dict:        
            self._V = torch.nn.Embedding(self._N, self._bits, dtype = torch.double)
            self._V.weight.requires_grad = False
            self._Y = torch.nn.Embedding(self._N, self._hidden, dtype = torch.double)
            self._Y.weight.requires_grad = False
            self._W = torch.nn.ModuleList(torch.nn.Linear(self._rho, self._rho, bias = False, dtype = torch.double)
                                          for _ in range(self._rho_heads))
            self._U = torch.nn.ModuleList(torch.nn.Linear(self._bits, self._hidden, bias = False, dtype = torch.double) 
                                          for _ in range(self._rho_heads))
            self._Wr = torch.nn.ModuleList(torch.nn.Linear(self._r, self._r, bias = False, dtype = torch.double) 
                                           for _ in range(self._r_heads))
            self._Ur = torch.nn.ModuleList(torch.nn.Linear(self._bits, self._hidden, bias = False, dtype = torch.double) 
                                           for _ in range(self._r_heads))
            self._V.weight.data = state_dict["_V.weight"]
            self._Y.weight.data = state_dict["_Y.weight"]
            for i in range(self._rho_heads):
                self._W[i].weight.data = state_dict[f"_W.{i}.weight"]
                self._U[i].weight.data = state_dict[f"_U.{i}.weight"]
            for i in range(self._r_heads):
                self._Wr[i].weight.data = state_dict[f"_Wr.{i}.weight"]    
                self._Ur[i].weight.data = state_dict[f"_Ur.{i}.weight"]
            self._Psi_r = state_dict["_Psi_r"]
            self._psi_r = state_dict["_psi_r"]
            self._Psi = state_dict["_Psi"]
            self._psi = state_dict["_psi"]    
        self.softmax = torch.softmax
        self.multiply = torch.multiply
        self.dot = torch.matmul
        self.log = torch.log
        self.exp = torch.exp


    @staticmethod
    def outer(x, y):
        return x.view(-1,1) * y

    # model's quadratic features
    def chi(self, X, v):
        return self.dot(X, v.unsqueeze(2)).squeeze(2)

    # simpler/non-attentive hidden state in case r or block_size is set to 1
    def z(self, X):
        return self.log(X)
    
    # model's hidden state
    def Z(self, X, v, W): 
        if W.shape[0] == 1:
            return self.z(X), None
        else:
            A = self.softmax(self.dot(W, self.chi(X, v).T), dim = 0)
            return self.dot(self.log(A).T.unsqueeze(1), X), A
            # return self.dot(self.log(self.softmax(self.dot(W, self.chi(X, v).T), dim = 0)).T.unsqueeze(1), X) 

    @add_start_docstrings_to_model_forward(SAFFU_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput, 
        config_class=_CONFIG_FOR_DOC,
    )
    # for training and validation/development
    def forward(self, b):
        X = (self.multiply(self._V.weight.data[b[2],:], self._Psi) if self._wave_encode else self._V.weight.data[b[2],:] if self._rho != 1 else
             self.multiply(self._V.weight.data[b[2],:], self._psi) if self._wave_encode else self._V.weight.data[b[2],:])
        Xr = (self.multiply(self._V.weight.data[b[1],:], self._Psi_r) if self._wave_encode else self._V.weight.data[b[1],:] if self._r != 1 else
              self.multiply(self._V.weight.data[b[1],:], self._psi_r) if self._wave_encode else self._V.weight.data[b[1],:])
        v = [(self.multiply(self._V.weight.data[b[3][i],:], self._psi) 
              if self._wave_encode else self._V.weight.data[b[3][i],:]) for i in range(self._rho_heads)]
        Z, A = zip(*[self.Z(X, v[i], W.weight.data) for i, W in enumerate(self._W)])
        vr = [(self.multiply(self._V.weight.data[b[3][i],:], self._psi_r) 
               if self._wave_encode else self._V.weight.data[b[3][i],:]) for i in range(self._r_heads)]
        Zr, _ = zip(*[self.Z(Xr, vr[i], W.weight.data) for i, W in enumerate(self._Wr)])
        H = -self.log(sum([self.softmax(self.dot(self._U[i].weight.data, Z[i].squeeze(1).T), dim = 0) for i, W in enumerate(self._W)] + 
                          [self.softmax(self.dot(self._Ur[i].weight.data, Zr[i].squeeze(1).T), dim = 0) for i, W in enumerate(self._Wr)]
                          )/(self._r_heads + self._rho_heads))
        return BaseModelOutput(last_hidden_state=H[:,-1], 
                               hidden_states=H, 
                               attentions=A)

@add_start_docstrings(
    """
    The SAFFU Model with a language modeling head on top (linear layer with weights tied to the vocabulary).
    """,
    SAFFU_START_DOCSTRING,
)
class SAFFULMHeadModel(SAFFUPreTrainedModel):

    def __init__(self, config, state_dict = {}):
        super().__init__(config)
        self.encoder = SAFFUModel(config, state_dict = state_dict)
        self._Uc = torch.nn.Linear(self.encoder._hidden, self.encoder._N, bias = False, dtype = torch.double)
        if state_dict:
            self._Uc.weight.data = state_dict["_Uc.weight"]

        # Initialize weights and apply final processing
        # self.post_init()

    @add_start_docstrings_to_model_forward(SAFFU_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def forward(self, b):
        out = self.encoder(b)
        H, A = out['hidden_states'], out['attentions']
        L = -self.encoder.log(self.encoder.softmax(self.encoder.dot(self._Uc.weight.data, self.encoder.exp(-H)), dim = 0))
        return CausalLMOutput(
            loss=torch.tensor([L[t,i] for i, t in enumerate(b[0])]),
            logits=L,
            hidden_states=H,
            attentions=A,
        )
