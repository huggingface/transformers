# coding=utf-8
# Copyright 2025 Fromthesky Research Labs, LLC. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code uses the Llama model implementation by Eleuther AI 
# and Huggingface teams in this library as a starting point and implements 
# the PLDR-LLM (Large Language Model from Power Law Decoder Representations)
#  architecture based on its implementation by the Fromthesky Research Labs team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer

from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from .configuration_pldrllm import PldrllmConfig

from dataclasses import dataclass
from ...utils import ModelOutput

logger = logging.get_logger(__name__)

################## PLDRLLM POWER LAW GRAPH ATTENTION IMPLEMENTATION ########################################

''''
Power law attention implementation for PLDR-LLM with KV-cache and G-cache.
'''

class PlgaLayer(nn.Module):
    '''
    Power law graph attention layer implementation.
    '''
    def __init__(self, config:PldrllmConfig, 
                 F_hidden:int, 
                 F_heads:int, 
                 layer_idx:int,  
                 device=None, 
                 **kwargs)->None:
        '''
        Args:
            F_hidden: hidden layer shape used in layer weight creation. For multi-head plga this is head_dim.
            F_heads: Number of attention heads.
            layer_idx: index for the decoder layer.
            device: device(cpu or gpu) to load tensors.
        '''

        super().__init__(**kwargs)
        self.F_hidden=F_hidden
        self.F_heads=F_heads
        self.layer_idx=layer_idx
        self.device=device
        self.config=config
        self.is_causal = True
        self.custom_G_type=config.custom_G_type
        self.attention_dropout=config.attention_dropout

        # default type is set as config.torch_dtype
        self.wdtype=None 

        if self.custom_G_type is None:
            self.build_weights()
        else:
            self.Wlst = None
            self.blst = None
            self.pwlst = None 
            self.alst = None
            self.balst = None



    def cg_align_one(self, Hin:torch.Tensor, 
                     Hk:torch.Tensor, 
                     Hv:torch.Tensor, 
                     A:torch.Tensor, 
                     a_vec:Optional[torch.Tensor], 
                     ba:Optional[torch.Tensor],
                     W:Optional[torch.Tensor],
                     b:Optional[torch.Tensor], 
                     pw:Optional[torch.Tensor],
                     past_G_values: Optional[torch.Tensor],
                     past_G_values_status: Optional[torch.BoolTensor]=None,
                     mask:Optional[torch.Tensor]=None,
                     use_cache: Optional[bool]=None,
                     **kwargs)->tuple[torch.Tensor, tuple[torch.Tensor,...]]:
        '''
        Alignment model for calculating attention weights
        Args:
            Hin: query
            Hk: key
            A: metric tensor instance
            a_vec: learned coupling coefficients.
            ba: bias for coupling coeffients
            W: weights applied on metric tensor before AdjActivation
            b: bias applied on metric tensor before AdjActivation
            pw: learned power exponents applied on metric tensor
            mask: padding or lookahead mask
        Returns:
            Hout: Attention output.
            A tuple of:
                A: metric tensor as output of residual metric learner layer, A
                AW: metric tensor after AdjActivation is applied, A_LM
                pw: learned power exponents
                a_vec: learned coupling coefficients for energy-curvature tensor
                ba: bias for energy-curvature tensor
                avAp: Energy curvature tensor, G_LM
                E: attention weights
        '''

        if self.custom_G_type is None and not (use_cache and past_G_values_status[self.layer_idx]):

            AdjActivation=iSwiGLU
            epsilonAdj=1e-9

            # make metric tensor positive definite
            AW=AdjActivation(torch.matmul(W,A)+b)+epsilonAdj

            # find energy curvature tensor and attention weights
            Ap=torch.pow(AW, pw)
            avAp=torch.matmul(a_vec, Ap)+ba # [batch_size, num_head,  depth, depth]

            if use_cache:
                # update only once if cache is enabled.
                G_batch_size=past_G_values.size()[2]
                past_G_values[self.layer_idx]=torch.stack([A[:G_batch_size,:,:,:], 
                                                            AW[:G_batch_size,:,:,:], 
                                                            avAp[:G_batch_size,:,:,:]], dim=0) # [3, batch_size, num_head,  depth, depth]
                past_G_values_status[self.layer_idx]=True
        else:
            AW=past_G_values[self.layer_idx, 1]
            avAp=past_G_values[self.layer_idx, 2]

        WHiWHj = torch.matmul(Hin, avAp) # [batch_size, num_head, seq_lenq, depth]

        # scale attention with square root of depth
        dk=torch.tensor(self.F_hidden).to(Hin.dtype)
        scaling=1/torch.sqrt(dk)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        query, key, value = WHiWHj.to(dtype=Hk.dtype), Hk, Hv

        Hout, E = attention_interface(
            self,
            query=query,
            key=key,
            value=value,
            attention_mask=mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=scaling,
            **kwargs
        )

        return Hout, (A, AW, pw, a_vec, ba, avAp, E)
    
    def cg_align_head(self, Hin:torch.Tensor, 
                      Hk:torch.Tensor, 
                      Hv:torch.Tensor, 
                      A:torch.Tensor, 
                      mask:Optional[torch.Tensor]=None,
                      past_G_values: Optional[torch.Tensor]=None,
                      past_G_values_status: Optional[torch.BoolTensor]=None,
                      use_cache: Optional[bool]=None,
                      **kwargs)->tuple[torch.Tensor, tuple[torch.Tensor,...]]:
        '''
        Method for linear propagation of attention weights over values.
        '''

        Hout, att_weights=self.cg_align_one(Hin=Hin, Hk=Hk, Hv=Hv, A=A, 
                                            a_vec=self.alst,
                                            ba=self.balst,
                                            W=self.Wlst,
                                            b=self.blst,
                                            pw=self.pwlst, 
                                            mask=mask,
                                            past_G_values=past_G_values,
                                            past_G_values_status=past_G_values_status,
                                            use_cache=use_cache,
                                            **kwargs)

        return Hout, att_weights



    def build_weights(self)->None:
        '''
        Used to initialize learnable parameters for the layer:
        W: weights to apply on metric tensor.
        b: bias to apply on metric tensor.
        a: coupling coefficients for energy-curvature (G) tensor.
        ba: bias for energy-curvature tensor.
        pw: power exponent weights for potential tensor.
        '''

        weight_shape=[self.F_heads, self.F_hidden, self.F_hidden] # [num_heads, depth, depth]

        add_weight_Wpart= torch.empty(weight_shape, dtype=self.wdtype, device=self.device) 
        add_weight_bpart=torch.empty(weight_shape, dtype=self.wdtype, device=self.device) 
        add_weight_pwpart=torch.empty(weight_shape, dtype=self.wdtype, device=self.device)
        add_weight_apart = torch.empty(weight_shape, dtype=self.wdtype, device=self.device)
        add_weight_bapart=torch.empty(weight_shape, dtype=self.wdtype, device=self.device)

        self.Wlst = nn.Parameter(add_weight_Wpart, requires_grad=True)
        self.blst = nn.Parameter(add_weight_bpart, requires_grad=True) 
        self.pwlst = nn.Parameter(add_weight_pwpart, requires_grad=True)  
        self.alst = nn.Parameter(add_weight_apart, requires_grad=True) 
        self.balst = nn.Parameter(add_weight_bapart, requires_grad=True) 


    def forward(self, inputs:tuple[torch.Tensor,...],
                past_G_values: Optional[torch.Tensor]=None,
                past_G_values_status: Optional[torch.BoolTensor]=None,
                use_cache:Optional[bool]=False, 
                **kwargs)->tuple[torch.Tensor, tuple[torch.Tensor,...]]:
        '''
        execute the forward propagation
        inputs[0] = query = Hin
        inputs[1] = key = Hk
        inputs[2] = value = Hv
        inputs[3] = metric tensor = A
        inputs[4] = mask
        '''

        Hin, Hk, Hv, A, mask=inputs
        H_next, att_weights = self.cg_align_head(Hin=Hin, Hk=Hk, Hv=Hv, A=A, mask=mask,
                                                 past_G_values=past_G_values,
                                                 past_G_values_status=past_G_values_status,
                                                 use_cache=use_cache, **kwargs)
        return H_next, att_weights

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs:Unpack[TransformersKwargs],
    )->tuple[torch.Tensor, torch.Tensor]:

    keyt=torch.permute(key, [0, 1, 3, 2])  # [batch_size, num_head, depth, seq_lenk]
    attn_weights = torch.matmul(query, keyt) * scaling # [batch_size, num_head, seq_lenq, seq_lenk]
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = torch.permute(attn_output, [0, 2, 1, 3])
    attn_output = attn_output.contiguous()

    return attn_output, attn_weights

def iSwiGLU(x):
    '''SwiGLU activation function with weights W,V equal to identity matrix and no bias.'''
    gate=F.silu(x)
    out=torch.mul(x, gate)
    return out

################################### END OF PLDRLLM POWER LAW GRAPH ATTENTION IMPLEMENTATION ############################################

#################################### PLDR-LLM MODEL IMPLEMENTATION ################################################################

'''
Model Implementation for Large Language Model from Power Law Decoder Representations with KV-cache and G-cache.
'''

class PldrllmAttention(nn.Module):
    '''
    Power Law Multihead Attention Implementation for PLDR-LLM.
    '''
    def __init__(self,config: PldrllmConfig, 
                 layer_idx:int, 
                 device=None, 
                 **kwargs)->None:


        super().__init__(**kwargs)
        self.num_heads = config.num_attention_heads 
        self.d_model = config.hidden_size 
        self.A_dff = config.A_dff
        self.num_denseA = config.num_denseA
        self.num_reslayerA = config.num_reslayerA
        self.activation=ACT2FN[config.hidden_act]
        self.max_seq_len=config.max_position_embeddings
        self.layer_idx=layer_idx
        self.device=device
        self.attention_bias=config.attention_bias
        self.custom_G_type=config.custom_G_type
        self.layer_norm_eps=config.layer_norm_eps
        self.glu_bias=config.glu_bias
        self.reference_rope=config.reference_rope
        self.wdtype=None

        assert self.d_model % self.num_heads == 0
        self.depth = config.head_dim

        self.wq = nn.Linear(self.d_model, self.d_model, bias=self.attention_bias, device=self.device, dtype=self.wdtype)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=self.attention_bias, device=self.device, dtype=self.wdtype)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=self.attention_bias, device=self.device, dtype=self.wdtype)

        self.plgatt_layer= PlgaLayer(config=config,
                                      F_hidden=self.depth,
                                      F_heads= self.num_heads,
                                      layer_idx=self.layer_idx,
                                      device=self.device)

        self.dense = nn.Linear(self.d_model, self.d_model, bias=self.attention_bias, device=self.device, dtype=self.wdtype)

        if self.custom_G_type is None:
            # residual layers for metric tensor learning
            self.reslayerAs=nn.ModuleList([ResLayerA(depth=self.depth, 
                                                    A_dff=self.A_dff,
                                                    num_denseA=self.num_denseA,
                                                    layer_norm_eps=self.layer_norm_eps,
                                                    glu_bias=self.glu_bias,
                                                    activation=self.activation,
                                                    device=self.device,
                                                    dtype=self.wdtype) for _ in range(self.num_reslayerA)])
            
            self.layernorm1 = nn.LayerNorm(self.depth, eps=self.layer_norm_eps, device=self.device, dtype=self.wdtype)
        
        if self.reference_rope:
            # keep initialization and forward in same module for reference rope implementation
            self.rotary_embedding=RotaryPositionalEmbeddings(dim=self.depth, 
                                                             max_seq_len=self.max_seq_len, 
                                                             base=config.rope_theta
                                                             ).to(device=self.device, dtype=self.wdtype)

        

    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (num_heads, depth).
        '''
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x # [batch_size, seq_len, num_heads, depth]

    def forward(self, inputs:tuple[torch.Tensor, ...],
                position_embeddings:torch.Tensor,
                position_ids: Optional[torch.LongTensor]=None,
                cache_position:Optional[torch.LongTensor]=None,
                past_G_values: Optional[torch.Tensor]=None,
                past_G_values_status: Optional[torch.BoolTensor]=None,
                past_key_values: Optional[Cache]=None,
                use_cache:Optional[bool]=None,
                **kwargs: Unpack[TransformersKwargs]               
                )->tuple[torch.Tensor, tuple[torch.Tensor,...]]:

        q, k, v, mask = inputs
        batch_size = q.size()[0]

        q = self.wq(q)  # [batch_size, seq_len, d_model]
        k = self.wk(k)
        v = self.wv(v)


        q = self.split_heads(q, batch_size)  # [batch_size, seq_len, num_heads, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)


        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q=q, k=k, cos=cos, sin=sin, unsqueeze_dim=2)
        else:
            q=self.rotary_embedding(q, input_pos=position_ids)
            k=self.rotary_embedding(k, input_pos=position_ids)
        
        q = torch.permute(q, [0, 2, 1, 3]) # [batch_size, num_heads, seq_len, depth]
        k = torch.permute(k, [0, 2, 1, 3]) 
        v = torch.permute(v, [0, 2, 1, 3]) 
        
        if self.custom_G_type is None and not (use_cache and past_G_values_status[self.layer_idx]):
            # Calculate density matrix using linear self attention           
            qt = torch.permute(q, [0, 1, 3, 2])
            A = torch.matmul(qt, q)  # [batch_size, num_head, depth, depth]
            A=self.layernorm1(A)

            #Deep residual network for learning metric tensor
            for i in range(self.num_reslayerA):
                A=self.reslayerAs[i]([A])
        else:
            A=past_G_values[self.layer_idx,0] # [1, num_head, depth, depth]
        
        if use_cache:
            #cache position for static cache
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_values.update(key_states=k, value_states=v, layer_idx=self.layer_idx, cache_kwargs=cache_kwargs)

        #Apply multi-head power law attention
        Hnext, att_weights = self.plgatt_layer((q, k, v, A, mask),
                                               past_G_values,
                                               past_G_values_status,
                                               use_cache, **kwargs)

        Hnext= Hnext.reshape(batch_size, -1, self.d_model) # [batch_size, seq_len, d_model]

        output = self.dense(Hnext)

        return output, att_weights


class PLDR_DecoderLayer(GradientCheckpointingLayer):
    '''
    Single decoder layer implementation for PLDR-LLM with single masked multihead attention.
    '''
    def __init__(self, config: PldrllmConfig, 
                 layer_idx:int, 
                 device=None, 
                 **kwargs)->None:

        super().__init__(**kwargs)

        self.d_model=config.hidden_size 
        self.num_heads=config.num_attention_heads
        self.dff=config.intermediate_size
        self.A_dff=config.A_dff
        self.num_denseA = config.num_denseA
        self.num_reslayerA = config.num_reslayerA
        self.activation=ACT2FN[config.hidden_act]
        self.max_seq_len=config.max_position_embeddings
        self.layer_idx=layer_idx
        self.device=device
        self.layer_norm_eps=config.layer_norm_eps
        self.glu_bias=config.glu_bias
        self.wdtype=None

        self.mha1 = PldrllmAttention(config=config, layer_idx=layer_idx, device=self.device)

        self.ffn = self.dec_point_wise_feed_forward_network()

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps, device=self.device, dtype=self.wdtype)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps,  device=self.device, dtype=self.wdtype)

    def forward(self, 
                hidden_states:torch.Tensor,
                look_ahead_mask:torch.Tensor,
                position_embeddings:torch.Tensor,
                position_ids:Optional[torch.LongTensor]=None,
                cache_position:Optional[torch.LongTensor]=None,
                use_cache:Optional[bool]=None,
                past_key_values:Optional[Cache]=None,
                past_G_values:Optional[torch.Tensor]=None,
                past_G_values_status:Optional[list[bool]]=None,
                **kwargs:Unpack[TransformersKwargs]
                )->tuple[torch.Tensor, tuple[torch.Tensor,...]]:

        attn1, att_weights = self.mha1(inputs=[hidden_states, hidden_states, hidden_states, look_ahead_mask],
                                        position_embeddings=position_embeddings,
                                        position_ids=position_ids,
                                        cache_position=cache_position,
                                        past_key_values=past_key_values,
                                        past_G_values=past_G_values,
                                        past_G_values_status=past_G_values_status,
                                        use_cache=use_cache,
                                        **kwargs
                                        )
        out1 = self.layernorm1(attn1 + hidden_states)

        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(ffn_output + out1)  # [batch_size, target_seq_len, d_model]

        return out2, att_weights


    # GLUVariant implementation for feedforward network, scale dff accordingly (i.e., 2/3 of original).
    def dec_point_wise_feed_forward_network(self):
        return GLUVariant(self.d_model, self.dff, self.d_model, 
                          glu_bias=self.glu_bias,
                          activation=self.activation, 
                          device=self.device,
                          dtype=self.wdtype)


class ResLayerA(nn.Module):
    '''
    Residual Layer implementation for metric learner of PLDR-LLM
    '''
    def __init__(self, depth:int,
                 A_dff:int,
                 num_denseA:int, 
                 layer_norm_eps:float, 
                 glu_bias:bool,
                 activation:Callable=F.silu,  
                 device=None,
                 dtype=None,  
                 **kwargs)->None:
        super().__init__(**kwargs)
        self.depth=depth
        self.A_dff = A_dff
        self.num_denseA = num_denseA
        self.activation=activation
        self.device=device
        self.layer_norm_eps=layer_norm_eps
        self.glu_bias=glu_bias

        self.denseAs = nn.ModuleList([GLUVariant(self.depth, self.A_dff, self.depth,
                                                 glu_bias=self.glu_bias,
                                                 activation=self.activation, 
                                                 device=self.device,
                                                 dtype=dtype) for _ in range(self.num_denseA)])

        self.layernormA = nn.LayerNorm(self.depth, eps=self.layer_norm_eps, device=self.device, dtype=dtype)
        self.identity=nn.Identity()
    
    def ResUnit(self, A:torch.Tensor)->torch.Tensor:
        Ain = self.identity(A)
        for i in range(self.num_denseA):
            A = self.denseAs[i](A)
        A = self.layernormA(A + Ain)
        return A

    def forward(self, inputs:list[torch.Tensor], **kwargs)->torch.Tensor:
        A=inputs[0]
        return self.ResUnit(A)


class GLUVariant(nn.Module):
    '''
    Implementation of GLU variants with default activation for SwiGLU configuration 
    For the hidden layer dff, to match size with non-SwiGLU FFN version scaling with 2/3 may be useful.
    '''
    def __init__(self, d_model:int, 
                 dff:int, 
                 depth:int, 
                 glu_bias:bool,
                 activation:Callable=F.silu, 
                 device=None,
                 dtype=None,
                 **kwargs)->None:
        super().__init__(**kwargs)
        self.dff=dff
        self.depth=depth
        self.d_model=d_model
        self.activation=activation
        self.device=device
        self.glu_bias=glu_bias

        self.gluw1=nn.Linear(self.d_model, self.dff, bias=self.glu_bias, device=self.device, dtype=dtype)
        self.gluw2=nn.Linear(self.d_model, self.dff, bias=self.glu_bias, device=self.device, dtype=dtype)
        self.gluw3=nn.Linear(self.dff, self.depth, bias=self.glu_bias, device=self.device, dtype=dtype)

    def forward(self, input:torch.Tensor, **kwargs)->torch.Tensor:
        x1=self.gluw1(input)
        x1=self.activation(x1)
        x2=self.gluw2(input)
        return self.gluw3(torch.mul(x1, x2))


###################################### END OF PLDRLLM MODEL IMPLEMENTATION #####################################################


# RotaryPositionalEmbeddings is from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
# This implementation was  used in the original pytorch based implementation of PLDR-LLM.
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)



class PldrllmRotaryEmbedding(nn.Module):
    def __init__(self, config: PldrllmConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

############# END OF ROTARY EMBEDDING IMPLEMENTATION #################################################

@dataclass
class BasePLDRModelOutputWithPast(ModelOutput):
    """
    Base class for [`PldrllmModel`] outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        pldr_attentions (`tuple(tuple(torch.Tensor)))`, *optional*, returned when `output_pldr_attentions=True` is passed or when `config.output_pldr_attentions=True`):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of the deductive outputs and learnable parameters of power law graph attention module.

            The tuple for each layer contains:
            output of the residual metric learner (metric tensor, A) of shape `(batch_size, num_heads, head_dim,head_dim)`,
            output after application of iSwiGLU on metric tensor, A_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned exponents of potential tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned weights for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned bias for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            energy-curvature tensor G_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            attention weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    """
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    pldr_attentions:Optional[tuple[tuple[torch.Tensor, ...]]]  = None

@dataclass
class CausalPLDRLLMOutputWithPast(ModelOutput):
    """
    Base class for [`PldrllmForCausalLM`] causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        pldr_attentions (`tuple(tuple(torch.Tensor)))`, *optional*, returned when `output_pldr_attentions=True` is passed or when `config.output_pldr_attentions=True`):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of the deductive outputs and learnable parameters of power law graph attention module.

            The tuple for each layer contains:
            output of the residual metric learner (metric tensor, A) of shape `(batch_size, num_heads, head_dim,head_dim)`,
            output after application of iSwiGLU on metric tensor, A_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned exponents of potential tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned weights for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned bias for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            energy-curvature tensor G_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            attention weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    pldr_attentions:Optional[tuple[tuple[torch.Tensor, ...]]] = None

@dataclass
class TokenClassifierPLDRLLMOutput(ModelOutput):
    """
    Base class for outputs of [`PldrllmForTokenClassification`] token classification model.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        pldr_attentions (`tuple(tuple(torch.Tensor)))`, *optional*, returned when `output_pldr_attentions=True` is passed or when `config.output_pldr_attentions=True`):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of the deductive outputs and learnable parameters of power law graph attention module.

            The tuple for each layer contains:
            output of the residual metric learner (metric tensor, A) of shape `(batch_size, num_heads, head_dim,head_dim)`,
            output after application of iSwiGLU on metric tensor, A_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned exponents of potential tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned weights for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned bias for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            energy-curvature tensor G_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            attention weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    pldr_attentions:Optional[tuple[tuple[torch.Tensor, ...]]] = None

@dataclass
class QuestionAnsweringPLDRModelOutput(ModelOutput):
    """
    Base class for outputs of [`PldrllmForQuestionAnswering`] question answering model.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        pldr_attentions (`tuple(tuple(torch.Tensor)))`, *optional*, returned when `output_pldr_attentions=True` is passed or when `config.output_pldr_attentions=True`):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of the deductive outputs and learnable parameters of power law graph attention module.

            The tuple for each layer contains:
            output of the residual metric learner (metric tensor, A) of shape `(batch_size, num_heads, head_dim,head_dim)`,
            output after application of iSwiGLU on metric tensor, A_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned exponents of potential tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned weights for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned bias for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            energy-curvature tensor G_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            attention weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    loss: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    pldr_attentions:Optional[tuple[tuple[torch.Tensor, ...]]] = None

@dataclass
class SequenceClassifierPLDRLLMOutputWithPast(ModelOutput):
    """
    Base class for outputs of [`PldrllmForSequenceClassification`] sentence classification model.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        pldr_attentions (`tuple(tuple(torch.Tensor)))`, *optional*, returned when `output_pldr_attentions=True` is passed or when `config.output_pldr_attentions=True`):
        Tuple of `tuple(torch.Tensor)` (one for each layer) of the deductive outputs and learnable parameters of power law graph attention module.

            The tuple for each layer contains:
            output of the residual metric learner (metric tensor, A) of shape `(batch_size, num_heads, head_dim,head_dim)`,
            output after application of iSwiGLU on metric tensor, A_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned exponents of potential tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned weights for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            learned bias for energy-curvature tensor of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            energy-curvature tensor G_LM of shape `(batch_size, num_heads, head_dim,head_dim)`, 
            attention weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    pldr_attentions:Optional[tuple[tuple[torch.Tensor, ...]]] = None


@auto_docstring
class PldrllmPreTrainedModel(PreTrainedModel):
    config_class = PldrllmConfig
    base_model_prefix = "decoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PLDR_DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_attention_backend = True
    _can_compile_fullgraph=False
    
    def __init__(self, config: PldrllmConfig)->None:
        super().__init__(config)
        self.custom_G_type=config.custom_G_type
        if self.custom_G_type is not None:
            self._can_compile_fullgraph=True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, PlgaLayer):
            if module.Wlst is not None:
                nn.init.xavier_uniform_(module.Wlst.data)
            if module.pwlst is not None:
                nn.init.xavier_uniform_(module.pwlst.data)
            if module.alst is not None:
                nn.init.xavier_uniform_(module.alst.data)
            if module.blst is not None:
                module.blst.data.zero_()
            if module.balst is not None:
                module.balst.data.zero_()

MODEL_COMMON_CUSTOM_ARGS=r"""
        output_pldr_attentions (`bool`, *optional*, defaults to `False`):
            Whether to return the deductive outputs and learnable parameters of power law graph attention module as tuple containing:
            the output of the residual metric learner (metric tensor, A), output (A_LM) after application of iSwiGLU on metric tensor, learned 
            exponents of potential tensor, learned weights for energy-curvature tensor, learned bias for
            energy-curvature tensor, energy-curvature tensor (G_LM), and attention weights.
        cache_first_G (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the G values from first sample in a batch or G values from all samples for past_G_values initialization. 
            When `cache_first_G=true`, the batch_size of past_G_values is 1. This argument should be set to True for contrastive text generation 
            with learned G values.
        """


@auto_docstring(custom_intro="""
                Large Language Model From Power Law Decoder Representations (PLDR-LLM) with decoder hidden state as output.
                PLDR-LLM is a model architecture that utilizes Power Law Graph Attention (PLGA) in decoder layers.
                For details of model architecture, check out these papers:
                [Paper-1](https://huggingface.co/papers/2107.02039) [Paper-2](https://huggingface.co/papers/2410.16703) [Paper-3](https://huggingface.co/papers/2502.13502)
                """
                )
class PldrllmModel(PldrllmPreTrainedModel):
    def __init__(self, config: PldrllmConfig)->None:
        super().__init__(config)

        # Initialize weights and apply final processing
        self.num_layers = config.num_hidden_layers
        self.d_model=config.hidden_size
        self.num_heads=config.num_attention_heads
        self.target_vocab_size =config.vocab_size
        self.max_seq_len=config.max_position_embeddings
        self.reference_rope=config.reference_rope
        self.pldr_device=None
        self.gradient_checkpointing = False
        self.layer_norm_eps=config.layer_norm_eps
        self.wdtype=None

        assert self.d_model % self.num_heads == 0
        self.depth = config.head_dim

        self.custom_G_type=config.custom_G_type

        if self.custom_G_type is not None:
            # predefined past_G_values are initialized for both training and inference
            past_G_values, past_G_values_status=self.G_values_init(device=self.pldr_device, dtype=self.wdtype)
            self.register_buffer("past_G_values_status", past_G_values_status, persistent=True)
            self.register_buffer("past_G_values", past_G_values, persistent=True)

            logger.warning("\nIMPORTANT: decoder.past_G_values are set to predefined values and deep PLGA layers will be skipped. "
                           "Set config.custom_G_type=None to enable deep PLGA layers.")
            if self.custom_G_type=="external":
                logger.warning("\nIMPORTANT: config.custom_G_type is selected as 'external' and an external value of decoder.past_G_values[:,2,...] is expected. "
                               "decoder.past_G_values[:,2,...] are initialized to identity tensor by default. This is equivalent to an LLM with SDPA. To provide external values "
                               "to the decoder.past_G_values, either load these values along with the pretrained model or set decoder.past_G_values to a torch.float tensor of " 
                               "size (num_layers, 3, 1, num_heads, head_dim, head_dim) after model is initialized.\n")
        else:
            # learned past_G_values is initialized at inference.
            self.register_buffer("past_G_values_status", None, persistent=False)
            self.register_buffer("past_G_values", None, persistent=False)
            self.is_past_G_values_initialized=False


        self.embedding = nn.Embedding(self.target_vocab_size, self.d_model, device=self.pldr_device, dtype=self.wdtype)

        self.dec_layers = nn.ModuleList([PLDR_DecoderLayer(config,
                                                           layer_idx=i,
                                                           device=self.pldr_device) for i in range(self.num_layers)])

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps, device=self.pldr_device, dtype=self.wdtype)

        if not self.reference_rope:
            self.rotary_embedding=PldrllmRotaryEmbedding(config=config)

        self.post_init()

    def G_values_init(self, batch_size=1, device=None, dtype=None):
        G_values_dim=(self.num_layers, 1, self.num_heads, self.depth, self.depth) # [num_layers, 1, num_heads, depth, depth]
        zeros_tensor=torch.zeros(G_values_dim, device=device, dtype=dtype) 
        identity_tensor=torch.eye(self.depth).repeat(self.num_layers, 1, self.num_heads, 1, 1).to(device=device, dtype=dtype)
        random_tensor=torch.randn(G_values_dim, device=device, dtype=dtype)     
        CUSTOM_G_VALUES={
                         'identity':torch.stack([zeros_tensor, zeros_tensor, identity_tensor], dim=1), # [num_layers, 3, num_heads, depth, depth]
                         'random': torch.stack([zeros_tensor, zeros_tensor, random_tensor], dim=1),
                         'external': torch.stack([zeros_tensor, zeros_tensor, identity_tensor], dim=1)
                         }

        if self.custom_G_type is None:
            # 3 tensors for A, AW and avAp per layer
            past_G_values = torch.zeros((self.num_layers, 3, batch_size, self.num_heads, self.depth, self.depth), device=device, dtype=dtype) 
            past_G_values_status=torch.tensor([False]*self.num_layers, dtype=torch.bool, device=device)
        elif self.custom_G_type in ['identity', 'random', 'external']:
            past_G_values=CUSTOM_G_VALUES[self.custom_G_type]
            past_G_values_status=torch.tensor([True]*self.num_layers,  dtype=torch.bool, device=device)
        else:
            raise ValueError("Invalid custom_G_type value. Available values are "
                             "None, 'identity', 'random', and 'external'.")
        
        self.is_past_G_values_initialized=True
        return past_G_values, past_G_values_status

    @can_return_tuple
    @auto_docstring(
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache]=None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_pldr_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                cache_first_G: Optional[bool] = None,
                **kwargs: Unpack[TransformersKwargs]
                        ):

        use_cache=use_cache if use_cache is not None else self.config.use_cache
        cache_first_G=cache_first_G if cache_first_G is not None else self.config.cache_first_G
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_pldr_attentions=output_pldr_attentions if output_pldr_attentions is not None else self.config.output_pldr_attentions
        output_hidden_states=output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if (self.gradient_checkpointing or self.training) and use_cache:
            logger.warning_once(
                "During training, setting `use_cache=False`. Additionally, `use_cache=True` is incompatible with gradient checkpointing."
            )
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds  # [batch_size, target_seq_len, d_model]

        dec_att_weights=() if output_pldr_attentions else None
        dec_attentions=() if output_attentions else None
 
        dec_outputs=(inputs_embeds,) if output_hidden_states else None

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # reset past_G_Values_status if they are not custom and predefined.
        if use_cache and self.custom_G_type is None and not isinstance(past_key_values, StaticCache) and  past_key_values.get_seq_length()==0:
            self.past_G_values_status=torch.tensor([False]*self.num_layers, dtype=torch.bool, device=inputs_embeds.device)
            self.is_past_G_values_initialized=False
        
        if use_cache and isinstance(past_key_values, StaticCache) and ((self.custom_G_type is None) or
                                                                         "flash_attention" in self.config._attn_implementation):
            raise ValueError("Static Cache is only supported with predefined past_G_values. "
                             "Flash attention is not supported. "
                             "Supported models are with config.custom_G_type set to 'random', 'identity' or 'external'.")
        
        if not self.is_past_G_values_initialized  and self.custom_G_type is None:
            if use_cache:
                batch_size=1 if cache_first_G else inputs_embeds.size()[0]
                self.past_G_values, self.past_G_values_status=self.G_values_init(batch_size=batch_size,
                                                                                 device=inputs_embeds.device, 
                                                                                 dtype=inputs_embeds.dtype)
            else:
                self.past_G_values_status=torch.tensor([False]*self.num_layers, dtype=torch.bool, device=inputs_embeds.device)
                self.past_G_values=None
                self.is_past_G_values_initialized=True

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        hidden_states=inputs_embeds
        # create position embeddings to be shared across the decoder layers
        if not self.reference_rope:
            position_embeddings = self.rotary_embedding(hidden_states, position_ids)
        else:
            # defer reference rope initialization in the PldrllmAttention module.
            position_embeddings=None

        hidden_states *= torch.sqrt(torch.tensor(self.d_model).to(dtype=hidden_states.dtype))

        hidden_states=self.layernorm1(hidden_states)

        for i in range(self.num_layers):
            hidden_states, dec_att_w= self.dec_layers[i](hidden_states, 
                                                         causal_mask,
                                                         position_embeddings=position_embeddings,
                                                         position_ids=position_ids,
                                                         cache_position=cache_position,
                                                         use_cache=use_cache,
                                                         past_key_values=past_key_values,
                                                         past_G_values=self.past_G_values,
                                                         past_G_values_status=self.past_G_values_status,
                                                         **kwargs
                                                         )

            if output_pldr_attentions:
                dec_att_weights += (dec_att_w,)

            if output_attentions:
                dec_attentions += (dec_att_w[-1],)

            if output_hidden_states:
                dec_outputs += (hidden_states,)

        last_hidden_state=hidden_states

        return BasePLDRModelOutputWithPast(
            last_hidden_state = last_hidden_state,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=dec_outputs,
            attentions=dec_attentions,
            pldr_attentions=dec_att_weights
        )

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

@auto_docstring(custom_intro="""
                Large Language Model From Power Law Decoder Representations (PLDR-LLM) with LM Head as final layer.
                PLDR-LLM is a model architecture that utilizes Power Law Graph Attention (PLGA) in decoder layers.
                For details of model architecture, check out these papers:
                [Paper-1](https://huggingface.co/papers/2107.02039) [Paper-2](https://huggingface.co/papers/2410.16703) [Paper-3](https://huggingface.co/papers/2502.13502)
                """
                )
class PldrllmForCausalLM(PldrllmPreTrainedModel, GenerationMixin):
    def __init__(self, config: PldrllmConfig)->None:
        super().__init__(config)

        self.d_model=config.hidden_size
        self.input_vocab_size =config.vocab_size
        self.final_bias=config.final_bias
        self.pldr_device=None
        self.decoder=PldrllmModel(config=config)
        self.wdtype=None

        self.final_layer = nn.Linear(self.d_model, self.input_vocab_size, bias=self.final_bias, device=self.pldr_device, dtype=self.wdtype)

        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embedding


    def set_input_embeddings(self, value):
        self.decoder.embedding = value

    def get_output_embeddings(self):
        return self.final_layer

    def set_output_embeddings(self, new_embeddings):
        self.final_layer = new_embeddings

    def set_decoder(self, decoder):
        self.decoder = decoder

    def get_decoder(self):
        return self.decoder

    @can_return_tuple
    @auto_docstring(
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(self, 
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache]=None,
                use_cache: Optional[bool] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_pldr_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                cache_first_G: Optional[bool] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs: Unpack[TransformersKwargs],
                )-> CausalPLDRLLMOutputWithPast:

        outputs: BasePLDRModelOutputWithPast=self.decoder(input_ids=input_ids,
                                                          attention_mask=attention_mask,
                                                          position_ids=position_ids,
                                                          past_key_values=past_key_values,
                                                          use_cache=use_cache,
                                                          inputs_embeds=inputs_embeds,
                                                          output_attentions=output_attentions,
                                                          output_pldr_attentions=output_pldr_attentions,
                                                          output_hidden_states=output_hidden_states,
                                                          cache_position=cache_position,
                                                          cache_first_G=cache_first_G,
                                                          **kwargs
                                                         )


        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep        
        logits = self.final_layer(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalPLDRLLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions= outputs.attentions, #list of E
            pldr_attentions=outputs.pldr_attentions
        )

@auto_docstring
class PldrllmForTokenClassification(PldrllmPreTrainedModel):
    def __init__(self, config:PldrllmConfig)->None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.decoder = PldrllmModel(config)
        self.wdtype=None
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=self.wdtype)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embedding

    def set_input_embeddings(self, value):
        self.decoder.embedding = value

    @can_return_tuple
    @auto_docstring(
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_pldr_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_first_G: Optional[bool] = None,
    ) -> TokenClassifierPLDRLLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BasePLDRModelOutputWithPast = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_pldr_attentions=output_pldr_attentions,
            cache_first_G=cache_first_G
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierPLDRLLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pldr_attentions=outputs.pldr_attentions
        )


@auto_docstring
class PldrllmForQuestionAnswering(PldrllmPreTrainedModel):

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama->Pldrllm
    def __init__(self, config:PldrllmConfig):
        super().__init__(config)
        self.decoder = PldrllmModel(config)
        self.wdtype=None
        self.qa_outputs = nn.Linear(config.hidden_size, 2, bias=True, dtype=self.wdtype)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embedding

    def set_input_embeddings(self, value):
        self.decoder.embedding = value

    @can_return_tuple
    @auto_docstring(
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_pldr_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_first_G: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringPLDRModelOutput:
        outputs: BasePLDRModelOutputWithPast = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_pldr_attentions=output_pldr_attentions,
            cache_first_G=cache_first_G
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringPLDRModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pldr_attentions=outputs.pldr_attentions
        )

@auto_docstring(
    custom_intro="""
    The PLDR-LLM with a sequence classification head on top (linear layer).

    [`PldrllmForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class PldrllmForSequenceClassification(PldrllmPreTrainedModel):
    def __init__(self, config:PldrllmConfig)->None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.decoder = PldrllmModel(config)
        self.wdtype=None
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False, dtype=self.wdtype)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embedding

    def set_input_embeddings(self, value):
        self.decoder.embedding = value

    @can_return_tuple
    @auto_docstring(
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_pldr_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_first_G: Optional[bool] = None
    ) -> SequenceClassifierPLDRLLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BasePLDRModelOutputWithPast = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_pldr_attentions=output_pldr_attentions,
            output_hidden_states=output_hidden_states,
            cache_first_G=cache_first_G
        )
        hidden_states = outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierPLDRLLMOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pldr_attentions=outputs.pldr_attentions
        )


__all__ = [
    "PldrllmForCausalLM",
    "PldrllmModel",
    "PldrllmPreTrainedModel",
    "PldrllmForTokenClassification",
    "PldrllmForQuestionAnswering",
    "PldrllmForSequenceClassification"
]
