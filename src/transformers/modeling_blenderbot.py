import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
from configuration_blenderbot import BlenderbotConfig
import numpy as np
import warnings

BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_MAP = {"blenderbot": "https://cdn.huggingface.co/"}


class BlenderbotMultiHeadAttention(nn.Module):
    """
    multiheadattention from vaswani paper
    """

    def __init__(self, n_heads, dim, dropout=0.0):
        """

        :param n_heads: number of attention heads
        :param dim: dimension
        :param dropout: attention dropout
        """
        super(BlenderbotMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.att_dropout = nn.Dropout(p=dropout)
        self.k_lin = nn.Linear(dim, dim)
        self.q_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

        # weight initialization will be done using the pretrained model

    def forward(self, query, key=None, value=None, mask=None, incremental_state=None, static_key_val=False):
        """

        :param query: attention query: torch.tensor
        :param key: attention key: torch.tensor, default to None
        :param value: attention value: torch.tensor, default to None
        :param mask: allow attention or not: torch.tensor, default to None
        :param incremental_state: dictionary with values representing the previous states of the key, value, and mask
        :param static_key_val: Boolean, True if the key and value are held constant during decoding
        :return: (attention tensor, new incremental_state)
        """
        assert mask is not None, "mask is None, please specify a mask"
        batch_size, query_len, dim = query.size()
        assert (dim == self.dim), "Dimensions does not match, query dimension is {} and configured dimension {}".format(dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        #print(query.size())
        if key is None and value is None:
            key = value = query
            _, key_len,dim=query.size()
        
        elif value is None:
            value = key
        assert key is not None
        #print(key.size())
        _, key_len, dim = key.size()
        q = reshape_head(self.q_lin(query), n_heads, dim_per_head)
        k = reshape_head(self.k_lin(key), n_heads, dim_per_head)
        v = reshape_head(self.v_lin(value), n_heads, dim_per_head)
        if incremental_state is None:
            incremental_state = {}
        if 'prev_value' in incremental_state:
            prev_value = incremental_state['prev_value'].view(batch_size * n_heads, -1, dim_per_head)
            if static_key_val:
                v = prev_value
            else:
                v = torch.cat([prev_value, v], dim=1)
        if 'prev_key' in incremental_state:
            prev_key = incremental_state['prev_key'].view(batch_size * n_heads, -1, dim_per_head)
            if static_key_val:
                k = prev_key
            else:
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_mask' in incremental_state:
            if static_key_val:
                mask = incremental_state['prev_mask']
            else:
                mask = torch.cat([incremental_state['prev_mask'], mask], dim=2)  # key_len dimension
        scale = math.sqrt(dim_per_head)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        key_len = k.size(1)
        attn_mask = (mask == 0).view(batch_size, 1, -1, key_len).repeat(1, n_heads, 1, 1) \
            .expand(batch_size, n_heads, query_len, key_len).view(batch_size * n_heads, query_len, key_len)
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill(attn_mask, neginf(dot_prod.dtype))
        attn_weight = F.softmax(dot_prod, dim=-1, dtype=torch.float).type_as(query)
        attn_weight = self.att_dropout(attn_weight)
        attention = attn_weight.bmm(v)
        attention = (attention.type_as(query)).view(batch_size, n_heads, query_len, dim_per_head).transpose(1, 2) \
            .contiguous().view(batch_size, query_len, dim)
        output = self.out_lin(attention)
        new_incremental_state = {
            'prev_key': k.view(batch_size, n_heads, -1, dim_per_head),
            'prev_value': v.view(batch_size, n_heads, -1, dim_per_head),
            'prev_mask': mask
        }
        return  output, new_incremental_state


class BlenderbotFFN(nn.Module):
    """
    implements the FFN
    """

    def __init__(self, dim, hidden_dim, relu_dropout=0.0, activation='relu'):
        super(BlenderbotFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.non_linear = F.relu
        elif activation == 'gelu':
            self.non_linear = gelu
        else:
            raise ValueError('Not defined activation function')
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)

        # weight initialization will be done using the pretrained model

        # # initialise bias to zeros
        # if self.lin1.bias is not None:
        #     nn.init.zeros_(self.lin1.bias)
        # if self.lin2.bias is not None:
        #     nn.init.zeros_(self.lin2.bias)

    def forward(self, input):
        input = self.lin1(input)
        input = self.non_linear(input)
        input = self.relu_dropout(input)
        input = self.lin2(input)
        return input


class BlenderbotEncoderLayer(nn.Module):
    """
    implemments a single layer of the encoder
    """

    def __init__(self, n_heads, embedding_size, ffn_size, 
                 relu_dropout=0.0, attention_dropout=0.0, activation='relu', dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.ffn_size = ffn_size
        self.embedding_size = embedding_size
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = LayerNorm(self.embedding_size, eps=layer_norm_eps)
        self.norm2 = LayerNorm(self.embedding_size, eps=layer_norm_eps)
        self.attention = BlenderbotMultiHeadAttention(self.n_heads, self.embedding_size, dropout=attention_dropout)
        self.ffn = BlenderbotFFN(self.embedding_size, ffn_size, relu_dropout=relu_dropout, activation=self.activation)

    def forward(self, input_tensor, mask):
        tensor_copy = input_tensor
        attention, _ = self.attention(tensor_copy, mask=mask)
        input_tensor = tensor_copy + self.dropout(attention)
        input_tensor = self.norm1(input_tensor)
        tensor_copy = input_tensor
        input_tensor = tensor_copy + self.dropout(self.ffn(input_tensor))
        input_tensor = self.norm2(input_tensor)
        input_tensor *= mask.unsqueeze(-1).type_as(input_tensor)
        return input_tensor


class BlenderbotEncoder(nn.Module):
    """
    implements the encoder model of the blender bot generative model
    """

    def __init__(self,
                 n_heads,
                 n_layers,
                 embedding_size,
                 ffn_size,
                 vocabulary_size,
                 embedding=None,
                 dropout=0.0,
                 relu_dropout=0.0,
                 attention_dropout=0.0,
                 padding_idx=0,
                 embedding_scale=False,
                 learn_positional_embeddings=False,
                 n_positions=1024,
                 reduction_type='mean',
                 activation='relu',
                 n_segments=0,
                 output_scaling=1.0):
        super(BlenderbotEncoder, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.vocabulary_size = vocabulary_size
        self.dropout = nn.Dropout(p=dropout)
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        self.embedding_scale = embedding_scale
        self.n_segments = n_segments
        self.n_positions = n_positions
        self.output_dim = embedding_size

        if embedding is not None:
            assert (self.embedding_size is None or self.embedding_size == embedding.weight.shape[1]), \
                "Embedding dimension must match embedding size"
            self.embeddings = embedding
        else:
            raise AssertionError("None embedding is not allowed")
        assert (self.embedding_size % self.n_heads == 0), "Embedding size should be a multiple of n_heads"
        assert self.padding_idx is not None
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size, padding_idx=self.padding_idx)
        # weight initialization will be done using the pretrained model
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not learn_positional_embeddings:
            positional_coding(self.n_positions, self.embedding_size, self.position_embeddings.weight)
        else:
            # initialize position embeddings using a normal distribution
            nn.init.normal_(self.position_embeddings.weight, 0, self.embedding_size ** (-0.5))
        if self.n_segments >= 1:
            self.segments_embeddinggs = nn.Embedding(self.n_segments, self.embedding_size)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(BlenderbotEncoderLayer(self.n_heads, self.embedding_size, self.ffn_size,
                                                      relu_dropout=relu_dropout, attention_dropout=attention_dropout, activation=activation, dropout=dropout))
        self.output_scaling = output_scaling

    def forward(self, input, positions=None, segments=None):
        """

        :param input: inputs ids tensor
        :param positions: position encoddings
        :param segments: add segments as extra embedding features
        :return:
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        input_tensor = self.embedding(input)
        if self.embedding_scale:
            input_tensor *= math.sqrt(self.embedding_size)
        positions_embeddings = self.position_embeddings(positions).expand_as(input_tensor)
        input_tensor += positions_embeddings
        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)
            input_tensor += self.segments_embeddinggs(segments)
        input_tensor = self.dropout(input_tensor)
        input_tensor *= mask.unsqueeze(-1).type_as(input_tensor)

        # To Do: add model parallel option here

        for i in range(self.n_layers):
            input_tensor = self.layers[i](input_tensor, mask)
        input_tensor *= self.output_scaling

        # make the reduction of the tensor. Reduction can be first ([cls] token), mean (mean over all tokens vectors), max (maximun token vector s summation)
        if self.reduction_type == 'mean':
            div = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(input_tensor)
            output_tensor = input_tensor.sum(dim=1) / div
        elif self.reduction_type == 'first':
            output_tensor = input_tensor[:, 0, :]
        elif self.reduction_type == 'max':
            output_tensor = input_tensor.max(dim=1)[0]
        elif self.reduction_type is None or 'none' in self.reduction_type.lower():
            output_tensor = input_tensor
        else:
            raise ValueError("{} is not a known reduction_type value".format(self.reduction_type))
        return output_tensor.unsqueeze(1), mask


class BlenderbotDecoderLayer(nn.Module):
    """
    implements a layer of the decoder model
    Decoder layers are similar to encoder layers except that:
        Self-attention is limited in a casaul (auto-regressive) manner.
        Attend over all of the encoder states
    """

    def __init__(self, n_heads, embedding_size, ffn_size, activation='relu', dropout=0.0, attention_dropout=0.0,
                 relu_dropout=0.0, layer_norm_eps=1e-5
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.self_attention = BlenderbotMultiHeadAttention(self.n_heads, self.embedding_size, dropout=attention_dropout)
        self.norm1 = LayerNorm(self.embedding_size, eps=layer_norm_eps)
        self.encoder_attention = BlenderbotMultiHeadAttention(self.n_heads, self.embedding_size, dropout=attention_dropout)
        self.norm2 = LayerNorm(self.embedding_size, eps=layer_norm_eps)
        self.ffn = BlenderbotFFN(self.embedding_size, self.ffn_size, relu_dropout=relu_dropout, activation=self.activation)
        self.norm3 = LayerNorm(self.embedding_size, eps=layer_norm_eps)

    def forward(self, input, encoder_output, encoder_mask, incremental_state=None):
        """

        :param input:  current decoder inputs
        :param encoder_output: output from the encoder
        :param encoder_mask:  mask from the encoder
        :param incremental_state:  incremental state (values for self and encoder attention
        :return:  decoder final output and new incremental state
        """
        if incremental_state is None:
            incremental_state = {}
        decoder_mask = create_selfattn_mask(input)
        residual = input
        input, final_self_attn_incr_state = self.self_attention(query=input, mask=decoder_mask, incremental_state=incremental_state.get('self_attn'),
                                                                static_key_val=False)
        input = self.dropout(input)
        input += residual
        input = self.norm1(input)
        residual = input
        input, final_enc_attn_incr_state = self.encoder_attention(query=input, key=encoder_output, value=encoder_output,
                                                                  mask=encoder_mask, incremental_state=incremental_state.get('encoder_attn'),
                                                                  static_key_val=True)
        input = self.dropout(input)
        input += residual
        input = self.norm2(input)
        residual = input
        input = self.ffn(input)
        input = self.dropout(input)
        input += residual
        input = self.norm3(input)
        new_incremental_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_enc_attn_incr_state
        }
        return input, new_incremental_state


class BlenderbotDecoder(nn.Module):
    """
    implements the decoder part of the blender bot
    """
    def __init__(self, n_heads, embedding_size, ffn_size, n_layers, vocabulary_size, embedding=None,
                 attention_dropout=0.0, dropout=0.0, relu_dropout=0.0, embedding_scale=True,
                 learn_positional_embeddings=False, padding_idx=0, n_positions = 1024, n_segments=0,
                 activation='relu'):
        """

        :param n_heads: number of multi attention heads in the decoder
        :param embedding_size: embedding size
        :param ffn_size:  size of hidden layers in the FFN
        :param n_layers:  number of layers
        :param vocabulary_size: size of the vocabulary
        :param embedding:  embedding matrix
        :param attention_dropout: dropout after the multihead attention
        :param dropout:  dropout around embeddings
        :param relu_dropout: dropout after Relu
        :param embedding_scale:  scale embeddings
        :param learn_positional_encodings: If off, sinusoidal embeddings are used. If on, position embeddings are learned from scratch.
        :param padding_idx:padding index
        :param n_positions:size of the position embeddings
        :param n_segments:
        :param activation: activation function to use
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_size = ffn_size
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_scale = embedding_scale
        self.n_positions = n_positions
        self.n_segment = n_segments
        self.embeddings = embedding
        assert self.embedding_size % self.n_heads == 0, "embedding size must be a multiple of n_heads"
        self.positional_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not learn_positional_embeddings:
            positional_coding(self.n_positions, self.embedding_size, self.positional_embeddings)
        else:
            nn.init.normal_(self.positional_embeddings.weight, 0, self.embedding_size**(-0.5))
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(BlenderbotDecoderLayer(self.n_heads, self.embedding_size, self.ffn_size,
                                                      activation, dropout, attention_dropout, relu_dropout))

    def forward(self, input, encoder_state, incremental_state=None):
        encoder_output, encoder_mask = encoder_state
        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        if incremental_state:
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incremental_state = {}
        input_tensor = self.embeddings(input)
        if self.embedding_scale:
            input_tensor *= math.sqrt(self.embedding_size)
        input_tensor += self.positional_embeddings(positions).expand_as(input_tensor)
        input_tensor = self.dropout(input_tensor)
        new_incremental_state = {}
        for idx, layer in enumerate(self.layers):
            input_tensor, new_incremental_state[idx] = layer(input_tensor, encoder_output, encoder_mask, incremental_state.get(idx))
        return input_tensor, new_incremental_state


class BlenderbotPretrained(PreTrainedModel):
    """
        An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = BlenderbotConfig
    pretrained_model_archive_map = BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'transformer'

    def _init_weights(self, module):
        """
        initialize weights
        :param module:
        :return:
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


BLENDERBOT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Args:
        config (:class:`~transformers.BlenderbotConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
 
 Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.BlenderbotTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
            The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices 
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
       
"""



class BlenbderbotGeneratorModel(BlenderbotPretrained):
    """
    implements a blenderbot generator model with an encoder and decoder
    """
    def __init__(self, config):
        self.pad_idx = config.pad_idx
        self.start_idx = config.start_idx
        self.end_idx = config.end_idx
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size, self.pad_idx)
        self.n_positions = config.n_positions if config.n_positions > 0 else 1024
        self.encoder = BlenderbotEncoder(
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            embedding_size=config.embedding_size,
            ffn_size=config.ffn_size,
            vocabulary_size=config.vocab_size,
            embedding=self.embeddings,
            dropout=config.dropout,
            relu_dropout=config.relu_dropout,
            attention_dropout=config.attention_dropout,
            padding_idx=self.pad_idx,
            learn_positional_embeddings=config.learn_positional_embeddings,
            n_positions=self.n_positions,
            activation=config.activation,
        )
        self.decoder = BlenderbotDecoder(
            n_heads=config.n_heads,
            embedding_size=config.embedding_size,
            ffn_size=config.ffn_size,
            n_layers=config.n_layers,
            vocabulary_size=config.vocab_size,
            embedding=self.embeddings,
            attention_dropout=config.attention_dropout,
            dropout=config.dropout,
            relu_dropout=config.relu_dropout,
            learn_positional_embeddings=config.learn_positional_embeddings,
            padding_idx=self.pad_idx,
            n_positions=self.n_positions,
            activation=config.activation
        )
        self.init_weights()

    def forward(self, input_ids, decoder_input_ids=None, position_ids=None, token_type_ids=None):
        if input_ids is None:
            raise ValueError("Your input tensor is None, you should specify it")
        encoder_out = self.encoder(input_ids, token_type_ids)
        assert isinstance(encoder_out, tuple)

        def prepare_decoder_inputs(input_ids):
            prev_output_tokens = input_ids.clone()
            index_of_eos = (input_ids.ne(self.end_idx).sum(dim=1) - 1).unsqueeze(-1)
            prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
            prev_output_tokens[:, 1:] = input_ids[:, :-1]
            return prev_output_tokens
        if decoder_input_ids is None:
            decoder_input_ids = prepare_decoder_inputs(input_ids)
        decoder_output = self.decoder(decoder_input_ids, encoder_out)
        assert isinstance(decoder_output, tuple)
        tensor_out, new_inc_state = decoder_output
        output = F.linear(tensor_out, self.embeddings.weight)
        # force the start token  probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output


def reshape_head(tensor, n_heads, dim_per_head):
    batch_size, seq_len, _ = tensor.size()
    tensor = tensor.view(batch_size, seq_len, n_heads, dim_per_head)
    tensor = tensor.transpose(1, 2).contiguous().view(batch_size * n_heads, seq_len, dim_per_head)
    return tensor


def neginf(dtype: torch.dtype):
    """
    Return a representable finite number near -inf for a dtype.(credit to parlai)
    """
    if dtype is torch.float16:
        return -1e20
    else:
        return -65504


def gelu(tensor):
    """
    Compute gelu activation function as defined in Gaussian Error Linear Units (GELUs)
    """
    return 0.5 * tensor * (1.0 + torch.erf(tensor / math.sqrt(2.0)))


def positional_coding(n_pos, dim, out):
    """
    Implements the positional encoding from Vaswani paper
    :param n_pos: number of positional codes
    :param dim: dimension
    :param out: output to store positional code
    :return:

    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)] for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False


def create_selfattn_mask(x):
    bsz = x.size(0)
    time = x.size(1)
    mask = torch.tril(x.new(time, time).fill_(1))
    # broadcast across batch
    mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    return mask
