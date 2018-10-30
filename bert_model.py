"""
A PyTorch implementation of Google's BERT Model.

From "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
By Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
Link: http://arxiv.org/abs/1810.04805

Adapted from HuggingFace's OpenAI PyTorch code and its adaptation by AllenNLP.
"""

# pylint: disable=invalid-name,arguments-differ
from typing import NamedTuple, List
import copy
import io
import json
import logging
import math
import pathlib
import re
import tarfile

import numpy as np
import torch
from torch.nn import Parameter

# pylint: disable=line-too-long
_PARAMETER_NAMES = ["model/we:0",
                    "model/h0/attn/c_attn/w:0", "model/h0/attn/c_attn/b:0", "model/h0/attn/c_proj/w:0",
                    "model/h0/attn/c_proj/b:0", "model/h0/ln_1/g:0", "model/h0/ln_1/b:0", "model/h0/mlp/c_fc/w:0",
                    "model/h0/mlp/c_fc/b:0", "model/h0/mlp/c_proj/w:0", "model/h0/mlp/c_proj/b:0", "model/h0/ln_2/g:0",
                    "model/h0/ln_2/b:0", "model/h1/attn/c_attn/w:0", "model/h1/attn/c_attn/b:0", "model/h1/attn/c_proj/w:0",
                    "model/h1/attn/c_proj/b:0", "model/h1/ln_1/g:0", "model/h1/ln_1/b:0", "model/h1/mlp/c_fc/w:0",
                    "model/h1/mlp/c_fc/b:0", "model/h1/mlp/c_proj/w:0", "model/h1/mlp/c_proj/b:0", "model/h1/ln_2/g:0",
                    "model/h1/ln_2/b:0", "model/h2/attn/c_attn/w:0", "model/h2/attn/c_attn/b:0", "model/h2/attn/c_proj/w:0",
                    "model/h2/attn/c_proj/b:0", "model/h2/ln_1/g:0", "model/h2/ln_1/b:0", "model/h2/mlp/c_fc/w:0",
                    "model/h2/mlp/c_fc/b:0", "model/h2/mlp/c_proj/w:0", "model/h2/mlp/c_proj/b:0", "model/h2/ln_2/g:0",
                    "model/h2/ln_2/b:0", "model/h3/attn/c_attn/w:0", "model/h3/attn/c_attn/b:0", "model/h3/attn/c_proj/w:0",
                    "model/h3/attn/c_proj/b:0", "model/h3/ln_1/g:0", "model/h3/ln_1/b:0", "model/h3/mlp/c_fc/w:0",
                    "model/h3/mlp/c_fc/b:0", "model/h3/mlp/c_proj/w:0", "model/h3/mlp/c_proj/b:0", "model/h3/ln_2/g:0",
                    "model/h3/ln_2/b:0", "model/h4/attn/c_attn/w:0", "model/h4/attn/c_attn/b:0", "model/h4/attn/c_proj/w:0",
                    "model/h4/attn/c_proj/b:0", "model/h4/ln_1/g:0", "model/h4/ln_1/b:0", "model/h4/mlp/c_fc/w:0",
                    "model/h4/mlp/c_fc/b:0", "model/h4/mlp/c_proj/w:0", "model/h4/mlp/c_proj/b:0", "model/h4/ln_2/g:0",
                    "model/h4/ln_2/b:0", "model/h5/attn/c_attn/w:0", "model/h5/attn/c_attn/b:0", "model/h5/attn/c_proj/w:0",
                    "model/h5/attn/c_proj/b:0", "model/h5/ln_1/g:0", "model/h5/ln_1/b:0", "model/h5/mlp/c_fc/w:0",
                    "model/h5/mlp/c_fc/b:0", "model/h5/mlp/c_proj/w:0", "model/h5/mlp/c_proj/b:0", "model/h5/ln_2/g:0",
                    "model/h5/ln_2/b:0", "model/h6/attn/c_attn/w:0", "model/h6/attn/c_attn/b:0", "model/h6/attn/c_proj/w:0",
                    "model/h6/attn/c_proj/b:0", "model/h6/ln_1/g:0", "model/h6/ln_1/b:0", "model/h6/mlp/c_fc/w:0",
                    "model/h6/mlp/c_fc/b:0", "model/h6/mlp/c_proj/w:0", "model/h6/mlp/c_proj/b:0", "model/h6/ln_2/g:0",
                    "model/h6/ln_2/b:0", "model/h7/attn/c_attn/w:0", "model/h7/attn/c_attn/b:0", "model/h7/attn/c_proj/w:0",
                    "model/h7/attn/c_proj/b:0", "model/h7/ln_1/g:0", "model/h7/ln_1/b:0", "model/h7/mlp/c_fc/w:0",
                    "model/h7/mlp/c_fc/b:0", "model/h7/mlp/c_proj/w:0", "model/h7/mlp/c_proj/b:0", "model/h7/ln_2/g:0",
                    "model/h7/ln_2/b:0", "model/h8/attn/c_attn/w:0", "model/h8/attn/c_attn/b:0", "model/h8/attn/c_proj/w:0",
                    "model/h8/attn/c_proj/b:0", "model/h8/ln_1/g:0", "model/h8/ln_1/b:0", "model/h8/mlp/c_fc/w:0",
                    "model/h8/mlp/c_fc/b:0", "model/h8/mlp/c_proj/w:0", "model/h8/mlp/c_proj/b:0", "model/h8/ln_2/g:0",
                    "model/h8/ln_2/b:0", "model/h9/attn/c_attn/w:0", "model/h9/attn/c_attn/b:0", "model/h9/attn/c_proj/w:0",
                    "model/h9/attn/c_proj/b:0", "model/h9/ln_1/g:0", "model/h9/ln_1/b:0", "model/h9/mlp/c_fc/w:0",
                    "model/h9/mlp/c_fc/b:0", "model/h9/mlp/c_proj/w:0", "model/h9/mlp/c_proj/b:0", "model/h9/ln_2/g:0",
                    "model/h9/ln_2/b:0", "model/h10/attn/c_attn/w:0", "model/h10/attn/c_attn/b:0", "model/h10/attn/c_proj/w:0",
                    "model/h10/attn/c_proj/b:0", "model/h10/ln_1/g:0", "model/h10/ln_1/b:0", "model/h10/mlp/c_fc/w:0",
                    "model/h10/mlp/c_fc/b:0", "model/h10/mlp/c_proj/w:0", "model/h10/mlp/c_proj/b:0", "model/h10/ln_2/g:0",
                    "model/h10/ln_2/b:0", "model/h11/attn/c_attn/w:0", "model/h11/attn/c_attn/b:0", "model/h11/attn/c_proj/w:0",
                    "model/h11/attn/c_proj/b:0", "model/h11/ln_1/g:0", "model/h11/ln_1/b:0", "model/h11/mlp/c_fc/w:0",
                    "model/h11/mlp/c_fc/b:0", "model/h11/mlp/c_proj/w:0", "model/h11/mlp/c_proj/b:0", "model/h11/ln_2/g:0",
                    "model/h11/ln_2/b:0", "model/clf/w:0", "model/clf/b:0"]
# pylint: enable=line-too-long

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class BERTConfig(NamedTuple):
    """
    BERT's hyper-parameters
    """
    embedding_dim: int = 768
    num_heads: int = 12
    dropout: float = 0.1

class LayerNorm(torch.nn.Module):
    """
    A layernorm module in the Tensorflow style (with the epsilon inside the square root).
    """

    def __init__(self, n_state, e=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(n_state))
        self.b = torch.nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(torch.nn.Module):
    """
    A batched linear layer using torch.addmm
    """
    def __init__(self, nf: int, rf: int, nx: int) -> None:
        super().__init__()
        self.rf = rf
        self.nf = nf
        w = torch.empty(nx, nf)
        torch.nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)
        self.b = Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
        x = x.view(*size_out)
        return x

class Attention(torch.nn.Module):
    """
    A self-attention layer comprising a sequence of:
        - a linear layer: instance of the `Conv1D` class,
        - spliting the inputs in key, value, query tensors (x.split),
        - reshaping key, value, query tensors according to the number of head (self.split_heads)
        - appying self attention (self._attn)
        - merging back the heads results (self.merge_heads)
        - a linear layer: instance of the `Conv1D` class,
        - a dropout layer: instance of `torch.nn.Dropout` class.

    See above for the details of Conv1D.
    """
    def __init__(self,
                 nx: int,
                 n_ctx: int,
                 config: BERTConfig,
                 scale: bool = False) -> None:
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.num_heads == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.num_heads
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w = torch.nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x: torch.Tensor):
        # pylint: disable=no-self-use
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x: torch.Tensor, k: bool = False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(torch.nn.Module):
    """
    A multi-layer perceptron layer comprising a sequence of:
        - a linear layer: instance of the `Conv1D` class,
        - an activation function: the `gelu` function,
        - another linear layer: instance of the `Conv1D` class,
        - a dropout layer: instance of `torch.nn.Dropout` class.

    See above for the details of Conv1D and the gelu function.
    """
    def __init__(self, n_state: int, config: BERTConfig) -> None:  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.c_fc = Conv1D(n_state, 1, config.embedding_dim)
        self.c_proj = Conv1D(config.embedding_dim, 1, n_state)
        self.act = gelu
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(torch.nn.Module):
    """
    A Transformer Block comprising a sequence of:
        - a self-attention layer: instance of the `Attention` class,
        - a Layer Normalization layer: instance of the `LayerNorm` class,
        - a Multi-layer perceptron layer: instance of the `MLP` class,
        - another Layer Normalization layer: instance of the `LayerNorm` class.

    See above for the details of these classes.
    """
    def __init__(self,
                 n_ctx: int,
                 config: BERTConfig,
                 scale: bool = False) -> None:
        super().__init__()
        nx = config.embedding_dim
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h

class BERT(torch.nn.Module):
    """
    Google's BERT Model.
    Default parameters are the ones for Google's pretrained model.

    Parameters
    ----------
    vocab_size: ``int`` (optional, default: 40478)
        The size of the vocabulary (number of byte pair embeddings)
        excluding the n_special embeddings (if any), and the positional embeddings.
    n_ctx: ``int`` (optional, default: 512)
        The number of positional encodings to use for evaluation.
    embedding_dim: ``int`` (optional, default: 768)
        The dimension of the output embeddings.
    num_heads: ``int`` (optional, default: 12)
        How many "heads" the attention has.
    num_layers: ``int`` (optional, default: 12)
        How many layers of "blocks" the transformer has.
    dropout_probability: ``float`` (optional, default: 0.1)
        Dropout for all layers.
    model_path: ``str`` (optional, default: ``None``)
        A tar.gz file containing serialized model weights. If supplied,
        the weights will be loaded from that file.
    requires_grad: ``bool`` (optional, default: ``False``)
        If true, the transformer will be fine-tuneable.
    n_special: ``int`` (optional, default: ``-1``)
        The number of special tokens added to the byte pair vocabulary
    """
    def __init__(self,
                 vocab_size: int = 40478,
                 n_ctx: int = 512,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 dropout_probability: float = 0.1,
                 model_path: str = None,
                 requires_grad: bool = False,
                 n_special: int = -1) -> None:
        super().__init__()

        config = BERTConfig(
                embedding_dim,
                num_heads,
                embedding_dropout_probability,
                attention_dropout_probability,
                residual_dropout_probability,
                activation_function,
        )

        # the embedding size is vocab_size + n_special embeddings + n_ctx
        embedding_size = vocab_size + max(n_special, 0) + n_ctx
        self.vocab_size = embedding_size
        self.n_ctx = n_ctx
        self.n_special = n_special

        self.num_output_layers = 1 + num_layers

        self.embed = torch.nn.Embedding(embedding_size, embedding_dim)
        self.drop = torch.nn.Dropout(embedding_dropout_probability)

        block = Block(n_ctx, config, scale=True)
        self.h = torch.nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        self.decoder = torch.nn.Linear(embedding_dim, embedding_size, bias=False)
        self.decoder.weight = self.embed.weight  # Tied weights
        # To reproduce the noise_shape parameter of TF implementation

        torch.nn.init.normal_(self.embed.weight, std=0.02)

        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

        if model_path:
            self.load_weights(model_path, n_special=n_special, n_ctx=n_ctx)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        #x = x.view(-1, x.size(2), x.size(3))

        # x is (batch_size, sequence_length) tensor of byte-pair ids

        # e is (batch_size, sequence_length, 2, embedding_dim) tensor of embeddings
        e = self.embed(x)

        # h is (batch_size, sequence_length, embedding_dim)
        h = e.sum(dim=2)

        all_layers = [h]
        for block in self.h:
            h = block(h)
            all_layers.append(h)

        # result is list of (batch_size, sequence_length, embedding_dim)
        return all_layers

    def load_weights(self,
                     bert_model_path: str,
                     n_ctx: int = -1,
                     n_special: int = -1,
                     n_transfer: int = 12,
                     n_embd: int = 768,
                     names: List[str] = _PARAMETER_NAMES) -> None:
        # pylint: disable=dangerous-default-value

        logger.info(f"loading weights from {bert_model_path}")
        # if `file_path` is a URL, redirect to the cache

        with tarfile.open(bert_model_path) as tmp:
            num_params_files = len([member for member in tmp.getmembers() if member.name.endswith('.npy')])
            shapesfile = tmp.extractfile('model/params_shapes.json')
            if shapesfile:
                shapes = json.loads(shapesfile.read())
            else:
                raise ConfigurationError("unable to find model/params_shapes.json in the archive")

            # numpy can't read from a tarfile directly, so we need a workaround
            # https://github.com/numpy/numpy/issues/7989#issuecomment-341656702
            init_params: List[np.ndarray] = []
            for n in range(num_params_files):
                array_file = io.BytesIO()
                array_file.write(tmp.extractfile(f'model/params_{n}.npy').read())
                array_file.seek(0)
                # each np.load is a (11653478,) numpy array
                init_params.append(np.load(array_file))

        # init_params is a list of 10 arrays of size (11653578,)
        # shapes are [[512, 768], [40478, 768], [1, 768, 2304], [2304], ...  # 146 elts
        # products are [512 * 768, 40478 * 768, ...]
        # offsets is [512 * 768, 512 * 768 + 40478 * 768, ...]
        offsets = np.cumsum([np.prod(shape) for shape in shapes])

        # split into the 146 subarrays corresponding to shapes
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]

        # reshape
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]

        # truncate if necessary
        if n_ctx > 0:
            # positional embeddings?
            # init_params[0] is (512, 768) = (max_chars, embedding_dim)
            init_params[0] = init_params[0][:n_ctx]

        # combine init_params[1] and init_params[0]
        if n_special > 0:
            # init_params[1] is (40478, 768)
            # special is (n_special, 768)
            # init_params[0] is (512, 768)
            # result is (40990 + n_special, 768)
            init_params[0] = np.concatenate(
                    [init_params[1],
                     (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
                     init_params[0]],
                    0
            )
        else:
            # result is (40990, 768)
            init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
        del init_params[1]

        # number of dimensions to transfer, 12 per layer, plus one extra
        if n_transfer == -1:
            n_transfer = 0
        else:
            n_transfer = 1 + n_transfer * 12

        # squeeze?
        init_params = [arr.squeeze() for arr in init_params]

        # embedding.weight is (vocab_size, embedding_dim)
        # make sure init_params[0] has the same shape
        try:
            assert self.embed.weight.shape == init_params[0].shape
        except AssertionError as e:
            e.args += (self.embed.weight.shape, init_params[0].shape)
            raise

        # and then assign it
        self.embed.weight.data = torch.from_numpy(init_params[0])
        self.decoder.weight = self.embed.weight

        # for each (name, array) pair to transfer over
        for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
                                            # "model/h0/attn/c_attn/w:0"
            name = name[6:]                 # "h0/attn/c_attn/w:0"
            assert name[-2:] == ":0"
            name = name[:-2]                # "h0/attn/c_attn/w"
            name_parts = name.split('/')    # ['h0', 'attn', 'c_attn', 'w']

            pointer = self
            for m_name in name_parts:
                if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                    l = re.split(r'(\d+)', m_name)   # ['h', '0', '']
                else:
                    l = [m_name]                     # ['attn']
                pointer = getattr(pointer, l[0])
                if len(l) >= 2:
                    num = int(l[1])
                    pointer = pointer[num]
            try:
                assert pointer.shape == ip.shape
            except AssertionError as e:
                e.args += (pointer.shape, ip.shape)
                raise

            pointer.data = torch.from_numpy(ip)  # pylint: disable=attribute-defined-outside-init

    def dump_weights(self,
                     output_dir: str,
                     num_pieces: int = 10) -> None:
        output_path = pathlib.Path(output_dir) / 'model'
        output_path.mkdir(exist_ok=True, parents=True)  # pylint: disable=no-member

        named_parameters = list(self.named_parameters())

        # embedding weights get special treatment
        _, array = named_parameters[0]
        num_bpe = self.vocab_size - self.n_ctx
        byte_pair_embeddings = array[:num_bpe]
        positional_embeddings = array[num_bpe:]

        arrays = [positional_embeddings.numpy().ravel(), byte_pair_embeddings.numpy().ravel()]
        shapes = [positional_embeddings.shape, byte_pair_embeddings.shape]
        names = ["model/we:0"]

        for param_name, tensor in named_parameters[1:]:
            param_name = f'h{param_name}'            # 'h0.attn.c_attn.w'
            parts = param_name.split(".")            # ['h0', 'attn', 'c_attn', 'w']
            name = "model/" + '/'.join(parts) + ':0' # 'model/h0/attn/c_attn/w:0'
            array = tensor.numpy().ravel()

            arrays.append(array)
            shapes.append(list(tensor.shape))
            names.append(name)

        # write out the arrays
        big_array = np.concatenate(arrays)
        total_size = len(big_array)
        batch_size = math.ceil(total_size / num_pieces)

        for i in range(num_pieces):
            filename = output_path / f"params_{i}.npy"
            start = i * batch_size
            end = start + batch_size
            subarray = big_array[start:end]

            np.save(filename, subarray)

        # write out the shapes
        with open(output_path / 'params_shapes.json', 'w') as shapes_file:
            json.dump(shapes, shapes_file)
