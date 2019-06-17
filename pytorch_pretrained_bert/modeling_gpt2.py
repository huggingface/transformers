# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from .file_utils import cached_path, CONFIG_NAME, WEIGHTS_NAME
from .modeling import BertLayerNorm as LayerNorm

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {"gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
                                "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin"}
PRETRAINED_CONFIG_ARCHIVE_MAP = {"gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json",
                                 "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-config.json"}

def load_tf_weights_in_gpt2(model, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'w' or l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'wpe' or l[0] == 'wte':
                pointer = getattr(pointer, l[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GPT2Config(object):
    """Configuration class to store the configuration of a `GPT2Model`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_special=0,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        predict_special_tokens=True
    ):
        """Constructs GPT2Config.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `GPT2Model` or a configuration json file.
            n_special: The number of special tokens to learn during fine-tuning ('[SEP]', '[CLF]', ...)
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            predict_special_tokens: should we predict special tokens (when the model has a LM head)
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_special = n_special
            self.n_ctx = n_ctx
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
            self.predict_special_tokens = predict_special_tokens
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @property
    def total_tokens_embeddings(self):
        return self.vocab_size + self.n_special

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPT2Config` from a Python dictionary of parameters."""
        config = GPT2Config(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPT2Config` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, output_attentions=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = output_attentions
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if self.output_attentions:
            return w, torch.matmul(w, v)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        if self.output_attentions:
            attentions, a = a
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        if self.output_attentions:
            return attentions, a, present
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, output_attentions=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.output_attentions = output_attentions
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale, output_attentions)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        output_attn = self.attn(self.ln_1(x), layer_past=layer_past)
        if self.output_attentions:
            attentions, a, present = output_attn
        else:
            a, present = output_attn
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        if self.output_attentions:
            return attentions, x, present
        return x, present


class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.predict_special_tokens = config.predict_special_tokens
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights, predict_special_tokens=True):
        self.predict_special_tokens = predict_special_tokens
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        if not self.predict_special_tokens:
            lm_logits = lm_logits[..., :self.vocab_size]
        return lm_logits


class GPT2MultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(GPT2MultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.linear = nn.Linear(config.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, num_choices)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        # (bsz, num_choices, 1, hidden_size)
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # (bsz, num_choices, hidden_size)
        multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        # (bsz, num_choices)
        return multiple_choice_logits


class GPT2PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a GPT2PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `gpt2`
                - a path or url to a pretrained model archive containing:
                    . `gpt2_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a GPT2Model instance
                - a path or url to a pretrained model archive containing:
                    . `gpt2_config.json` a configuration file for the model
                    . a TensorFlow checkpoint with trained weights
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific GPT2 class
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)
        num_special_tokens = kwargs.get('num_special_tokens', None)
        kwargs.pop('num_special_tokens', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path, ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), pretrained_model_name_or_path,
                    archive_file, config_file
                )
            )
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info("loading weights file {}".format(archive_file))
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))
        # Load config
        config = GPT2Config.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            # Directly load from a TensorFlow checkpoint (stored as NumPy array)
            return load_tf_weights_in_gpt2(model, resolved_archive_file)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_model = model
        if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
            start_model = model.transformer
        load(start_model, prefix="")

        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )

        # Add additional embeddings for special tokens if needed
        # This step also make sure we are still sharing the output and input embeddings after loading weights
        model.set_num_special_tokens(num_special_tokens if num_special_tokens is not None else config.n_special)
        return model


class GPT2Model(GPT2PreTrainedModel):
    """OpenAI GPT-2 model ("Language Models are Unsupervised Multitask Learners").

    GPT-2 use a single embedding matrix to store the word and special embeddings.
    Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
    Special tokens need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    The embeddings are ordered as follow in the token embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1]                  ______________________

    where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
        total_tokens_embeddings = config.vocab_size + config.n_special
    You should use the associate indices to index the embeddings.

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs a tuple consisting of:
        `hidden_states`: the encoded-hidden-states at the top of the model
            as a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
            (or more generally [d_1, ..., d_n, hidden_size] were d_1 ... d_n are the dimension of input_ids)
        `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
            torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2Model(config)
    hidden_states, presents = model(input_ids)
    ```
    """

    def __init__(self, config, output_attentions=False):
        super(GPT2Model, self).__init__(config)
        self.output_attentions = output_attentions
        self.wte = nn.Embedding(config.total_tokens_embeddings, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        block = Block(config.n_ctx, config, scale=True, output_attentions=output_attentions)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        if self.config.n_special == num_special_tokens:
            return
        # Update config
        self.config.n_special = num_special_tokens
        # Build new embeddings and initialize all new embeddings (in particular the special tokens)
        old_embed = self.wte
        self.wte = nn.Embedding(self.config.total_tokens_embeddings, self.config.n_embd)
        self.wte.to(old_embed.weight.device)
        self.init_weights(self.wte)
        # Copy word embeddings from the previous weights
        self.wte.weight.data[:self.config.vocab_size, :] = old_embed.weight.data[:self.config.vocab_size, :]

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        presents = []
        all_attentions = []
        for block, layer_past in zip(self.h, past):
            if self.output_attentions:
                attentions, hidden_states, present = block(hidden_states, layer_past)
                all_attentions.append(attentions)
            else:
                hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        if self.output_attentions:
            return all_attentions, hidden_states.view(*output_shape), presents
        return hidden_states.view(*output_shape), presents


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` is not `None`:
            Outputs the language modeling loss.
        else a tuple:
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, config.vocab_size]
                (or more generally [d_1, ..., d_n, config.vocab_size] were d_1 ... d_n are the dimension of input_ids)
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2LMHeadModel(config)
    lm_logits, presents = model(input_ids)
    ```
    """

    def __init__(self, config, output_attentions=False):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config, output_attentions=output_attentions)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens, predict_special_tokens=True):
        """ Update input and output embeddings with new embedding matrice
            Make sure we are sharing the embeddings
        """
        self.config.predict_special_tokens = self.transformer.config.predict_special_tokens = predict_special_tokens
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight, predict_special_tokens=predict_special_tokens)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        transformer_output = self.transformer(input_ids, position_ids, token_type_ids, past)
        if self.transformer.output_attentions:
            all_attentions, hidden_states, presents = transformer_output
        else:
            hidden_states, presents = transformer_output
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            return loss
        if self.transformer.output_attentions:
            return all_attentions, lm_logits, presents
        return lm_logits, presents


class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    """OpenAI GPT-2 model with a Language Modeling and a Multiple Choice head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
            indices selected in the range [0, config.vocab_size[
        `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
            which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with indices selected in [-1, 0, ..., config.vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., config.vocab_size]
        `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` and `multiple_choice_labels` are not `None`:
            Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
        else: a tuple with
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, config.vocab_size]
            `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
    mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2DoubleHeadsModel(config)
    lm_logits, multiple_choice_logits, presents = model(input_ids, mc_token_ids)
    ```
    """

    def __init__(self, config, output_attentions=False):
        super(GPT2DoubleHeadsModel, self).__init__(config)
        self.transformer = GPT2Model(config, output_attentions=output_attentions)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.multiple_choice_head = GPT2MultipleChoiceHead(config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens, predict_special_tokens=True):
        """ Update input and output embeddings with new embedding matrice
            Make sure we are sharing the embeddings
        """
        self.config.predict_special_tokens = self.transformer.config.predict_special_tokens = predict_special_tokens
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight, predict_special_tokens=predict_special_tokens)

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None, past=None):
        transformer_output = self.transformer(input_ids, position_ids, token_type_ids, past)
        if self.transformer.output_attentions:
            all_attentions, hidden_states, presents = transformer_output
        else:
            hidden_states, presents = transformer_output
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if losses:
            return losses
        if self.transformer.output_attentions:
            return all_attentions, lm_logits, mc_logits, presents
        return lm_logits, mc_logits, presents
