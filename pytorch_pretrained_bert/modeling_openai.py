import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import collections

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

from .modeling import BertLayerNorm as LayerNorm
from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'openai-gpt': "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt.tar.gz",
}
CONFIG_NAME = 'openai_gpt_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}

class OpenAIGPTConfig(object):
    """Configuration class to store the configuration of a `OpenAIGPTModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file=40478,
                 n_special=0,
                 n_ctx=512,
                 n_embd=768,
                 n_layer=12,
                 n_head=12,
                 afn="gelu",
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 initializer_range=0.02):
        """Constructs OpenAIGPTConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `OpenAIGPTModel` or a configuration json file.
            n_special: The number of special tokens to learn during fine-tuning ('[SEP]', '[CLF]', ...)
            n_ctx: Number of positional embeddings.
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            afn: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.n_special = n_special
            self.n_ctx = n_ctx
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.afn = afn
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @property
    def total_num_embeddings(self):
        return self.vocab_size + self.n_special + self.n_ctx

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `OpenAIGPTConfig` from a Python dictionary of parameters."""
        config = OpenAIGPTConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `OpenAIGPTConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
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

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.weight = Parameter(w)
            self.bias = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
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


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class OpenAIGPTLMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(OpenAIGPTLMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class OpenAIGPTMultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(OpenAIGPTMultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        # self.multiple_choice_token = multiple_choice_token
        self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.linear = nn.Linear(config.n_embd, 1)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, multiple_choice_token_mask):
        # Classification logits
        # hidden_states = hidden_states.view(-1, self.n_embd)
        # multiple_choice_token_mask = multiple_choice_token_mask.view(-1, 1).expand_as(hidden_states)
        multiple_choice_h = hidden_states * multiple_choice_token_mask.unsqueeze(-1)
        multiple_choice_h = multiple_choice_h.sum(dim=-2)
        # flat = x[..., 0].contiguous().view(-1)
        # multiple_choice_h = multiple_choice_h[flat == self.multiple_choice_token, :]
        # multiple_choice_h = multiple_choice_h.view(-1, x.size(1), self.n_embd, 1)
        # # This double transposition is there to replicate the behavior
        # # of the noise_shape argument in the tensorflow
        # # implementation.  For more details, see
        # # https://github.com/huggingface/pytorch-openai-transformer-lm/issues/11
        # multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
        # multiple_choice_h = multiple_choice_h.contiguous().view(-1, self.n_embd)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        return multiple_choice_logits


class OpenAIGPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(OpenAIGPTPreTrainedModel, self).__init__()
        if not isinstance(config, OpenAIGPTConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `OpenAIGPTConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
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

    def set_num_special_tokens(self, num_special_tokens):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name, num_special_tokens=0, state_dict=None, cache_dir=None,
                        *inputs, **kwargs):
        """
        Instantiate a OpenAIGPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `openai-gpt`
                - a path or url to a pretrained model archive containing:
                    . `openai_gpt_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a OpenAIGPTModel instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = OpenAIGPTConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model.transformer if hasattr(model, 'transformer') else model, prefix='')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        # Add additional embeddings for special tokens if needed
        if num_special_tokens != config.n_special:
            model.set_num_special_tokens(num_special_tokens)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model ("Improving Language Understanding by Generative Pre-Training").

    The main implementation difference between BERT and the OpenAI is the use, in OpenAI GPT, of a single embedding matrix
    to store the word, special ([SEP], [CLS]...) and position embeddings.
    The embeddings are ordered as follow in the word embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1,                  ______________________
         config.vocab_size + config.n_special,
         ...                                                        -> position embeddings
         total_num_embeddings - 1]                                  ______________________

    where total_num_embeddings can be obtained as config.total_num_embeddings and is:
        total_num_embeddings = config.vocab_size + config.n_special + config.n_ctx
    You should use the associate indices to index the embeddings.

    The special embeddings ([SEP], [CLS]...) are not pre-trained and need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [config.vocab_size + config.n_special, config.vocab_size + config.n_special + config.n_ctx - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third embedding (the previous two being the word and position embeddings)
            to each token in the sentence.

    Outputs:
        `hidden_states`: the encoded-hidden-states at the top of the model
            as a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
            (or more generally [d_1, ..., d_n, hidden_size] were d_1 ... d_n are the dimension of input_ids)

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTModel(config)
    hidden_states = model(input_ids)
    ```
    """
    def __init__(self, config):
        super(OpenAIGPTModel, self).__init__(config)
        total_embeddings_size = config.vocab_size + config.n_special + config.n_ctx
        self.embed = nn.Embedding(total_embeddings_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])

        self.apply(self.init_weights)
        # nn.init.normal_(self.embed.weight, std=0.02)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice "
        # Update config
        self.config.n_special = num_special_tokens
        # # Build new embeddings and initialize
        old_embed = self.embed
        self.embed = nn.Embedding(self.config.total_num_embeddings, self.config.n_embd)
        # Initialize all new embeddings (in particular the special tokens)
        self.init_weights(self.embed)
        # Copy word and positional embeddings from the previous weights
        self.embed.weight.data[:self.config.vocab_size, :] = old_embed.weight.data[:self.config.vocab_size, :]
        self.embed.weight.data[-self.config.n_ctx:, :] = old_embed.weight.data[-self.config.n_ctx:, :]

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        if position_ids is None:
            start = self.config.vocab_size + self.config.n_special
            end = start + input_ids.size(-1)
            position_ids = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.embed(input_ids)
        position_embeds = self.embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.embed(token_type_ids)
        else:
            token_type_embeds = 0
        # Add the position information to the input embeddings
        # h = e.sum(dim=2)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        for block in self.h:
            hidden_states = block(hidden_states)
        return hidden_states.view(*input_shape, hidden_states.size(-1))

class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model with a Language Modeling head ("Improving Language Understanding by Generative Pre-Training").

    There are two main implementation differences between BERT and the OpenAI GPT:
        - the use of an LM loss in OpenAI GPT which means the Transformer is trained to predict the NEXT token for each input token
            vs. predict the SAME token for BERT (i.e. you need to shift your labels to the right)
        - the use, in OpenAI GPT, of a single embedding matrix to store the word, special ([SEP], [CLS]...) and position embeddings.
    The embeddings are ordered as follow in the word embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1,                  ______________________
         config.vocab_size + config.n_special,
         ...                                                        -> position embeddings
         total_num_embeddings - 1]                                  ______________________

    where total_num_embeddings can be obtained as config.total_num_embeddings and is:
        total_num_embeddings = config.vocab_size + config.n_special + config.n_ctx
    You should use these indices to index the word, special and position embeddings.

    The special embeddings ([SEP], [CLS]...) are not pre-trained and need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length]
            were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, config.vocab_size[
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [config.vocab_size + config.n_special, config.vocab_size + config.n_special + config.n_ctx - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third embedding (the previous two being the word and position embeddings)
            to each token in the sentence.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `lm_labels` is not `None`:
            Outputs the language modeling loss.
        else:
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, total_num_embeddings]
                (or more generally [d_1, ..., d_n, total_num_embeddings] were d_1 ... d_n are the dimension of input_ids)

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTLMHeadModel(config)
    lm_logits = model(input_ids)
    ```
    """
    def __init__(self, config):
        super(OpenAIGPTLMHeadModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = OpenAIGPTLMHead(self.transformer.embed.weight, config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input and output embeddings with new embedding matrice "
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.embed.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None):
        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits

class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model with a Language Modeling and a Multiple Choice heads ("Improving Language Understanding by Generative Pre-Training").

    There are two main implementation differences between BERT and the OpenAI GPT:
        - the use of an LM loss in OpenAI GPT which means the Transformer is trained to predict the NEXT token for each input token
            vs. predict the SAME token for BERT (i.e. you need to shift your labels to the right)
        - the use, in OpenAI GPT, of a single embedding matrix to store the word, special ([SEP], [CLS]...) and position embeddings.
    The embeddings are ordered as follow in the word embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1,                  ______________________
         config.vocab_size + config.n_special,
         ...                                                        -> position embeddings
         total_num_embeddings - 1]                                  ______________________

    where total_num_embeddings can be obtained as config.total_num_embeddings and is:
        total_num_embeddings = config.vocab_size + config.n_special + config.n_ctx
    You should use these indices to index the word, special and position embeddings.

    The special embeddings ([SEP], [CLS]...) are not pre-trained and need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word BPE token indices selected in the range [0, config.vocab_size[
        `multiple_choice_token_mask`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with a value of 1 were the last hidden state is (usually the [CLS] token) and 0 otherwise.
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [config.vocab_size + config.n_special,
            config.vocab_size + config.n_special + config.n_ctx - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third embedding (the previous two being the word and position embeddings)
            to each token in the sentence.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with indices selected in [-1, 0, ..., total_num_embeddings]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., total_num_embeddings]
        `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `lm_labels` and `multiple_choice_labels` are not `None`:
            Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
        else: a tuple with
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, total_num_embeddings]
            `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    multiple_choice_token_mask = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTLMHeadModel(config)
    lm_logits, multiple_choice_logits = model(input_ids, multiple_choice_token_mask)
    ```
    """
    def __init__(self, config):
        super(OpenAIGPTDoubleHeadsModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = OpenAIGPTLMHead(self.transformer.embed.weight, config)
        self.multiple_choice_head = OpenAIGPTMultipleChoiceHead(config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input and output embeddings with new embedding matrice "
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.embed.weight)

    def forward(self, input_ids, multiple_choice_token_mask, position_ids=None, token_type_ids=None,
                lm_labels=None, multiple_choice_labels=None):
        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        lm_logits = self.lm_head(hidden_states)
        multiple_choice_logits = self.multiple_choice_head(hidden_states, multiple_choice_token_mask)
        losses = []
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)))
        if multiple_choice_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(multiple_choice_logits, multiple_choice_labels.view(-1)))
        if losses:
            return losses
        return lm_logits, multiple_choice_logits
