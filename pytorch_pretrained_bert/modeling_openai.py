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
                 intermediate_size=3072,
                 afn="gelu",
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 type_vocab_size=2,
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
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            afn: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `OpenAIGPTModel`.
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
            self.intermediate_size = intermediate_size
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.type_vocab_size = type_vocab_size
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

class OpenAIGPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(OpenAIGPTPreTrainedModel, self).__init__()
        if not isinstance(config, OpenAIGPTConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `OpenAIGPTConfig`. "
                "To create a model from a Google pretrained model use "
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

    def post_loading(self):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
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
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
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
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        model.post_loading()
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model

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
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
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
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class OpenAIGPTLMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, cfg):
        super(OpenAIGPTLMHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights # Tied weights

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class OpenAIGPTClfHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, clf_token, cfg):
        super(OpenAIGPTClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout2d(cfg.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.linear = nn.Linear(cfg.n_embd, 1)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        # Classification logits
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = clf_h.view(-1, x.size(1), self.n_embd, 1)
        # This double transposition is there to replicate the behavior
        # of the noise_shape argument in the tensorflow
        # implementation.  For more details, see
        # https://github.com/huggingface/pytorch-openai-transformer-lm/issues/11
        clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)
        clf_h = clf_h.contiguous().view(-1, self.n_embd)
        clf_logits = self.linear(clf_h)

        return clf_logits.view(-1, x.size(1))


class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    """ OpenAI GPT model """

    def __init__(self, cfg):
        super(OpenAIGPTModel, self).__init__(cfg)
        total_embeddings_size = cfg.vocab_size + cfg.n_special + cfg.n_ctx
        self.embed = nn.Embedding(total_embeddings_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(cfg.n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])

        self.apply(self.init_weights)
        # nn.init.normal_(self.embed.weight, std=0.02)

    def set_num_special_tokens(self, num_special_tokens):
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

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.embed(x)
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        for block in self.h:
            h = block(h)
        return h


class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    """ OpenAI GPT model with language model and classification heads """
    def __init__(self, cfg, clf_token='[CLS]'):
        super(OpenAIGPTDoubleHeadsModel, self).__init__(cfg)
        self.transformer = OpenAIGPTModel(cfg)
        self.lm_head = OpenAIGPTLMHead(self.transformer.embed.weight, cfg)
        self.clf_head = OpenAIGPTClfHead(clf_token, cfg)
        self.apply(self.init_weights)

    def post_loading(self):
        " Set the number of special tokens to 1 (for the [CLS] token) "
        self.set_num_special_tokens(1)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input and output embeddings with new embedding matrice "
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.embed.weight)

    def forward(self, x, lm_labels=None, clf_labels=None):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        clf_logits = self.clf_head(h, x)
        losses = []
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(lm_logits, lm_labels))
        if clf_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(clf_logits, clf_labels))
        if losses:
            return losses
        return lm_logits, clf_logits
