"""
TODO: add dropout where appropriate and a training indicator
"""

import os
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import torch

from typing import Callable

import haiku as hk
from haiku.initializers import Constant
from jax.random import PRNGKey
from jax.tree_util import tree_unflatten
from collections import defaultdict
from haiku._src.data_structures import frozendict

from transformers import BertModel as PTBertModel, BertTokenizerFast, BertConfig
from transformers.modeling_jax_utils import JaxPreTrainedModel


# Models are loaded from Pytorch checkpoints
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}



def gelu(x):
    """
    Less efficient but more precise than 
    the approximation used by jax.nn.gelu
    """
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


ACT2FN = {
    "gelu": gelu,
    "relu": jax.nn.relu,
    "swish": jax.nn.swish,
    "gelu_new": jax.nn.gelu,
}


class PretrainedModule(hk.Module):
    """
    Lightweight wrapper around hk.Module to expose 
    pre-trained parameters when provided
    """

    def __init__(self, name=None, state=None, **settings):
        super().__init__(name=name)
        if state is None:
            self.state = {}
        else:
            self.state = state
        for setting_key, setting_value in settings.items():
            setattr(self, setting_key, setting_value)
    
    def pretrained(self, key):
        try:
            return Constant(self.state[key])
        except KeyError:
            print(f"Failed to find {key} in `{self.__class__.__name__}({self.name})` state: {self.state.keys()}")
            return None


class BertLayerNorm(PretrainedModule):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    """

    def __init__(
        self,
        name="LayerNorm",
        state=None,
        epsilon=1e-6,
        bias=True,
        scale=True,
        bias_init=hk.initializers.Constant(0.0),
        scale_init=hk.initializers.Constant(1.0),
    ):
        super().__init__(
            name=name, state=state, epsilon=epsilon, bias=bias, scale=scale, bias_init=bias_init, scale_init=scale_init
        )

    def __call__(self, x):
        """Applies layer normalization on the input.
        It normalizes the activations of the layer for each given example in a
        batch independently, rather than across a batch like Batch Normalization.
        i.e. applies a transformation that maintains the mean activation within
        each example close to 0 and the activation standard deviation close to 1.
        Args:
          x: the inputs
          epsilon: A small float added to variance to avoid dividing by zero.
          dtype: the dtype of the computation (default: float32).
          bias:  If True, bias (beta) is added.
          scale: If True, multiply by scale (gamma). When the next layer is linear
            (also e.g. nn.relu), this can be disabled since the scaling will be done
            by the next layer.
          bias_init: Initializer for bias, by default, zero.
          scale_init: Initializer for scale, by default, one.
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        return hk.LayerNorm(
            name="LayerNorm",
            axis=-1,
            eps=self.epsilon,
            create_scale=self.scale,
            create_offset=self.bias,
            scale_init=self.pretrained("gamma") or self.scale_init,
            offset_init=self.pretrained("beta") or self.bias_init,
        )(x)


class BertEmbedding(PretrainedModule):
    """
    Handles word_embedding / position embedding and token type embedding
    """

    def __call__(self, x):
        flat_token_ids = jnp.reshape(x, [1, x.shape[0] * x.shape[1]])
        flat_token_embeddings = hk.Embed(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            # TODO: get actual weight stddev value
            w_init=self.pretrained("weight") or hk.initializers.RandomNormal(0.1),
        )(flat_token_ids)
        token_embeddings = jnp.reshape(flat_token_embeddings, [x.shape[0], x.shape[1], flat_token_embeddings.shape[-1]])
        return token_embeddings


class BertEmbeddings(PretrainedModule):
    def __call__(self, input_ids, token_type_ids):
        # Embed
        w_emb = BertEmbedding(
            name="word_embeddings",
            state=self.state["word_embeddings"],
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )(jnp.atleast_2d(input_ids.astype("i4")))
        p_emb = BertEmbedding(
            name="position_embeddings",
            state=self.state["position_embeddings"],
            vocab_size=self.max_length,
            hidden_size=self.hidden_size,
        )(jnp.atleast_2d(np.arange(input_ids.shape[-1])))
        t_emb = BertEmbedding(
            name="token_type_embeddings",
            state=self.state["token_type_embeddings"],
            vocab_size=self.type_vocab_size,
            hidden_size=self.hidden_size,
        )(jnp.atleast_2d(token_type_ids.astype("i4")))

        # Sum all embeddings
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb

        # Layer Norm
        norm = BertLayerNorm(name="LayerNorm", state=self.state["LayerNorm"])(summed_emb)
        return norm


class Linear(PretrainedModule):
    def __call__(self, x):
        return hk.Linear(
            name=self.name,
            output_size=self.output_size,
            w_init=self.pretrained("weight") or hk.initializers.RandomNormal(0.1),
            b_init=self.pretrained("bias") or hk.initializers.Constant(0.0),
        )(x)


class SelfAttention(PretrainedModule):
    def _split_into_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.num_heads, self.head_size])

    def _join_heads(self, x):
        return jnp.reshape(x, [x.shape[0], x.shape[1], self.num_heads * self.head_size])

    def __call__(self, x, mask):
        # Project to queries, keys, and values
        # Shapes are all [batch, sequence_length, hidden_size]
        hidden_size = self.num_heads * self.head_size

        queries = Linear(name="query", state=self.state["query"], output_size=hidden_size)(x)
        keys = Linear(name="key", state=self.state["key"], output_size=hidden_size)(x)
        values = Linear(name="value", state=self.state["value"], output_size=hidden_size)(x)

        # Reshape our hidden state to group into heads
        # New shapes are:
        # [batch, sequence_length, n_heads, size_per_head]
        queries = self._split_into_heads(queries)
        keys = self._split_into_heads(keys)
        values = self._split_into_heads(values)

        # Compute per head attention weights
        # b: batch
        # s: source sequence
        # t: target sequence
        # n: number of heads
        # h: per-head hidden state
        attention_logits = jnp.einsum("bsnh,btnh->bnst", queries, keys)
        attention_logits /= np.sqrt(queries.shape[-1])

        # Add logits of mask tokens with a large negative number
        # to prevent attending to those terms.
        attention_logits += jnp.reshape((1 - mask) * -(2 ** 32), [mask.shape[0], 1, 1, mask.shape[1]])
        
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        per_head_attention_output = jnp.einsum("btnh,bnst->bsnh", values, attention_weights)
        attention_output = self._join_heads(per_head_attention_output)

        return attention_output


class BertAttention(PretrainedModule):
    def __call__(self, hidden_state, attention_mask):
        attention_output = SelfAttention(
            name="self", state=self.state["self"], num_heads=self.num_heads, head_size=self.head_size,
        )(hidden_state, attention_mask)
        output = BertOutput(name='output', state=self.state['output'], output_size=self.num_heads * self.head_size)(
            hidden_state=attention_output, residual=hidden_state
        )
        return output


class BertIntermediate(PretrainedModule):
    def __call__(self, hidden_state):
        h = Linear(name="dense", state=self.state["dense"], output_size=self.intermediate_size)(hidden_state)
        return ACT2FN[self.activation_fn](h)


class BertOutput(PretrainedModule):
    def __call__(self, hidden_state, residual):
        h = Linear(name="dense", state=self.state["dense"], output_size=self.output_size)(hidden_state)
        h = BertLayerNorm(name="LayerNorm", state=self.state["LayerNorm"],)(h + residual)
        return h


class BertLayer(PretrainedModule):
    def __call__(self, hidden_state, attention_mask):
        attention = BertAttention(
            name="attention", state=self.state["attention"], num_heads=self.num_heads, head_size=self.head_size
        )(hidden_state, attention_mask)
        intermediate = BertIntermediate(
            name="intermediate",
            state=self.state["intermediate"],
            intermediate_size=self.intermediate_size,
            activation_fn=self.activation_fn,
        )(attention)
        output = BertOutput(name="output", state=self.state["output"], output_size=self.num_heads * self.head_size)(
            intermediate, attention
        )
        return output


class BertEncoder(PretrainedModule):
    def __call__(self, x, mask):
        assert self.num_layers > 0, "num_layers should be >= 1, got ({})".format(self.num_layers)
        # Forward over all encoders
        for i in range(self.num_layers):
            layer_name = "layer_{}".format(i)
            x = BertLayer(
                name=layer_name,
                state=self.state['layer'][str(i)],
                num_heads=self.num_heads,
                head_size=self.head_size,
                intermediate_size=self.intermediate_size,
                activation_fn=self.activation_fn,
            )(x, mask)
        return x


class BertPooler(PretrainedModule):
    def __call__(self, x):
        first_token = x[:, 0]
        out = Linear(name="dense", state=self.state["dense"], output_size=x.shape[-1])(first_token)
        return jax.lax.tanh(out)


class BertModel(PretrainedModule):
    def __call__(self, input_ids, token_type_ids, attention_mask):
        # Embedding
        embeddings = BertEmbeddings(
            name="embeddings",
            state=self.state["embeddings"],
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
        )(input_ids, token_type_ids)

        # N stacked encoding layers
        encoder = BertEncoder(
            name="encoder",
            state=self.state["encoder"],
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            intermediate_size=self.intermediate_size,
            activation_fn=self.activation_fn,
        )(embeddings, jnp.atleast_2d(attention_mask))

        pooled = BertPooler(name="pooler", state=self.state["pooler"],)(encoder)
        return encoder, pooled


class HaikuBertModel(JaxPreTrainedModel):
    """
    BERT implementation using JAX/Haiku as backend
    """
    MODEL_CLASS = BertModel
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config: BertConfig, state: dict, *model_args, **model_kwargs):
        self.config = config
        self.rng = PRNGKey(0)
        self.pretrained_state = state
        self.params = None
        self.predict_fn = None

    def __call__(self, input_ids, token_type_ids, attention_mask):

        if self.params is None or self.predict_fn is None:
            # Lazily initialize parameters and JIT compile
            def predict(input_ids, token_type_ids, attention_mask):
                model = BertModel(
                    name="bert",
                    conifg=self.config,
                    state=self.pretrained_state,
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.hidden_size,
                    type_vocab_size=self.config.type_vocab_size,
                    max_length=self.config.max_position_embeddings,
                    num_layers=self.config.num_hidden_layers,
                    num_heads=self.config.num_attention_heads,
                    head_size=self.config.hidden_size // self.config.num_attention_heads,
                    intermediate_size=self.config.intermediate_size,
                    activation_fn=self.config.hidden_act,
                )
                return model(
                    jnp.atleast_2d(jnp.asarray(input_ids, dtype="i4")),
                    jnp.atleast_2d(jnp.asarray(token_type_ids, dtype="i4")),
                    jnp.atleast_2d(jnp.asarray(attention_mask, dtype="i4")),
                )

            predict_module = hk.transform(predict, apply_rng=True)
            self.params = predict_module.init(self.rng, input_ids, token_type_ids, attention_mask)
            self.predict_fn = jax.jit(predict_module.apply)
        return self.predict_fn(self.params, self.rng, input_ids, token_type_ids, attention_mask)

    @classmethod
    def _pytorch_to_jax(cls, pt_state_dict, config):
        # Translate from flat dictionary to nested dictionary
        # And translate a few matrices
        state = dict(pt_state_dict)
        nested_dict = lambda: defaultdict(nested_dict)
        jax_state = nested_dict()
        for key, value in state.items():

            value = value.numpy()
            if key.endswith("weight") and not "embeddings" in key:
                value = value.T
            
            keys = key.split('.')[1:]

            current_dict = jax_state
            for k in keys[:-1]:
                current_dict = current_dict[k]
            current_dict[keys[-1]] = value
        
        return jax_state

    @classmethod
    def _jax_to_pytorch(cls, state_dict, config):
        nested_state = dict(state_dict)
        
        replacements = {
            'embed': '',
            'embeddings': 'weight',
            'scale': 'gamma',
            'offset': 'beta',
            'w': 'weight',
            'b': 'bias', 
        }

        def normalize(k):
            """
            Get rid of nested namespaces and convert haiku 
            format w/ slashes to dots
            """
            
            k = k.replace('/', '.').replace('layer_', 'layer.')

            chain = k.split('.')
            new_chain = [chain[0]]
            for item in chain[1:]:
                if item == new_chain[-1]:
                    continue
                else:
                    new_chain.append(item)

            k = ".".join(new_chain)
            for before, after in replacements.items():
                if k.endswith(before):
                    k = k.rpartition(before)[0] + after
            return k

        def flatten(flat_state, nested_state, prefix=""):
            for key, value in list(nested_state.items()):
                normalized_key = normalize(key)
                new_prefix = ".".join([prefix, normalized_key]).strip(".")
                if isinstance(value, (dict, frozendict)):
                    flatten(flat_state, value, new_prefix)
                else:
                    if new_prefix.endswith('weight') and not "embeddings" in new_prefix:
                        value = value.T
                    flat_state[new_prefix] = torch.tensor(value)
            return flat_state

        pytorch_state = flatten({}, nested_state, prefix="")
        return pytorch_state


if __name__ == "__main__":
    import numpy as np
    import tempfile
    
    MODEL_NAME = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    haiku_input = tokenizer.batch_encode_plus(["Thanks for the PR review Morgan"] * 2)
    pt_input = tokenizer.encode_plus("Thanks for the PR review Morgan", return_tensors="pt")

    model = HaikuBertModel.from_pretrained(MODEL_NAME)
    seq_features, pooled_features = model(**haiku_input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        reloaded_model = model.from_pretrained(tmp_dir)
        reloaded_seq_features, reloaded_pooled_features = reloaded_model(**haiku_input)
        assert np.allclose(seq_features, reloaded_seq_features, atol=1e-2)
        assert np.allclose(pooled_features, reloaded_pooled_features, atol=1e-2)

    for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys():
        print(f"Testing {model_name}...")
        try:
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            model_pt = PTBertModel.from_pretrained(model_name)
            model = HaikuBertModel.from_pretrained(model_name)
            
            # Inputs
            haiku_input = tokenizer.batch_encode_plus(["Thanks for the PR review Morgan"] * 2)
            pt_input = tokenizer.encode_plus("Thanks for the PR review Morgan", return_tensors="pt")

            # Forward
            model_pt.eval()
            pt_enc = model_pt(pt_input["input_ids"], pt_input["attention_mask"])
            pt_seq_features = pt_enc[0].detach().numpy()
            pt_pooled_features = pt_enc[1].detach().numpy()
            haiku_seq_features, haiku_pooled_features = model(**haiku_input)
            assert np.allclose(haiku_seq_features, pt_seq_features, atol=1e-2)
            assert np.allclose(haiku_pooled_features, pt_pooled_features, atol=1e-2)
        except:
            traceback.print_exc()
            print(f"Failed to load or compare {model_name} JAX version")
