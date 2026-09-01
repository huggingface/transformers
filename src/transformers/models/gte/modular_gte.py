# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss

from ...modeling_outputs import MaskedLMOutput
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, no_inherit_decorator
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertForSequenceClassification,
    BertForTokenClassification,
    BertPreTrainedModel,
)
from ..gemma.modeling_gemma import GemmaMLP
from ..jina_embeddings_v3.modeling_jina_embeddings_v3 import (
    JinaEmbeddingsV3Attention,
    JinaEmbeddingsV3Embeddings,
    JinaEmbeddingsV3ForMaskedLM,
    JinaEmbeddingsV3Layer,
    JinaEmbeddingsV3LMHead,
    JinaEmbeddingsV3Model,
    JinaEmbeddingsV3Pooler,
)
from ..llama.modeling_llama import LlamaRotaryEmbedding


@auto_docstring(checkpoint="Alibaba-NLP/gte-multilingual-base")
@strict
class GteConfig(BertConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import GteConfig, GteModel

    >>> # Initializing a GTE Alibaba-NLP/gte-multilingual-base style configuration
    >>> configuration = GteConfig()

    >>> # Initializing a model (with random weights) from the Alibaba-NLP/gte-multilingual-base style configuration
    >>> model = GteModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gte"

    vocab_size: int = 250048
    attention_probs_dropout_prob: float | int = 0.0
    type_vocab_size: int = 1
    pad_token_id: int | None = 1
    max_position_embeddings: int = 8192
    rope_parameters: RopeParameters | dict | None = None
    is_decoder = AttributeError()
    add_cross_attention = AttributeError()
    use_cache = AttributeError()

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}
        rope_theta = kwargs.pop("rope_theta", self.default_theta)

        # Static NTK scaling in Alibaba-NLP/gte-multilingual-base is exactly a linear scaling of `base * factor`.
        if rope_scaling is not None and rope_scaling["type"] == "ntk":
            head_dim = self.hidden_size // self.num_attention_heads
            factor = rope_scaling["factor"]
            self.rope_parameters.setdefault("rope_type", "linear")
            self.rope_parameters.setdefault("factor", factor ** (2 / head_dim))
            rope_theta = rope_theta * factor

        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.standardize_rope_params()
        return kwargs


class GteEmbeddings(JinaEmbeddingsV3Embeddings):
    def __init__(self, config: GteConfig):
        super().__init__(config)
        # GTE is rope-only: absolute position ids are never read, and token types always default to zeros.
        del self.position_ids
        del self.token_type_ids
        self.token_type_embeddings = (
            # CODEPATH: Only gte-base-en-v1.5 sets type_vocab_size=0, the others ship token type embeddings.
            nn.Embedding(config.type_vocab_size, config.hidden_size) if config.type_vocab_size else None
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        # Unlike JinaEmbeddingsV3Embeddings, the token type embedding is skipped when type_vocab_size is 0.
        embeddings = inputs_embeds
        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)

        input_shape = embeddings.shape[:-1]
        device = embeddings.device

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class GteRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class GteAttention(JinaEmbeddingsV3Attention):
    pass


@no_inherit_decorator
class GteMLP(GemmaMLP):
    def __init__(self, config: GteConfig):
        super().__init__(config)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unlike GemmaMLP, `down_proj` carries a bias and the gated activation is dropped out.
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        down_proj = self.down_proj(hidden_states)
        return down_proj


class GteLayer(JinaEmbeddingsV3Layer):
    pass


class GtePooler(JinaEmbeddingsV3Pooler):
    pass


class GtePreTrainedModel(BertPreTrainedModel):
    config_class = GteConfig
    base_model_prefix = "gte"

    # Are kept as non-persistent buffers to avoid being saved in the state dict
    # and causing mismatch when loading from a checkpoint that doesn't have them
    _keys_to_ignore_on_load_unexpected = ["inv_freq", "original_inv_freq"]
    _can_record_outputs = {
        "hidden_states": GteLayer,
        "attentions": GteAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        # None of the inherited buffer initialisations apply, GTE keeps no such buffers.
        PreTrainedModel._init_weights(self, module)


@auto_docstring
class GteModel(JinaEmbeddingsV3Model):
    def __init__(self, config: GteConfig, add_pooling_layer=False):
        r"""
        add_pooling_layer (`bool`, *optional*, defaults to `False`):
            Whether to add a pooling layer.
        """
        super().__init__(config, add_pooling_layer=add_pooling_layer)


class GteLMHead(JinaEmbeddingsV3LMHead):
    def __init__(self, config: GteConfig):
        super().__init__(config)
        # GTE folds the output bias into `decoder`, it has no separate bias parameter.
        del self.bias


@auto_docstring
class GteForMaskedLM(JinaEmbeddingsV3ForMaskedLM):
    _tied_weights_keys = {"lm_head.decoder.weight": "gte.embeddings.word_embeddings.weight"}

    def __init__(self, config: GteConfig):
        PreTrainedModel.__init__(self, config)

        self.gte = GteModel(config, add_pooling_layer=False)
        self.lm_head = GteLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        # Unlike JinaEmbeddingsV3ForMaskedLM, the backbone is bound as `self.gte`, which modular cannot rename.
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.gte(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GteForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: GteConfig):
        super().__init__(config)
        self.gte = GteModel(config, add_pooling_layer=True)


class GteForTokenClassification(BertForTokenClassification):
    pass


__all__ = [
    "GteConfig",
    "GtePreTrainedModel",
    "GteModel",
    "GteForMaskedLM",
    "GteForSequenceClassification",
    "GteForTokenClassification",
]
