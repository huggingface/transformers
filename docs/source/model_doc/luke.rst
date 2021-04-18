..
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

LUKE
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LUKE model was proposed in `LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention
<https://arxiv.org/abs/2010.01057>`_ by Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda and Yuji Matsumoto.
It is based on RoBERTa and adds entity embeddings as well as an entity-aware self-attention mechanism, which helps
improve performance on several downstream tasks involving reasoning about entities such as named-entity recognition,
extractive and cloze-style question answering, entity linking and relation classification between entities.

The abstract from the paper is the following:

*Entity representations are useful in natural language tasks involving entities. In this paper, we propose new
pretrained contextualized representations of words and entities based on the bidirectional transformer. The proposed
model treats words and entities in a given text as independent tokens, and outputs contextualized representations of
them. Our model is trained using a new pretraining task based on the masked language model of BERT. The task involves
predicting randomly masked words and entities in a large entity-annotated corpus retrieved from Wikipedia. We also
propose an entity-aware self-attention mechanism that is an extension of the self-attention mechanism of the
transformer, and considers the types of tokens (words or entities) when computing attention scores. The proposed model
achieves impressive empirical performance on a wide range of entity-related tasks. In particular, it obtains
state-of-the-art results on five well-known datasets: Open Entity (entity typing), TACRED (relation classification),
CoNLL-2003 (named entity recognition), ReCoRD (cloze-style question answering), and SQuAD 1.1 (extractive question
answering).*

Tips:

- This implementation is the same as :class:`~transformers.RobertaModel` with the addition of entity embeddings as well
  as an entity-aware self-attention mechanism, which improves performance on tasks involving reasoning about entities.
- LUKE adds :obj:`entity_ids`, :obj:`entity_attention_mask`, :obj:`entity_token_type_ids` and
  :obj:`entity_position_ids` as extra input to the model. Input entities can be special entities (e.g., [MASK]) or
  Wikipedia entities (e.g., New York City). You can obtain those using :class:`~transformers.LukeTokenizer`.

Example:

.. code-block::

    >>> from transformers import LukeTokenizer, LukeModel

    >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    >>> model = LukeModel.from_pretrained("studio-ousia/luke-base")

    # Compute the contextualized entity representation corresponding to the entity mention "Beyoncé"
    >>> text = "Beyoncé lives in New York."
    >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
    >>> encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
    >>> outputs = model(**encoding)
    >>> word_last_hidden_state = outputs.last_hidden_state
    >>> entity_last_hidden_state = outputs.entity_last_hidden_state

    # Input Wikipedia entities to obtain enriched contextualized representations.
    >>> text = "Beyoncé lives in New York."
    >>> entities = ["Beyoncé", "New York City"]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "New York"
    >>> entity_spans = [(0, 7), (17, 25)]  # character-based entity spans corresponding to "Beyoncé" and "New York"
    >>> encoding = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
    >>> outputs = model(**encoding)
    >>> word_last_hidden_state = outputs.last_hidden_state
    >>> entity_last_hidden_state = outputs.entity_last_hidden_state

The original code can be found `here <https://github.com/studio-ousia/luke>`__.


LukeConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LukeConfig
    :members:


LukeTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LukeTokenizer
    :members: __call__, save_vocabulary


LukeModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LukeModel
    :members: forward


LukeForEntityClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LukeForEntityClassification
    :members: forward


LukeForEntityPairClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LukeForEntityPairClassification
    :members: forward


LukeForEntitySpanClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LukeForEntitySpanClassification
    :members: forward
