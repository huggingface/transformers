# coding=utf-8
# Copyright 2020, The HuggingFace Inc. team.
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
"""PyTorch BORT model. """

from ...file_utils import add_start_docstrings
from ...utils import logging
from ..bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)
from .configuration_bort import BortConfig


logger = logging.get_logger(__name__)

_TOKENIZER_FOR_DOC = "RobertaTokenizer"

BORT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all BORT models at https://huggingface.co/models?filter=bort
]

BORT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BortConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


@add_start_docstrings(
    "The bare BORT Model transformer outputting raw hidden-states without any specific head on top.",
    BORT_START_DOCSTRING,
)
class BortModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::
        >>> from transformers import BortModel, BortTokenizer

        >>> model = BortModel.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer.encode_plus("The eastern endpoint of the canal is the Hubertusbrunnen.", return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> hidden_states = outputs.last_hidden_state
    """

    model_type = "bort"
    config_class = BortConfig


@add_start_docstrings(
    """BORT Model with a `language modeling` head on top. """,
    BORT_START_DOCSTRING,
)
class BortForMaskedLM(BertForMaskedLM):
    """
    This class overrides :class:`~transformers.BertForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::
        >>> from transformers import BortForMaskedLM, BortTokenizer
        >>> import torch

        >>> model = BortForMaskedLM.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer("The Nymphenburg Palace is <mask>.", return_tensors="pt")
        >>> labels = tokenizer("The Nymphenburg Palace is beautiful.", return_tensors="pt")["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    model_type = "bort"
    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BORT_START_DOCSTRING,
)
class BortForSequenceClassification(BertForSequenceClassification):
    """
    This class overrides :class:`~transformers.BertForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import BortForSequenceClassification, BortTokenizer
        >>> import torch

        >>> model = BortForSequenceClassification.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer("The Nymphenburg Palace is beautiful.", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    model_type = "bort"
    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BORT_START_DOCSTRING,
)
class BortForMultipleChoice(BertForMultipleChoice):
    """
    This class overrides :class:`~transformers.BertForMultipleChoice`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::
        >>> from transformers import BortForMultipleChoice, BortTokenizer
        >>> import torch

        >>> model = BortForMultipleChoice.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> prompt = "The Nymphenburg Palace is a:"
        >>> choice0 = "Baroque palace."
        >>> choice1 = "Gothic palace."  # no!
        >>> labels = torch.tensor(0).unsqueeze(0)  # it is a beautiful Baroque palace

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
        >>> outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    model_type = "bort"
    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BORT_START_DOCSTRING,
)
class BortForTokenClassification(BertForTokenClassification):
    """
    This class overrides :class:`~transformers.BertForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import BortForTokenClassification, BortTokenizer
        >>> import torch

        >>> model = BortForTokenClassification.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer("The Nymphenburg Palace in Munich.", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    model_type = "bort"
    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`
    """,
    BORT_START_DOCSTRING,
)
class BortForQuestionAnswering(BertForQuestionAnswering):
    """
    This class overrides :class:`~transformers.BertForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import BortForQuestionAnswering, BortTokenizer
        >>> import torch

        >>> model = BortForQuestionAnswering.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> question, text = "Where is Nymphenburg Palace?", "Nymphenburg Palace is located in Munich"
        >>> inputs = tokenizer(question, text, return_tensors='pt')
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([6])

        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
    """

    model_type = "bort"
    config_class = BortConfig
