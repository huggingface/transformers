# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
"""PyTorch Pegasus model, ported from https://github.com/google-research/pegasus"""


from .configuration_pegasus import PegasusConfig
from .file_utils import add_start_docstrings
from .modeling_bart import BART_START_DOCSTRING, BartForConditionalGeneration


@add_start_docstrings("The Pegasus Model for summarization ", BART_START_DOCSTRING)
class PegasusForConditionalGeneration(BartForConditionalGeneration):
    config_class = PegasusConfig
    authorized_missing_keys = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"model.encoder.embed_positions",
        "model.decoder.embed_positions",
    ]
    r"""
    Pytorch version of google's pegasus model for summarization.
    Model API is identical to BartForConditionalGeneration.
    Available models are listed at `Model List <https://huggingface.co/models?search=pegasus>`__

    Examples::

        >>> from transformers import PegasusTokenizer, PegasusForConditionalGeneration
        >>> from typing import List
        >>> PGE_ARTICLE = "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
        >>> mname = "google/pegasus-xsum"

        >>> model = PegasusForConditionalGeneration.from_pretrained(mname)
        >>> tok = PegasusTokenizer.from_pretrained(mname)
        >>> batch = tok.prepare_seq2seq_batch(src_texts=[PGE_ARTICLE])  # don't need tgt_text for inference
        >>> gen = model.generate(**batch)  # for forward pass: model(**batch)
        >>> summary: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
        >>> assert summary == "California's largest electricity provider has turned off power to tens of thousands of customers."

    """
    # All the code is in src/transformers/modeling_bart.py
