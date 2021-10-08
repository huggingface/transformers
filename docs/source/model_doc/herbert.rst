.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

HerBERT
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HerBERT model was proposed in `KLEJ: Comprehensive Benchmark for Polish Language Understanding
<https://www.aclweb.org/anthology/2020.acl-main.111.pdf>`__ by Piotr Rybak, Robert Mroczkowski, Janusz Tracz, and
Ireneusz Gawlik. It is a BERT-based Language Model trained on Polish Corpora using only MLM objective with dynamic
masking of whole words.

The abstract from the paper is the following:

*In recent years, a series of Transformer-based models unlocked major improvements in general natural language
understanding (NLU) tasks. Such a fast pace of research would not be possible without general NLU benchmarks, which
allow for a fair comparison of the proposed methods. However, such benchmarks are available only for a handful of
languages. To alleviate this issue, we introduce a comprehensive multi-task benchmark for the Polish language
understanding, accompanied by an online leaderboard. It consists of a diverse set of tasks, adopted from existing
datasets for named entity recognition, question-answering, textual entailment, and others. We also introduce a new
sentiment analysis task for the e-commerce domain, named Allegro Reviews (AR). To ensure a common evaluation scheme and
promote models that generalize to different NLU tasks, the benchmark includes datasets from varying domains and
applications. Additionally, we release HerBERT, a Transformer-based model trained specifically for the Polish language,
which has the best average performance and obtains the best results for three out of nine tasks. Finally, we provide an
extensive evaluation, including several standard baselines and recently proposed, multilingual Transformer-based
models.*

Examples of use:

.. code-block::

    >>> from transformers import HerbertTokenizer, RobertaModel

    >>> tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
    >>> model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

    >>> encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors='pt')
    >>> outputs = model(encoded_input)

    >>> # HerBERT can also be loaded using AutoTokenizer and AutoModel:
    >>> import torch
    >>> from transformers import AutoModel, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
    >>> model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")


This model was contributed by `rmroczkowski <https://huggingface.co/rmroczkowski>`__. The original code can be found
`here <https://github.com/allegro/HerBERT>`__.


HerbertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.HerbertTokenizer
    :members: 

HerbertTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.HerbertTokenizerFast
    :members: 
