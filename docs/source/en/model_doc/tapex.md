<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TAPEX

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The TAPEX model was proposed in [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/abs/2107.07653) by Qian Liu,
Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou. TAPEX pre-trains a BART model to solve synthetic SQL queries, after
which it can be fine-tuned to answer natural language questions related to tabular data, as well as performing table fact checking. 

TAPEX has been fine-tuned on several datasets: 
- [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) (Sequential Question Answering by Microsoft)
- [WTQ](https://github.com/ppasupat/WikiTableQuestions) (Wiki Table Questions by Stanford University)
- [WikiSQL](https://github.com/salesforce/WikiSQL) (by Salesforce)
- [TabFact](https://tabfact.github.io/) (by USCB NLP Lab).

The abstract from the paper is the following:

*Recent progress in language model pre-training has achieved a great success via leveraging large-scale unstructured textual data. However, it is
still a challenge to apply pre-training on structured tabular data due to the absence of large-scale high-quality tabular data. In this paper, we
propose TAPEX to show that table pre-training can be achieved by learning a neural SQL executor over a synthetic corpus, which is obtained by automatically
synthesizing executable SQL queries and their execution outputs. TAPEX addresses the data scarcity challenge via guiding the language model to mimic a SQL
executor on the diverse, large-scale and high-quality synthetic corpus. We evaluate TAPEX on four benchmark datasets. Experimental results demonstrate that
TAPEX outperforms previous table pre-training approaches by a large margin and achieves new state-of-the-art results on all of them. This includes improvements
on the weakly-supervised WikiSQL denotation accuracy to 89.5% (+2.3%), the WikiTableQuestions denotation accuracy to 57.5% (+4.8%), the SQA denotation accuracy
to 74.5% (+3.5%), and the TabFact accuracy to 84.2% (+3.2%). To our knowledge, this is the first work to exploit table pre-training via synthetic executable programs
and to achieve new state-of-the-art results on various downstream tasks.*

## Usage tips

- TAPEX is a generative (seq2seq) model. One can directly plug in the weights of TAPEX into a BART model. 
- TAPEX has checkpoints on the hub that are either pre-trained only, or fine-tuned on WTQ, SQA, WikiSQL and TabFact.
- Sentences + tables are presented to the model as `sentence + " " + linearized table`. The linearized table has the following format: 
  `col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...`.
- TAPEX has its own tokenizer, that allows to prepare all data for the model easily. One can pass Pandas DataFrames and strings to the tokenizer,
  and it will automatically create the `input_ids` and `attention_mask` (as shown in the usage examples below). 

### Usage: inference

Below, we illustrate how to use TAPEX for table question answering. As one can see, one can directly plug in the weights of TAPEX into a BART model.
We use the [Auto API](auto), which will automatically instantiate the appropriate tokenizer ([`TapexTokenizer`]) and model ([`BartForConditionalGeneration`]) for us,
based on the configuration file of the checkpoint on the hub.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")

>>> # prepare table + question
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> question = "how many movies does Leonardo Di Caprio have?"

>>> encoding = tokenizer(table, question, return_tensors="pt")

>>> # let the model generate an answer autoregressively
>>> outputs = model.generate(**encoding)

>>> # decode back to text
>>> predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
>>> print(predicted_answer)
53
```

Note that [`TapexTokenizer`] also supports batched inference. Hence, one can provide a batch of different tables/questions, or a batch of a single table
and multiple questions, or a batch of a single query and multiple tables. Let's illustrate this:

```python
>>> # prepare table + question
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> questions = [
...     "how many movies does Leonardo Di Caprio have?",
...     "which actor has 69 movies?",
...     "what's the first name of the actor who has 87 movies?",
... ]
>>> encoding = tokenizer(table, questions, padding=True, return_tensors="pt")

>>> # let the model generate an answer autoregressively
>>> outputs = model.generate(**encoding)

>>> # decode back to text
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
[' 53', ' george clooney', ' brad pitt']
```

In case one wants to do table verification (i.e. the task of determining whether a given sentence is supported or refuted by the contents
of a table), one can instantiate a [`BartForSequenceClassification`] model. TAPEX has checkpoints on the hub fine-tuned on TabFact, an important
benchmark for table fact checking (it achieves 84% accuracy). The code example below again leverages the [Auto API](auto).

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
>>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact")

>>> # prepare table + sentence
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> sentence = "George Clooney has 30 movies"

>>> encoding = tokenizer(table, sentence, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**encoding)

>>> # print prediction
>>> predicted_class_idx = outputs.logits[0].argmax(dim=0).item()
>>> print(model.config.id2label[predicted_class_idx])
Refused
```

<Tip> 

TAPEX architecture is the same as BART, except for tokenization. Refer to [BART documentation](bart) for information on 
configuration classes and their parameters. TAPEX-specific tokenizer is documented below.  

</Tip>

## TapexTokenizer

[[autodoc]] TapexTokenizer
    - __call__
    - save_vocabulary