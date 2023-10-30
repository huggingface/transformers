<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PhoBERT

## Overview

The PhoBERT model was proposed in [PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92.pdf) by Dat Quoc Nguyen, Anh Tuan Nguyen.

The abstract from the paper is the following:

*We present PhoBERT with two versions, PhoBERT-base and PhoBERT-large, the first public large-scale monolingual
language models pre-trained for Vietnamese. Experimental results show that PhoBERT consistently outperforms the recent
best pre-trained multilingual model XLM-R (Conneau et al., 2020) and improves the state-of-the-art in multiple
Vietnamese-specific NLP tasks including Part-of-speech tagging, Dependency parsing, Named-entity recognition and
Natural language inference.*

Example of use:

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> phobert = AutoModel.from_pretrained("vinai/phobert-base")
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

>>> # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
>>> line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = phobert(input_ids)  # Models outputs are now tuples

>>> # With TensorFlow 2.0+:
>>> # from transformers import TFAutoModel
>>> # phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
```

This model was contributed by [dqnguyen](https://huggingface.co/dqnguyen). The original code can be found [here](https://github.com/VinAIResearch/PhoBERT).

## PhobertTokenizer

[[autodoc]] PhobertTokenizer
