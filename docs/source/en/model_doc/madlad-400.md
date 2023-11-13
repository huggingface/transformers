<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MADLAD-400

## Overview

MADLAD-400 models were released in the paper [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](MADLAD-400: A Multilingual And Document-Level Large Audited Dataset) and converted to transformers' format by [Juarez Bochi](https://github.com/jbochi). - These are machine translation models based on the T5 architecture.

One can directly use MADLAD-400 weights without finetuning the model:

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("jbochi/madlad400-3b-mt")
>>> tokenizer = AutoTokenizer.from_pretrained("jbochi/madlad400-3b-mt")

>>> inputs = tokenizer("<2pt> I love pizza!", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Eu amo pizza!']
```

Google has released the following variants:

- [jbochi/madlad400-3b-mt](https://huggingface.co/jbochi/madlad400-3b-mt)

- [jbochi/madlad400-7b-mt](https://huggingface.co/jbochi/madlad400-7b-mt)

- [jbochi/madlad400-7b-mt-bt](https://huggingface.co/jbochi/madlad400-7b-mt-bt)

- [jbochi/madlad400-10b-mt](https://huggingface.co/jbochi/madlad400-10b-mt)

The original checkpoints can be found [here](https://github.com/google-research/google-research/tree/master/madlad_400).

<Tip>

Refer to [T5's documentation page](t5) for all API reference, code examples and notebooks. For more details regarding training and evaluation of the MADLAD-400, refer to the model card.

</Tip>