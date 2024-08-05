<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# StableLM

## Overview

`StableLM 3B 4E1T` was proposed in [`StableLM 3B 4E1T`: Technical Report](https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo) by Stability AI and is the first model in a series of multi-epoch pre-trained language models.

### Model Details

`StableLM 3B 4E1T` is a decoder-only base language model pre-trained on 1 trillion tokens of diverse English and code datasets for four epochs.
The model architecture is transformer-based with partial Rotary Position Embeddings, SwiGLU activation, LayerNorm, etc.

We also provide `StableLM Zephyr 3B`, an instruction fine-tuned version of the model that can be used for chat-based applications.

### Usage Tips

- The architecture is similar to LLaMA but with RoPE applied to 25% of head embedding dimensions, LayerNorm instead of RMSNorm, and optional QKV bias terms.
- `StableLM 3B 4E1T`-based models uses the same tokenizer as [`GPTNeoXTokenizerFast`].

`StableLM 3B 4E1T` and `StableLM Zephyr 3B` can be found on the [Huggingface Hub](https://huggingface.co/stabilityai)

The following code snippet demonstrates how to use `StableLM 3B 4E1T` for inference:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> device = "cuda" # the device to load the model onto

>>> set_seed(0)

>>> tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
>>> model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t")
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> model_inputs = tokenizer("The weather is always wonderful in", return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(**model_inputs, max_length=32, do_sample=True)
>>> responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
>>> responses
['The weather is always wonderful in Costa Rica, which makes it a prime destination for retirees. That’s where the Pensionado program comes in, offering']
```

## Combining StableLM and Flash Attention 2

First, make sure to install the latest version of Flash Attention v2.

```bash
pip install -U flash-attn --no-build-isolation
```

Also make sure that your hardware is compatible with Flash-Attention 2. Read more about it in the official documentation of the [`flash-attn`](https://github.com/Dao-AILab/flash-attention) repository. Note: you must load your model in half-precision (e.g. `torch.bfloat16`).

Now, to run the model with Flash Attention 2, refer to the snippet below:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> device = "cuda" # the device to load the model onto

>>> set_seed(0)

>>> tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
>>> model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")  # doctest: +SKIP
>>> model.to(device)  # doctest: +SKIP

>>> model_inputs = tokenizer("The weather is always wonderful in", return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(**model_inputs, max_length=32, do_sample=True)  # doctest: +SKIP
>>> responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  # doctest: +SKIP
>>> responses  # doctest: +SKIP
['The weather is always wonderful in Costa Rica, which makes it a prime destination for retirees. That’s where the Pensionado program comes in, offering']
```


## StableLmConfig

[[autodoc]] StableLmConfig

## StableLmModel

[[autodoc]] StableLmModel
    - forward

## StableLmForCausalLM

[[autodoc]] StableLmForCausalLM
    - forward

## StableLmForSequenceClassification

[[autodoc]] StableLmForSequenceClassification
    - forward

## StableLmForTokenClassification

[[autodoc]] StableLmForTokenClassification
    - forward
