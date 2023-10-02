<!--Copyright 2023 Mistral AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Mistral

## Overview

Mistral-7B-v0.1 is Mistral AI’s first Large Language Model (LLM). 

## Model Details

Mistral-7B-v0.1 is a decoder-based LM with the following architectural choices:
* Sliding Window Attention - Trained with 8k context length and fixed cache size, with a theoretical attention span of 128K tokens
* GQA (Grouped Query Attention) - allowing faster inference and lower cache size.
* Byte-fallback BPE tokenizer - ensures that characters are never mapped to out of vocabulary tokens.

We also provide an instruction fine-tuned model: `Mistral-7B-Instruct-v0.1` which can be used for chat-based inference.

For more details please read our [release blog post](https://mistral.ai/news/announcing-mistral-7b-v0.1/)

## License

Both `Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.1` are released under the Apache 2.0 license.

## Usage

`Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.1` can be found on the [Huggingface Hub](https://huggingface.co/mistralai)

These ready-to-use checkpoints can be downloaded and used via the HuggingFace Hub:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected outupt"
```

Raw weights for `Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.1` can be downloaded from:

| Model Name                 | Checkpoint                                                                              |
|----------------------------|-----------------------------------------------------------------------------------------|
| `Mistral-7B-v0.1`          | [Raw Checkpoint](https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar)          |
| `Mistral-7B-Instruct-v0.1` | [Raw Checkpoint](https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-instruct-v0.1.tar) |


To use these raw checkpoints with HuggingFace you can use the `convert_mistral_weights_to_hf.py` script to convert them to the HuggingFace format:

```bash
python src/transformers/models/mistral/convert_mistral_weights_to_hf.py \
    --input_dir /path/to/downloaded/mistral/weights --model_size 7B --output_dir /output/path
```

You can then load the converted model from the `output/path`:

```python
from transformers import MistralForCausalLM, LlamaTokenzier

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = MistralForCausalLM.from_pretrained("/output/path")
```

## The Mistral Team

Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.

## MistralConfig

[[autodoc]] MistralConfig

## MistralModel

[[autodoc]] MistralModel
    - forward

## MistralForCausalLM

[[autodoc]] MistralForCausalLM
    - forward

## MistralForSequenceClassification

[[autodoc]] MistralForSequenceClassification
    - forward