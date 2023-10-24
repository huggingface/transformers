<!-- MIT License

Copyright (c) 2023  NucleusAI and The HuggingFace Inc. team and github/syncdoth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. -->

# NucleusX

## Overview

The NucleusX model proposed by [NucleusAI](https://www.withnucleus.ai/), is
based on "Retentive Network (RetNet)" proposed in [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621).
For detail about the architecture, please read the original paper and this [blog post](https://medium.com/@choisehyun98/the-rise-of-rnn-review-of-retentive-network-a080a9a1ad1d).

## Model Details

The NucleusX builds on RetNet architecture with the following changes from the paper:

- SwiGLU: Swish Gated Linear Unit instead of GeLU-FFN.
- RMSNorm instead of LayerNorm. If possible, we use `apex.normalization.FusedRMSNorm`.
- Computes KV-cache for parallel forward mode.

This model is trained with a customized llama-2 tokenizer. During training, use
`forward_mode='parallel'` or `forward_mode='chunkwise'` if the sequence length gets longer, such as 8k or 16k. For
inference, the prompt tokens should be processed with `forward_mode='parallel'` and the completions with `forward_mode='recurrent'`
for best performance. This is automatically handled when called `.generate()`.

This model was contributed by [syncdoth](<https://huggingface.co/syncdoth). The original code can be found [here](https://github.com/syncdoth/retnet).


## License

NucleusX models are released under the MIT license.

## Usage

New NucleusXModels can be instantiated as follows:

```python
>>> from transformers import NucleusXConfig, NucleusXForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
>>> config = NucleusXConfig(vocab_size=len(tokenizer),
...                         pad_token_id=tokenizer.eos_token_id,
...                         eos_token_id=tokenizer.eos_token_id,
...                         bos_token_id=tokenizer.bos_token_id)
>>> model = NucleusXForCausalLM(config)

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

## NucleusXConfig

[[autodoc]] NucleusXConfig

## NucleusXModel

[[autodoc]] NucleusXModel
    - forward


## NucleusXForCausalLM

[[autodoc]] NucleusXForCausalLM
    - forward

## NucleusXForSequenceClassification

[[autodoc]] transformers.NucleusXForSequenceClassification
    - forward
