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

# RWKV

## Overview

The RWKV model was proposed in [this repo](https://github.com/BlinkDL/RWKV-LM)

It suggests a tweak in the traditional Transformer attention to make it linear. This way, the model can be used as recurrent network: passing inputs for timestamp 0 and timestamp 1 together is the same as passing inputs at timestamp 0, then inputs at timestamp 1 along with the state of timestamp 0 (see example below).

This can be more efficient than a regular Transformer and can deal with sentence of any length (even if the model uses a fixed context length for training).

This model was contributed by [sgugger](https://huggingface.co/sgugger).
The original code can be found [here](https://github.com/BlinkDL/RWKV-LM).

## Usage example

```py
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")
# Feed everything to the model
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# Using the state computed on the first inputs, we will get the same output
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

If you want to make sure the model stops generating when `'\n\n'` is detected, we recommend using the following stopping criteria:

```python 
from transformers import StoppingCriteria

class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [187,187], eos_token_id = 537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:,-2:].tolist()
        return self.eos_sequence in last_2_ids


output = model.generate(inputs["input_ids"], max_new_tokens=64, stopping_criteria = [RwkvStoppingCriteria()])
```

## RwkvConfig

[[autodoc]] RwkvConfig

## RwkvModel

[[autodoc]] RwkvModel
    - forward

## RwkvLMHeadModel

[[autodoc]] RwkvForCausalLM
    - forward

## Rwkv attention and the recurrent formulas

In a traditional auto-regressive Transformer, attention is written as

$$O = \hbox{softmax}(QK^{T} / \sqrt{d}) V$$

with \\(Q\\), \\(K\\) and \\(V\\) are matrices of shape `seq_len x hidden_size` named query, key and value (they are actually bigger matrices with a batch dimension and an attention head dimension but we're only interested in the last two, which is where the matrix product is taken, so for the sake of simplicity we only consider those two). The product \\(QK^{T}\\) then has shape `seq_len x seq_len` and we can take the matrix product with \\(V\\) to get the output \\(O\\) of the same shape as the others.  

Replacing the softmax by its value gives:

$$O_{i} = \frac{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}} V_{j}}{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}}}$$

Note that the entries in \\(QK^{T}\\) corresponding to \\(j > i\\) are masked (the sum stops at j) because the attention is not allowed to look at future tokens (only past ones).

In comparison, the RWKV attention is given by

$$O_{i} = \sigma(R_{i}) \frac{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}} V_{j}}{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}}}$$

where \\(R\\) is a new matrix called receptance by the author, \\(K\\) and \\(V\\) are still the key and value (\\(\sigma\\) here is the sigmoid function). \\(W\\) is a new vector that represents the position of the token and is given by

$$W_{0} = u \hbox{  and  } W_{k} = (k-1)w \hbox{ for } k \geq 1$$

with \\(u\\) and \\(w\\) learnable parameters called in the code `time_first` and `time_decay` respectively. The numerator and denominator can both be expressed recursively. Naming them \\(N_{i}\\) and \\(D_{i}\\) we have:

$$N_{i} = e^{u + K_{i}} V_{i} + \hat{N}_{i} \hbox{  where  } \hat{N}_{i} = e^{K_{i-1}} V_{i-1} + e^{w + K_{i-2}} V_{i-2} \cdots + e^{(i-2)w + K_{1}} V_{1}$$

so \\(\hat{N}_{i}\\) (called `numerator_state` in the code) satisfies

$$\hat{N}_{0} = 0 \hbox{  and  } \hat{N}_{j+1} = e^{K_{j}} V_{j} + e^{w} \hat{N}_{j}$$

and

$$D_{i} = e^{u + K_{i}} + \hat{D}_{i} \hbox{  where  } \hat{D}_{i} = e^{K_{i-1}} + e^{w + K_{i-2}} \cdots + e^{(i-2)w + K_{1}}$$

so \\(\hat{D}_{i}\\) (called `denominator_state` in the code) satisfies

$$\hat{D}_{0} = 0 \hbox{  and  } \hat{D}_{j+1} = e^{K_{j}} + e^{w} \hat{D}_{j}$$

The actual recurrent formula used are a tiny bit more complex, as for numerical stability we don't want to compute exponentials of big numbers. Usually the softmax is not computed as is, but the exponential of the maximum term is divided of the numerator and denominator:

$$\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}} = \frac{e^{x_{i} - M}}{\sum_{j=1}^{n} e^{x_{j} - M}}$$

with \\(M\\) the maximum of all \\(x_{j}\\). So here on top of saving the numerator state (\\(\hat{N}\\)) and the denominator state (\\(\hat{D}\\)) we also keep track of the maximum of all terms encountered in the exponentials. So we actually use

$$\tilde{N}_{i} = e^{-M_{i}} \hat{N}_{i} \hbox{  and  } \tilde{D}_{i} = e^{-M_{i}} \hat{D}_{i}$$

defined by the following recurrent formulas:

$$\tilde{N}_{0} = 0 \hbox{  and  } \tilde{N}_{j+1} = e^{K_{j} - q} V_{j} + e^{w + M_{j} - q} \tilde{N}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

and

$$\tilde{D}_{0} = 0 \hbox{  and  } \tilde{D}_{j+1} = e^{K_{j} - q} + e^{w + M_{j} - q} \tilde{D}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

and \\(M_{j+1} = q\\). With those, we can then compute

$$N_{i} = e^{u + K_{i} - q} V_{i} + e^{M_{i}} \tilde{N}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

and

$$D_{i} = e^{u + K_{i} - q} + e^{M_{i}} \tilde{D}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

which finally gives us

$$O_{i} = \sigma(R_{i}) \frac{N_{i}}{D_{i}}$$