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

# FalconMamba

## Overview

The FalconMamba model was proposed by TII UAE (Technology Innovation Institute) in their release.

The abstract from the paper is the following:

*We present FalconMamba, a new base large language model based on the novel Mamba architecture. FalconMamba is trained on 5.8 trillion tokens with carefully selected data mixtures. As a pure Mamba-based model, FalconMamba surpasses leading open-weight models based on Transformers, such as Mistral 7B, Llama3 8B, and Falcon2 11B. It is on par with Gemma 7B and outperforms models with different architecture designs, such as RecurrentGemma 9B. Currently, FalconMamba is the best-performing Mamba model in the literature at this scale, surpassing both existing Mamba and hybrid Mamba-Transformer models.
Due to its architecture, FalconMamba is significantly faster at inference and requires substantially less memory for long sequence generation. Despite recent studies suggesting that hybrid Mamba-Transformer models outperform pure architecture designs, we argue and demonstrate that the pure Mamba design can achieve similar, even superior results compared to the hybrid design. We make the weights of our implementation of FalconMamba publicly available under a permissive license.*

Tips:

- FalconMamba is mostly based on Mamba architecture, the same [tips and best practices](./mamba) would be relevant here.

The model has been trained on approximtely 6T tokens consisting a mixture of many data sources such as RefineWeb, Cosmopedia and Math data.

For more details about the training procedure and the architecture, have a look at [the technical paper of FalconMamba]() (coming soon).

# Usage

Below we demonstrate how to use the model:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b")

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

The architecture is also compatible with `torch.compile` for faster generation:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b", torch_dtype=torch.bfloat16).to(0)
model = torch.compile(model)

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

If you have access to a GPU that is compatible with `bitsandbytes`, you can also quantize the model in 4-bit precision:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b", quantization_config=quantization_config)

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

You can also play with the instruction fine-tuned model:

```python 
from transformers import FalconMambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b-instruct")
model = FalconMambaForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b-instruct")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## FalconMambaConfig

[[autodoc]] FalconMambaConfig

## FalconMambaModel

[[autodoc]] FalconMambaModel
    - forward

## FalconMambaLMHeadModel

[[autodoc]] FalconMambaForCausalLM
    - forward
