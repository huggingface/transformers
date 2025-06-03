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

# GPTNeoX

GPTNeoX is a 20 billion parameter autoregressive language model that represents a breakthrough in open-source large language models. What makes GPTNeoX unique is its use of rotary positional embeddings (RoPE) instead of learned positional embeddings, allowing for better extrapolation to longer sequences than traditional transformer models. It also employs parallel attention and feedforward layers, making it more efficient during both training and inference.

Developed by EleutherAI and trained on the comprehensive Pile dataset, GPTNeoX delivers particularly strong few-shot reasoning capabilities that often exceed similarly sized models like GPT-3. At the time of its release, it was the largest dense autoregressive model with publicly available weights.

The original paper can be found [here](https://hf.co/papers/2204.06745), and you can find the official checkpoints on the [Hugging Face Hub](https://huggingface.co/EleutherAI/gpt-neox-20b).

<Tip>

Click on the right sidebar for more examples of how to use GPTNeoX for other tasks!

</Tip>

## Usage

```python
from transformers import pipeline

# Text generation with pipeline
generator = pipeline("text-generation", model="EleutherAI/gpt-neox-20b")
result = generator("The future of artificial intelligence is", max_length=50, num_return_sequences=1)
print(result)
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Using AutoModel for more control
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)

# Generate text
inputs = tokenizer("The future of artificial intelligence is", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

```bash
# Using transformers-cli
transformers-cli env
```

### Quantization Example

For easier deployment on consumer hardware, you can use quantization:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load with 8-bit quantization
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neox-20b",
    load_in_8bit=True,
    device_map="auto"
)

inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

GPTNeoX uses rotary positional embeddings (RoPE) instead of learned positional embeddings, which allows for better extrapolation to longer sequences. The model also employs parallel attention and feedforward layers, making it more efficient during training.

## GPTNeoXConfig

[[autodoc]] GPTNeoXConfig

## GPTNeoXTokenizerFast

[[autodoc]] GPTNeoXTokenizerFast

## GPTNeoXModel

[[autodoc]] GPTNeoXModel
    - forward

## GPTNeoXForCausalLM

[[autodoc]] GPTNeoXForCausalLM
    - forward

## GPTNeoXForQuestionAnswering

[[autodoc]] GPTNeoXForQuestionAnswering
    - forward

## GPTNeoXForSequenceClassification

[[autodoc]] GPTNeoXForSequenceClassification
    - forward

## GPTNeoXForTokenClassification

[[autodoc]] GPTNeoXForTokenClassification
    - forward