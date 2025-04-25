<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Granite

[Granite](https://huggingface.co/ibm-granite) model was proposed in [Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler](https://arxiv.org/abs/2408.13359) by Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya Prasad, Adriana Meza Soria, David D. Cox, and Rameswar Panda.

Granite model is a series of AI language models built for businesses, not just for general use. It is a super-focused tool for enterprise tasks like summarizing reports, answering questions, or generating code. It is trained on curated datasets (think finance, legal, code, and academic stuff) and filters out junk like duplicates or harmful content with their HAP detector. It comes in different sizes, from super lightweight (sub-billion parameters) up to 34 billion, so you can pick what fits your needs.

You can find all the original Granite checkpoints under the [Granite](https://huggingface.co/ibm-granite) collection.

> [!TIP]
> Click on the Granite models in the right sidebar for more examples of how to apply Granite to different language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="ibm-granite/granite-3.3-8b-instruct",
    tokenizer=tokenizer,
    device_map = "auto",
    torch_dtype = torch.bfloat16,

)

prompt = "Explain quantum computing in simple terms:"
output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ibm/PowerLM-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

# Change input text as desired
prompt = "Write a code to find the maximum value in a list of numbers."

# tokenize the text
input_tokens = tokenizer(prompt, return_tensors="pt")
# generate output tokens
output = model.generate(**input_tokens, max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example, the batch size is 1
for i in output:
    print(i)
```
This model was contributed by [mayank-mishra](https://huggingface.co/mayank-mishra).
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    quantization_config=quantization_config,
)

input_text = "Explain artificial intelligence to a 10 year old"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

## Notes

- Granite models support context lengths up to 128K tokens (e.g., in Granite-3.3), thanks to Rotary Position Embeddings     (RoPE). For long-context tasks like document analysis, ensure your input sequences are padded appropriately to avoid      attention issues.
- Don’t use the torch_dtype parameter in from_pretrained() if you’re using FlashAttention-2, as it only supports fp16 or    bf16. Granite models are optimized for mixed precision, so use Automatic Mixed Precision; set fp16 or bf16 to True if 
  using Trainer, or use torch.autocast.
  
## GraniteConfig

[[autodoc]] GraniteConfig

## GraniteModel

[[autodoc]] GraniteModel
    - forward

## GraniteForCausalLM

[[autodoc]] GraniteForCausalLM
    - forward
