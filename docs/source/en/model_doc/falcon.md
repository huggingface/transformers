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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Falcon

[Falcon](https://huggingface.co/papers/2311.16867) is a family of large language models, available in 7B, 40B, and 180B parameters, as pretrained and instruction tuned variants. This model focuses on scaling pretraining over three categories, performance, data, and hardware. Falcon uses multigroup attention to significantly reduce inference memory requirements and rotary positional embeddings (RoPE). These models are pretrained on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), a high-quality and deduplicated 5T token dataset.

You can find all the original Falcon checkpoints under the [Falcon](https://huggingface.co/collections/tiiuae/falcon-64fb432660017eeec9837b5a) collection.

> [!TIP]
> Click on the Falcon models in the right sidebar for more examples of how to apply Falcon to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

# For text generation with Falcon-7B-Instruct
generator = pipeline(
    "text-generation", 
    model="tiiuae/falcon-7b-instruct",
    device_map="auto"
)
response = generator(
    "Write a short poem about coding",
    max_length=100,
    do_sample=True,
    temperature=0.7
)
print(response[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
model_id = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Load in bfloat16 for faster inference
    device_map="auto",  # Automatically determine device mapping
    attn_implementation="sdpa", # Use scaled dot product attention
)

# Prepare input
text = "Write a function in Python to calculate the Fibonacci sequence:"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

# Decode and print result
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli chat --model_name_or_path tiiuae/falcon-7b-instruct
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization]../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```python
# Make sure to have bitsandbytes available in the environment
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer and model with quantization
model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)

# Generate text with the quantized model
inputs = tokenizer("In quantum physics, entanglement means", return_tensors="pt").to(model.device)
outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- Falcon models come in different sizes (7B, 40B, 180B parameters) and variants (base and instruct).
- The "instruct" versions have been fine-tuned on instruction data and are better for conversational or instruction-following tasks.
- For most applications, using FlashAttention or SDPA optimization is recommended for the best performance.
- If you're upgrading from an older custom code checkpoint, remember to convert it to the official Transformers format using the conversion script located in the
[Falcon model directory](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon) to benefit from improved stability and performance. 

## FalconConfig

[[autodoc]] FalconConfig
    - all

## FalconModel

[[autodoc]] FalconModel
    - forward

## FalconForCausalLM

[[autodoc]] FalconForCausalLM
    - forward

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification
    - forward

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification
    - forward

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering
    - forward