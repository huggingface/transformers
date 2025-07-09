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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow&logoColor=white">
        <img alt="Transformers" src="https://img.shields.io/badge/🤗%20Transformers-4.0+-blue">
    </div>
</div>

# Switch Transformers

[Switch Transformers](https://huggingface.co/papers/2101.03961) is a Mixture of Experts (MoE) model trained on Masked Language Modeling (MLM) tasks. The model architecture builds upon the classic T5 encoder-decoder framework, but replaces the traditional Feed Forward layers with **Sparse MLP layers containing "experts"** for improved efficiency and scaling.

<ins>Key Features</ins>

- **Sparse Expert Routing**: Uses a learned routing mechanism to selectively activate only a subset of experts per token
- **Scalable Architecture**: Enables training larger models with the same computational budget
- **T5-Compatible**: Maintains compatibility with T5 tokenizer and generation methods
- **Efficient Inference**: Activates only a fraction of parameters during inference

Switch Transformers come in different sizes:

- **switch-base-8**: 8 experts, ~223M parameters
- **switch-base-16**: 16 experts, ~415M parameters  
- **switch-base-32**: 32 experts, ~799M parameters
- **switch-base-64**: 64 experts, ~1.6B parameters

You can find all the original Switch Transformers checkpoints under the [Switch Transformers](https://huggingface.co/collections/google/switch-transformers-65f9e0b4e3f5b0af6f4dbc6c) collection.


> [!TIP]
> This model was contributed by [Younes Belkada](https://huggingface.co/ybelkada) and [Arthur Zucker](https://huggingface.co/ArthurZ).
>
> Click on the Switch Transformers models in the right sidebar for more examples of how to apply Switch Transformers to different natural language tasks.

The example below demonstrates how to do Masked Language Modeling with the [`Pipeline`] and the [`AutoModel`] classes, or with `transfomer-cli`.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Text generation
generator = pipeline(
    task="text2text-generation", 
    model="google/switch-base-8"
)
result = generator("The capital of France is <extra_id_0>.")
print(result)
# [{'generated_text': 'Paris.'}]
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")

# Prepare input prompt
input_text = "The capital of France is <extra_id_0>."
input_ids = tokenizer(
    input_text, 
    return_tensors="pt"
).input_ids

# Generate output
outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=4, 
    early_stopping=True
)
result = tokenizer.decode(
    outputs[0], 
    skip_special_tokens=True
)
print(result)  
# Paris.
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "The capital of France is <extra_id_0>." | transformers run --task text2text-generation --model google/switch-base-8 --device 0
# [{'generated_text': 'Paris.'}]
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](https://huggingface.co/docs/bitsandbytes/v0.46.0/index) to only quantize the weights to int8.

```py
# pip install bitsandbytes
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# Define the quantization configuration using BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare input prompt
input_text = "The capital of France is <extra_id_0>."
input_ids = tokenizer(
    input_text,
    return_tensors="pt"
).input_ids.to("cuda")

# Generate output
outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=4,
    early_stopping=True
)
result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
print(result)
# Paris.
```


## Notes

### Model Architecture

Switch Transformers implements a **Mixture of Experts (MoE)** architecture with the following components:

- **Router Network**: Learns to route tokens to appropriate experts
- **Expert Layers**: Sparse feed-forward networks that specialize in different aspects
- **Load Balancing**: Ensures balanced utilization across experts
- **Capacity Factor**: Controls how many tokens each expert can process

### Training

The model was trained using:
- **Objective**: Span-based masked language modeling (similar to T5)
- **Dataset**: C4 (Colossal Clean Crawled Corpus)
- **Optimization**: Expert-specific load balancing and auxiliary losses
- **Scaling**: Increased number of experts while maintaining computational efficiency

### Tips and Recommendations

> [!TIP]
> **Memory Considerations**: Switch Transformers can be memory-intensive due to the multiple expert layers. Consider using smaller variants (switch-base-8) for resource-constrained environments.

> [!TIP]
> **Inference Optimization**: Only a subset of experts are activated during inference, making the model more efficient than dense alternatives of similar quality.

> [!TIP]
> **Fine-tuning**: When fine-tuning Switch Transformers, be aware that the routing mechanism may need adjustment. Consider using lower learning rates for the router parameters.

> [!TIP]
> **Tokenizer Compatibility**: Switch Transformers use the same tokenizer as T5, so existing T5 preprocessing pipelines can be directly applied.

> [!WARNING]
> **Expert Utilization**: Monitor expert utilization during training to ensure balanced load distribution. Unbalanced expert usage can lead to reduced model performance.


## Resources

- **[All Switch Transformers Checkpoints](https://huggingface.co/models?filter=switch-transformers)**: Browse available model variants
- **[Mixture of Experts Guide](https://huggingface.co/blog/moe)**: Learn more about MoE architectures


## SwitchTransformersConfig

[[autodoc]] SwitchTransformersConfig

## SwitchTransformersTop1Router

[[autodoc]] SwitchTransformersTop1Router
    - _compute_router_probabilities
    - forward

## SwitchTransformersSparseMLP

[[autodoc]] SwitchTransformersSparseMLP
    - forward

## SwitchTransformersModel

[[autodoc]] SwitchTransformersModel
    - forward

## SwitchTransformersForConditionalGeneration

[[autodoc]] SwitchTransformersForConditionalGeneration
    - forward

## SwitchTransformersEncoderModel

[[autodoc]] SwitchTransformersEncoderModel
    - forward
