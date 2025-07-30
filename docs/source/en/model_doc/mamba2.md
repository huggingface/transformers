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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>

# Mamba 2

[Mamba 2](https://huggingface.co/papers/2405.21060) is based on the state space duality (SSD) framework which connects structured state space models (SSMs) and attention variants. It uses a more efficient SSD algorithm that is 2-8x faster than Mamba and modifies the architecture to enable tensor parallelism and a grouped-value attention (GVA) head structure.

You can find all the original Mamba 2 checkpoints under the [State Space Models](https://huggingface.co/state-spaces) organization, but the examples shown below use [mistralai/Mamba-Codestral-7B-v0.1](https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1) because a Hugging Face implementation isn't supported yet for the original checkpoints.

> [!TIP]
> This model was contributed by [ArthurZ](https://huggingface.co/ArthurZ).
> Click on the Mamba models in the right sidebar for more examples of how to apply Mamba to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="mistralai/Mamba-Codestral-7B-v0.1",
    dtype=torch.bfloat16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", dtype=torch.bfloat16, device_map="auto")  
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")  

output = model.generate(**input_ids)  
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers-cli run --task text-generation --model mistralai/Mamba-Codestral-7B-v0.1 --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to 4-bit integers.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", dtype=torch.bfloat16, quantization_config=quantization_config, device_map="auto")
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
## Notes

- Codestral Mamba has `groups=8` which are similar to the number of kv heads in an attention-based model.
- Codestral Mamba has two different forward passes, `torch_forward` or `cuda_kernels_forward`, and their results are expected to be slightly different.
  - `torch_forward` without compilation is 3-4x faster than `cuda_kernels_forward`.
  - `cuda_kernels_forward` uses the original CUDA kernels if they're available in your environment. It is slower during prefill because it requires a "warmup run" due to the higher CPU overhead (see [these](https://github.com/state-spaces/mamba/issues/389#issuecomment-2171755306) [comments](https://github.com/state-spaces/mamba/issues/355#issuecomment-2147597457) for more details).

- There are no positional embeddings in this model, but there is an `attention_mask` and a specific logic to mask out hidden states in two places in the case of batched generation (see this [comment](https://github.com/state-spaces/mamba/issues/66#issuecomment-1863563829) for more details). This (and the addition of the reimplemented Mamba 2 kernels) results in a slight discrepancy between batched and cached generation.
 
- The SSM algorithm heavily relies on tensor contractions, which have matmul equivalents but the order of operations is slightly different. This makes the difference greater at smaller precisions. 

- Hidden states that correspond to padding tokens is shutdown in 2 places and is mostly tested with left-padding. Right-padding propagates noise down the line and is not guaranteed to yield satisfactory results. `tokenizer.padding_side = "left"` ensures you are using the correct padding side.

- The example below demonstrates how to fine-tune Mamba 2 with [PEFT](https://huggingface.co/docs/peft).

```python 
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "mistralai/Mamba-Codestral-7B-v0.1"
dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = SFTConfig(dataset_text_field="quote", gradient_checkpointing=True, per_device_train_batch_size=4)
lora_config =  LoraConfig(target_modules=["x_proj", "embeddings", "in_proj", "out_proj"])
trainer = SFTTrainer(
    model=model_id,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)
trainer.train()
```


## Mamba2Config

[[autodoc]] Mamba2Config

## Mamba2Model

[[autodoc]] Mamba2Model
    - forward

## Mamba2LMHeadModel

[[autodoc]] Mamba2ForCausalLM
    - forward
