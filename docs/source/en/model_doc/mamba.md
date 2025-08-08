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
</div>

# Mamba

[Mamba](https://huggingface.co/papers/2312.00752) is a selective structured state space model (SSMs) designed to work around Transformers computational inefficiency when dealing with long sequences.  It is a completely attention-free architecture, and comprised of a combination of H3 and gated MLP blocks (Mamba block). Mamba's "content-based reasoning" allows it to focus on specific parts of an input depending on the current token. Mamba also uses a new hardware-aware parallel algorithm to compensate for the lack of convolutional operations. As a result, Mamba has fast inference and can scale to very long sequences.

You can find all the original Mamba checkpoints under the [State Space Models](https://huggingface.co/state-spaces) organization.


> [!TIP]
> This model was contributed by [Molbap](https://huggingface.co/Molbap) and [AntonV](https://huggingface.co/AntonV).
> Click on the Mamba models in the right sidebar for more examples of how to apply Mamba to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="state-spaces/mamba-130m-hf",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype=torch.float16, device_map="auto",)  
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")  

output = model.generate(**input_ids)  
print(tokenizer.decode(output[0], skip_special_tokens=True)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model state-spaces/mamba-130m-hf --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to 4-bit integers.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int4WeightOnlyConfig

quantization_config = Int4WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf", dtype=torch.bfloat16, quantization_config=quantization_config, device_map="auto",)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
## Notes

- The current implementation uses the original CUDA kernels. The FlashAttention equivalent implementation is hosted in the [mamba-ssm](https://github.com/state-spaces/mamba) and [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d) repositories. Make sure to install them if your hardware supports it!
- Mamba stacks `mixer` layers which are equivalent to `Attention` layers. You can find the main logic of Mamba in the `MambaMixer` class.
- The example below demonstrates how to fine-tune Mamba with [PEFT](https://huggingface.co/docs/peft).

  ```py
  from datasets import load_dataset
  from trl import SFTConfig, SFTTrainer
  from peft import LoraConfig

  model_id = "state-spaces/mamba-130m-hf"
  dataset = load_dataset("Abirate/english_quotes", split="train")
  training_args = SFTConfig(dataset_text_field="quote")
  lora_config =  LoraConfig(target_modules=["x_proj", "embeddings", "in_proj", "out_proj"])
  trainer = SFTTrainer(
      model=model_id,
      args=training_args,
      train_dataset=dataset,
      peft_config=lora_config,
  )
  trainer.train()
   ```

## MambaCache

[[autodoc]] MambaCache
    - update_conv_state
    - update_ssm_state
    - reset

## MambaConfig

[[autodoc]] MambaConfig

## MambaModel

[[autodoc]] MambaModel
    - forward

## MambaLMHeadModel

[[autodoc]] MambaForCausalLM
    - forward
