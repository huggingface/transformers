<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EntroPack

[EntroPack](https://github.com/modelscope/entropack) compresses linear weights with lattice quantization and entropy coding to reduce model storage and memory usage. Unlike formats restricted to integer bit widths, EntroPack supports fractional target rates, such as 3.5 bits per parameter (bpp), without requiring a calibration dataset.

Transformers can compress a dense checkpoint with EntroPack as it is loaded, then save the packed weights for later use. Pre-quantized EntroPack checkpoints are detected from their `config.json` and loaded without passing another quantization config.

Install EntroPack with the extra that matches your CUDA version. Transformers requires EntroPack 0.1.3 or newer.

```bash
pip install --upgrade transformers "entropack[cuda12]"
# CUDA 13
pip install --upgrade transformers "entropack[cuda13]"
```

## Quantization examples

Pass an [`EntroPackConfig`] to [`~PreTrainedModel.from_pretrained`]. The linear weights are compressed as the checkpoint is loaded.

```py
import torch
from transformers import AutoModelForCausalLM, EntroPackConfig

model_id = "Qwen/Qwen3-1.7B"
quantization_config = EntroPackConfig(target_bpp=4.0)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)
```

Only `torch.nn.Linear` weights are compressed. Embeddings, output heads, biases, and modules excluded from conversion are left uncompressed.

### Compression settings

`target_bpp` defaults to 4.0. The default linear implementation accepts rates from 0.001 through 11 bpp. The FP8 and INT8 implementations require a rate below 8 bpp.

Use `entropack_lossy_config` for additional settings accepted by EntroPack's lossy compression backend. See the [EntroPack configuration reference](https://github.com/modelscope/entropack/blob/main/docs/en/Usage/Configuration.md#entropacklossyconfig) for the available fields. The value is a dictionary. For example, enable one round of row-wise rate-distortion optimization with `row_rdo_iterations`.

```py
quantization_config = EntroPackConfig(
    target_bpp=4.0,
    entropack_lossy_config={"row_rdo_iterations": 1},
)
```

Use `modules_to_not_convert` to keep selected modules uncompressed. Module names are matched by prefix or suffix.

```py
quantization_config = EntroPackConfig(
    target_bpp=4.0,
    modules_to_not_convert=["model.visual", "gate_proj"],
)
```

### Linear implementations

Set `linear_type` to choose how the decoded weight is used.

| `linear_type` | Layer | Computation | Minimum hardware |
|---|---|---|---|
| `"default"` | `CompressedLinear` | Reconstruct the weight and apply a regular linear operation | CUDA GPU |
| `"fp8"` | `CompressedFP8Linear` | FP8 E4M3FN weights and activations | CUDA SM8.9+ |
| `"int8"` | `CompressedINT8Linear` | Symmetric INT8 weights and activations | CUDA SM8.0+ |

```py
fp8_config = EntroPackConfig(target_bpp=4.0, linear_type="fp8")
int8_config = EntroPackConfig(target_bpp=4.0, linear_type="int8")
```

FP8 and INT8 checkpoints also store one FP32 scale per output channel.

## Serialization

Save an EntroPack model with [`~PreTrainedModel.save_pretrained`]. The packed weights are saved in safetensors format, and the EntroPack settings are included in `config.json`.

```py
from transformers import AutoTokenizer

output_dir = "Qwen3-1.7B-entropack-4bpp"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

## Loading quantized models

Load a saved EntroPack model without passing another [`EntroPackConfig`]. Transformers reads the quantization settings from `config.json`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen3-1.7B-entropack-4bpp"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

A compressed checkpoint cannot be recompressed by passing a different `target_bpp` or `linear_type` while loading it. Load the dense checkpoint again to create a model with different settings.

## Training

### PEFT

EntroPack weights are frozen, but gradients can pass through each linear implementation. This makes it possible to train adapters with [PEFT](https://github.com/huggingface/peft).

```py
from peft import LoraConfig, get_peft_model

model = get_peft_model(
    model,
    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
)
```

### DDP

PyTorch DDP cannot broadcast an EntroPack packed weight. After adding the adapters, call `prepare_entropack_model_for_ddp` before creating the [`Trainer`].

```py
from transformers.integrations.entropack import prepare_entropack_model_for_ddp

model = get_peft_model(model, lora_config)
prepare_entropack_model_for_ddp(model)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
```

### FSDP

EntroPack weights cannot be sharded because each weight is stored as one entropy-coded bitstream. Keep the compressed modules replicated and let FSDP shard the adapter parameters.

```py
import os

from transformers.integrations.entropack import get_entropack_fsdp_ignored_modules_pattern

model = get_peft_model(model, lora_config)
os.environ["FSDP_IGNORED_MODULES"] = get_entropack_fsdp_ignored_modules_pattern(model)
training_args = TrainingArguments(output_dir="output", fsdp=True)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
```

Call both distributed helpers after PEFT wraps the model so they can use the final module and parameter names.

## Limitations

- EntroPack requires an NVIDIA CUDA GPU and a matching CuPy build.
- Each forward pass entropy-decodes the stored bitstream into weights or quantized codes, which adds runtime overhead.
