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

# Fine-tune a model

Quantized weights can't be updated by gradients directly, so you fine-tune a quantized model by training a small adapter on top of the frozen quantized weights. Training a [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) adapter on a 4-bit model is known as QLoRA, and it lets you fine-tune a model that wouldn't fit in memory in bf16.

[`Trainer`] checks the model before training to make sure:

- The quantization method supports training. See the Fine-tune column in [Choose a method](./selecting).
- The model has a trainable adapter attached. Methods that support quantization-aware training are the exception, covered [below](#quantization-aware-training).
- The model isn't compiled with `torch.compile`.

## Train an adapter

Install [PEFT](https://huggingface.co/docs/peft/index).

```bash
pip install -U peft
```

Load the model with a method that supports training, attach a LoRA adapter with [`~integrations.PeftAdapterMixin.add_adapter`], and pass the model to [`Trainer`]. The example below uses 4-bit [bitsandbytes](./bitsandbytes) and assumes `train_dataset` is a tokenized dataset.

```py
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=quantization_config,
    device_map="auto",
)

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
model.add_adapter(lora_config)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="qwen3-qlora", bf16=True),
    train_dataset=train_dataset,
)
trainer.train()
```

Only the adapter weights train. The quantized base weights stay frozen.

## Save the adapter

[`~PreTrainedModel.save_pretrained`] saves only the adapter weights, which are a few megabytes. This works with every method, including ones that can't save their quantized weights.

```py
model.save_pretrained("qwen3-qlora-adapter")
```

To use the adapter, load the base model with the same quantization config and load the adapter on top of it. See [Loading an adapter](../peft#loading-an-adapter) for details.

## Quantization-aware training

[BitNet](./bitnet) and [compressed-tensors](./compressed_tensors) can train the full model without an adapter, as long as the weights stay in a trainable form.

- BitNet needs `linear_class="autobitlinear"` and `quantization_mode="online"` in [`BitNetQuantConfig`].
- compressed-tensors needs a checkpoint that isn't compressed, or `dequantize=True` in [`CompressedTensorsConfig`]. Checkpoints that use the FP8 kernels can't train.

See [Quantization concepts](./concept_guide) for how quantization-aware training differs from post-training quantization.
