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

# Parameter-efficient fine-tuning

[Parameter-efficient fine-tuning (PEFT)](https://huggingface.co/docs/peft/index) methods only fine-tune a small number of extra model parameters (adapters) on top of a pretrained model. Because only adapter parameters are updated, the optimizer tracks far fewer gradients and states, reducing memory usage significantly. Adapters are lightweight, making them convenient to share, store, and load.

Transformers integrates directly with the PEFT library through [`~integrations.PeftAdapterMixin`], added to all [`PreTrainedModel`] classes. You can load, add, train, switch, and delete adapters without wrapping your model in a separate [`~peft.PeftModel`]. All non-prompt-learning PEFT methods are supported (LoRA, IA3, AdaLoRA). Prompt-based methods like prompt tuning and prefix tuning require using the [PEFT library](https://huggingface.co/docs/peft/index) directly.

Install PEFT to get started. The integration requires `peft >= 0.18.0`.

```shell
pip install -U peft
```

## Add an adapter

Create a PEFT config, like [`~peft.LoraConfig`] for example, and attach it to a model with [`~integrations.PeftAdapterMixin.add_adapter`].

```py
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model.add_adapter(lora_config, adapter_name="my_adapter")
```

To train additional modules alongside an adapter (for example, the language model head), specify them in `modules_to_save`.

```py
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    modules_to_save=["lm_head"],
)

model.add_adapter(lora_config)
```

## Training

Pass the model with an attached adapter to [`Trainer`] and call [`~Trainer.train`]. [`Trainer`] only updates the adapter parameters (those with `requires_grad=True`) because the base model is frozen.

```py
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

During training, [`Trainer`] checkpoints contain only the adapter weights (`adapter_model.safetensors`) and configuration (`adapter_config.json`), keeping checkpoints small. The base model isn't included.

After training, save the final adapter with [`~PreTrainedModel.save_pretrained`].

```py
model.save_pretrained("./my_adapter")
```

### Resuming from a checkpoint

[`Trainer`] automatically detects adapter checkpoints when resuming. [`Trainer`] scans the checkpoint directory for subdirectories containing adapter weights and reloads each adapter with the correct trainable state.

```py
trainer.train(resume_from_checkpoint="./output/checkpoint-1000")
```

### Distributed training

PEFT adapters work with distributed training out of the box.

For ZeRO-3, [`Trainer`] passes `exclude_frozen_parameters=True` when saving checkpoints with a PEFT model. Frozen base model weights are skipped. Only the trainable adapter parameters are saved, reducing checkpoint size and save time.

For FSDP, [`Trainer`] updates the FSDP auto-wrap policy to correctly handle LoRA layers. For QLoRA (quantized base model + LoRA), [`Trainer`] also adjusts the mixed precision policy to match the quantization storage dtype.

## Loading an adapter

To load an adapter, the Hub repository or local directory must contain an `adapter_config.json` file and the adapter weights.

<hfoptions id="load">
<hfoption id="from_pretrained">

[`~PreTrainedModel.from_pretrained`] automatically detects adapters. When it finds an `adapter_config.json`, it reads the `base_model_name_or_path` field to load the correct base model, then loads the adapter on top.

```py
from transformers import AutoModelForCausalLM

# Automatically loads the base model and attaches the adapter
model = AutoModelForCausalLM.from_pretrained("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
<hfoption id="load_adapter">

To load an adapter onto an existing model, use [`~integrations.PeftAdapterMixin.load_adapter`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
model.load_adapter("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
</hfoptions>

For large models, load a quantized version in 8-bit or 4-bit precision with [bitsandbytes](./quantization/bitsandbytes) to save memory. Add `device_map="auto"` to distribute the model across available hardware.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "klcsp/gemma7b-lora-alpaca-11-v1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

### Loading from a state dict

To load an adapter programmatically from a state dict (without reading from disk), pass `adapter_state_dict` and `peft_config` directly.

```py
model.load_adapter(
    None,
    adapter_name="my_adapter",
    peft_config=peft_config_dict,
    adapter_state_dict=adapter_state_dict,
)
```

## Managing multiple adapters

A model can hold multiple adapters at once. Add adapters with unique names, and switch between them as needed.

```py
from peft import LoraConfig

model.add_adapter(LoraConfig(r=8, lora_alpha=32), adapter_name="adapter_1")
model.add_adapter(LoraConfig(r=16, lora_alpha=64), adapter_name="adapter_2")
```

Use [`~integrations.PeftAdapterMixin.set_adapter`] to activate a specific adapter. The other adapters are disabled but remain in memory.

```py
model.set_adapter("adapter_2")
```

[`~integrations.PeftAdapterMixin.enable_adapters`] enables all attached adapters, and [`~integrations.PeftAdapterMixin.disable_adapters`] disables all of them.

```py
# Disable all adapters for base model inference
model.disable_adapters()

# Re-enable all adapters
model.enable_adapters()
```

Use [`~integrations.PeftAdapterMixin.active_adapters`] to see which adapters are currently active.

```py
model.active_adapters()
# ["adapter_1"]
```

Remove adapters you no longer need with [`~integrations.PeftAdapterMixin.delete_adapter`] to free memory.

```py
model.delete_adapter("adapter_1")
```

## Hotswapping adapters

Loading a new adapter each time you serve a request allocates new memory. If the model is compiled with `torch.compile`, each new adapter triggers recompilation. Hotswapping replaces adapter weights in-place, avoiding both issues. Only LoRA adapters are supported.

Pass `hotswap=True` when loading a LoRA adapter to swap its weights into an existing adapter slot. Set `adapter_name` to the name of the adapter to replace (`"default"` is the default adapter name).

```py
model = AutoModel.from_pretrained(...)
# Load the first adapter normally
model.load_adapter(adapter_path_1)
# Generate outputs with adapter 1
...
# Hotswap the second adapter in-place
model.load_adapter(adapter_path_2, hotswap=True, adapter_name="default")
# Generate outputs with adapter 2
```

### torch.compile

For compiled models, call [`~integrations.peft.PeftAdapterMixin.enable_peft_hotswap`] *before* loading the first adapter and before compiling.

```py
model = AutoModel.from_pretrained(...)
max_rank = ...  # highest rank among all LoRAs you'll load
model.enable_peft_hotswap(target_rank=max_rank)
model.load_adapter(adapter_path_1, adapter_name="default")
model = torch.compile(model, ...)
output_1 = model(...)

# Hotswap without recompilation
model.load_adapter(adapter_path_2, adapter_name="default")
output_2 = model(...)
```

The `target_rank` argument sets the maximum rank among all LoRA adapters you'll load. If you have adapters with rank 8 and rank 16, pass `target_rank=16`. The default is 128.

After calling `enable_peft_hotswap`, all subsequent `load_adapter` calls hotswap by default. Pass `hotswap=False` explicitly to disable hotswapping.

Recompilation may still occur if the hotswapped adapter targets more layers than the initial adapter. Load the adapter that targets the most layers first to avoid recompilation.

> [!TIP]
> Wrap your code in `with torch._dynamo.config.patch(error_on_recompile=True)` to detect unexpected recompilation. If you detect recompilation despite following the steps above, open an issue with [PEFT](https://github.com/huggingface/peft/issues) with a reproducible example.

## Next steps

- The PEFT [documentation](https://huggingface.co/docs/peft/index) covers the full range of PEFT methods and options.
- The PEFT [hotswapping reference](https://huggingface.co/docs/peft/main/en/package_reference/hotswap#peft.utils.hotswap.hotswap_adapter) details limitations and edge cases.
- A [blog post](https://huggingface.co/blog/lora-fast) benchmarks how `torch.compile` with hotswapping improves runtime.
