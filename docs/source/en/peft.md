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

# PEFT

[[open-in-colab]]

[PEFT](https://huggingface.co/docs/peft/index), a library of parameter-efficient fine-tuning methods, enables training and storing large models on consumer GPUs. These methods only fine-tune a small number of extra model parameters, also known as adapters, on top of the pretrained model. A significant amount of memory is saved because the GPU doesn't need to store the optimizer states and gradients for the pretrained base model. Adapters are very lightweight, making it convenient to share, store, and load them.

This guide provides a short introduction to the PEFT library and how to use it for training with Transformers. For more details, refer to the PEFT [documentation](https://huggingface.co/docs/peft/index).

Install PEFT with the command below.

<hfoptions id="install">
<hfoption id="pip">

```bash
pip install -U peft
```

</hfoption>
<hfoption id="source">

```bash
pip install git+https://github.com/huggingface/peft.git
```

</hfoption>
</hfoptions>

> [!TIP]
> PEFT currently supports the LoRA, IA3, and AdaLoRA methods for Transformers. To use another PEFT method, such as prompt learning or prompt tuning, use the PEFT library directly.

[Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a very common PEFT method that decomposes the weight matrix into two smaller trainable matrices. Start by defining a [LoraConfig](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig) object with the parameters shown below.

```py
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

# create LoRA configuration object
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)
```

Add [LoraConfig](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig) to the model with [`~integrations.PeftAdapterMixin.add_adapter`]. The model is now ready to be passed to [`Trainer`] for training.

```py
model.add_adapter(lora_config, adapter_name="lora_1")
trainer = Trainer(model=model, ...)
trainer.train()
```

To add an additional trainable adapter on top of a model with an existing adapter attached, specify the modules you want to train in [modules_to_save()](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.modules_to_save).

For example, to train the `lm_head` module on top of a causal language model with a LoRA adapter attached, set `modules_to_save=["lm_head"]`. Add the adapter to the model as shown below, and then pass it to [`Trainer`].

```py
from transformers import AutoModelForCausalLM
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    modules_to_save=["lm_head"],
)

model.add_adapter(lora_config)
trainer = Trainer(model=model, ...)
trainer.train()
```

Save your adapter with [`~PreTrainedModel.save_pretrained`] to reuse it.

## Load adapter

To load an adapter with Transformers, the Hub repository or local directory must contain an `adapter_config.json` file and the adapter weights. Load the adapter with [`~PreTrainedModel.from_pretrained`] or with [`~integrations.PeftAdapterMixin.load_adapter`].

<hfoptions id="load">
<hfoption id="from_pretrained">

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
<hfoption id="load_adapter">

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
model.load_adapter("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
</hfoptions>

For very large models, it is helpful to load a quantized version of the model in 8 or 4-bit precision to save memory. Transformers supports quantization with its [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index) integration. Specify in [`BitsAndBytesConfig`] whether you want to load a model in 8 or 4-bit precision.

For multiple devices, add `device_map="auto"` to automatically distribute the model across your hardware.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "klcsp/gemma7b-lora-alpaca-11-v1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

## Set adapter

[`~integrations.PeftAdapterMixin.add_adapter`] adds a new adapter to a model. To add a second adapter, the new adapter must be the same type as the first adapter. Use the `adapter_name` parameter to assign a name to the adapter.

```py
model.add_adapter(lora_config, adapter_name="lora_2")
```

Once added, use [`~integrations.PeftAdapterMixin.set_adapter`] to force a model to use the specified adapter and disable the other adapters.

```py
model.set_adapter("lora_2")
```

## Enable and disable adapter

[`~integrations.PeftAdapterMixin.enable_adapters`] is a broader function that enables *all* adapters attached to a model, and [`~integrations.PeftAdapterMixin.disable_adapters`] disables *all* attached adapters.

```py
model.add_adapter(lora_1)
model.add_adapter(lora_2)
model.enable_adapters()

# disable all adapters
model.disable_adapters()
```

## Hotswapping adapters

A common use case when serving multiple adapters is to load one adapter first, generate output, load another adapter, generate more outputs, load another adapter, etc. This can be inefficient, since each time a new adapter is loaded, new memory is reserved; moreover, if the model is compiled with `torch.compile`, it needs to be re-compiled each time a new adapter is used. When switching frequently, the compilation time may never be amortized.

To better support this common workflow, you can "hotswap" a LoRA adapter, to avoid accumulating memory and, in some cases, recompilation. It requires an adapter to already be loaded, and the new adapter weights are swapped in-place for the existing adapter. Note that other PEFT methods are not supported yet, only LoRA.

Pass `hotswap=True` when loading a LoRA adapter to enable this feature. It is important to indicate the name of the existing adapter (`"default"` is the default adapter name) to be swapped.

```python
model = AutoModel.from_pretrained(...)
# load adapter 1 as normal
model.load_adapter(file_name_adapter_1)
# generate outputs with adapter 1
...
# now hotswap the 2nd adapter
model.load_adapter(file_name_adapter_2, hotswap=True, adapter_name="default")
# generate outputs with adapter 2
```

For compiled models, it is often necessary to call [`~integrations.peft.PeftAdapterMixin.enable_peft_hotswap`] to avoid recompilation. Call this method *before* loading the first adapter, while `torch.compile` should be called *after* loading the first adapter.

```python
model = AutoModel.from_pretrained(...)
max_rank = ...  # the highest rank among all LoRAs that you want to load
# call *before* compiling and loading the LoRA adapter
model.enable_peft_hotswap(target_rank=max_rank)
model.load_adapter(file_name_1, adapter_name="default")
# optionally compile the model now
model = torch.compile(model, ...)
output_1 = model(...)
# now you can hotswap the 2nd adapter, use the same name as for the 1st
model.load_adapter(file_name_2, adapter_name="default")
output_2 = model(...)
```

The `target_rank=max_rank` argument is important for setting the maximum rank among all LoRA adapters that will be loaded. If you have one adapter with rank 8 and another with rank 16, pass `target_rank=16`. You should use a higher value if in doubt. By default, this value is 128.

By default, hotswapping is disabled and requires you to pass `hotswap=True` to `load_adapter`. However, if you called `enable_peft_hotswap` first, hotswapping will be enabled by default. If you want to avoid using it, you need to pass `hotswap=False`.

However, there can be situations where recompilation is unavoidable. For example, if the hotswapped adapter targets more layers than the initial adapter, then recompilation is triggered. Try to load the adapter that targets the most layers first. Refer to the PEFT docs on [hotswapping](https://huggingface.co/docs/peft/main/en/package_reference/hotswap#peft.utils.hotswap.hotswap_adapter) for more details about the limitations of this feature.

> [!Tip]
> Move your code inside the `with torch._dynamo.config.patch(error_on_recompile=True)` context manager to detect if a model was recompiled. If you detect recompilation despite following all the steps above, please open an issue with [PEFT](https://github.com/huggingface/peft/issues) with a reproducible example.

For an example of how the use of `torch.compile` in combination with hotswapping can improve runtime, check out [this blogpost](https://huggingface.co/blog/lora-fast). Although that example uses Diffusers, similar improvements can be expected here.
