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

# Load adapters with 🤗 PEFT

[[open-in-colab]]

[Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) methods freeze the pretrained model parameters during fine-tuning and add a small number of trainable parameters (the adapters) on top of it. The adapters are trained to learn task-specific information. This approach has been shown to be very memory-efficient with lower compute usage while producing results comparable to a fully fine-tuned model. 

Adapters trained with PEFT are also usually an order of magnitude smaller than the full model, making it convenient to share, store, and load them.

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">The adapter weights for a OPTForCausalLM model stored on the Hub are only ~6MB compared to the full size of the model weights, which can be ~700MB.</figcaption>
</div>

If you're interested in learning more about the 🤗 PEFT library, check out the [documentation](https://huggingface.co/docs/peft/index).

## Setup

Get started by installing 🤗 PEFT:

```bash
pip install peft
```

If you want to try out the brand new features, you might be interested in installing the library from source:

```bash
pip install git+https://github.com/huggingface/peft.git
```

## Supported PEFT models

🤗 Transformers natively supports some PEFT methods, meaning you can load adapter weights stored locally or on the Hub and easily run or train them with a few lines of code. The following methods are supported:

- [Low Rank Adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)

If you want to use other PEFT methods, such as prompt learning or prompt tuning, or about the 🤗 PEFT library in general, please refer to the [documentation](https://huggingface.co/docs/peft/index).


## Load a PEFT adapter

To load and use a PEFT adapter model from 🤗 Transformers, make sure the Hub repository or local directory contains an `adapter_config.json` file and the adapter weights, as shown in the example image above. Then you can load the PEFT adapter model using the `AutoModelFor` class. For example, to load a PEFT adapter model for causal language modeling:

1. specify the PEFT model id
2. pass it to the [`AutoModelForCausalLM`] class

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

<Tip>

You can load a PEFT adapter with either an `AutoModelFor` class or the base model class like `OPTForCausalLM` or `LlamaForCausalLM`.

</Tip>

You can also load a PEFT adapter by calling the `load_adapter` method:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

## Load in 8bit or 4bit

The `bitsandbytes` integration supports 8bit and 4bit precision data types, which are useful for loading large models because it saves memory (see the `bitsandbytes` integration [guide](./quantization#bitsandbytes-integration) to learn more). Add the `load_in_8bit` or `load_in_4bit` parameters to [`~PreTrainedModel.from_pretrained`] and set `device_map="auto"` to effectively distribute the model to your hardware:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)
```

## Add a new adapter

You can use [`~peft.PeftModel.add_adapter`] to add a new adapter to a model with an existing adapter as long as the new adapter is the same type as the current one. For example, if you have an existing LoRA adapter attached to a model:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

To add a new adapter:

```py
# attach new adapter with same config
model.add_adapter(lora_config, adapter_name="adapter_2")
```

Now you can use [`~peft.PeftModel.set_adapter`] to set which adapter to use:

```py
# use adapter_1
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))

# use adapter_2
model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## Enable and disable adapters

Once you've added an adapter to a model, you can enable or disable the adapter module. To enable the adapter module:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

# to initiate with random weights
peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```

To disable the adapter module:

```py
model.disable_adapters()
output = model.generate(**inputs)
```

## Train a PEFT adapter

PEFT adapters are supported by the [`Trainer`] class so that you can train an adapter for your specific use case. It only requires adding a few more lines of code. For example, to train a LoRA adapter:

<Tip>

If you aren't familiar with fine-tuning a model with [`Trainer`], take a look at the [Fine-tune a pretrained model](training) tutorial.

</Tip>

1. Define your adapter configuration with the task type and hyperparameters (see [`~peft.LoraConfig`] for more details about what the hyperparameters do).

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

2. Add adapter to the model.

```py
model.add_adapter(peft_config)
```

3. Now you can pass the model to [`Trainer`]!

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

To save your trained adapter and load it back:

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

<!--
TODO: (@younesbelkada @stevhliu)
-   Link to PEFT docs for further details
-   Trainer  
-   8-bit / 4-bit examples ?
-->
