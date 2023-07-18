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

[Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) methods aim to fine-tune a pretrained model while keeping the number of trainable parameters as low as possible. This is achieved by freezing the pretrained model and adding a small number of trainable parameters (the adapters) on top of it. The adapters are trained to learn task-specific information. This approach has been shown to be very memory-efficient with lower compute usage while producing results comparable to a fully fine-tuned model. 

Adapters trained with PEFT are also usually an order of magnitude smaller than the full model, making it convenient to share, store, and load them. In the image below, the adapter weights for a [`OPTForCausalLM`] model are stored on the Hub, and its weights are only ~6MB compared to the full size of the model weights, which can be ~700MB. 

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
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

## Load PEFT adapter

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

## Load in 8bit or 4bit

The `bitsandbytes` integration supports 8bit and 4bit precision data types, which are useful for loading large models because it saves memory (see the `bitsandbytes` integration [guide](./quantization#bitsandbytes-integration) to learn more). Add the `load_in_8bit` or `load_in_4bit` parameters to [`PreTrainedModel.from_pretrained`] and set `device_map="auto"` to effectively distribute the model to your hardware:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)
```

## Train a PEFT adapter

You can also train a PEFT adapter for your specific use-case with the [`Trainer`] class. For example, to train a `roberta-large` model for token classification with LoRA:

1. setup a configuration with the task type and some hyperparameters (see [`~peft.LoraConfig`] for more details about what the hyperparameters do)
2. create a [`~peft.PeftModel`] class with the `get_peft_model()` function which takes the base `roberta-large` model and the configuration
3. pass the newly created [`~peft.PeftModel`] to the [`Trainer`] to start finetuning the model

```py
model_name = "roberta-large"
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = AutoModelForTokenClassification.from_pretrained(model_name)
model = get_peft_model(model, peft_config)
trainer = Trainer(model=model, ...)
trainer.train()
```

<!--

TODO: (@younesbelkada @stevhliu)

-   From pretrained example - make sure to tell it works for auto mapping models + non-auto mapping models.
-   Link to PEFT docs for further details
-   Trainer integration - provide small snippets 
-   8-bit / 4-bit examples ?
-->
