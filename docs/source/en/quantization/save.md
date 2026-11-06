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

# Save and share a model

Saving a model after quantizing it while loading lets you, and anyone you share it with, skip the quantization step next time. Calibration methods like [GPTQ](./gptq) benefit the most because calibration can take minutes to hours.

## Save the model

Call [`~PreTrainedModel.save_pretrained`] on the quantized model to write the quantized weights and add the `quantization_config` to `config.json`.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=quantization_config,
    device_map="auto",
)
model.save_pretrained("Qwen3-0.6B-bnb-4bit")
```

Save the tokenizer to the same directory so the checkpoint is complete.

## Push the model to the Hub

Call [`~PreTrainedModel.push_to_hub`] to upload the quantized weights and config to a Hub repository. Log in with `hf auth login` first.

```py
model.push_to_hub("Qwen3-0.6B-bnb-4bit")
```

## Load the saved model

Load the saved checkpoint with [`~PreTrainedModel.from_pretrained`] and no quantization config. Transformers reads the method from the `quantization_config` in `config.json` and loads the weights already quantized.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-username/Qwen3-0.6B-bnb-4bit", device_map="auto")
```

The method's library must be installed wherever the checkpoint is loaded.

## Methods that can't save

Some methods can't save their quantized weights, and [`~PreTrainedModel.save_pretrained`] raises a `ValueError` for them.

| Method | Why |
|---|---|
| [GGUF](./gguf) | Weights load from a GGUF file. Save in GGUF format with [llama.cpp](https://github.com/ggml-org/llama.cpp) tools instead. |
| [NVFP4](./nvfp4) | Saving isn't supported. Quantize the model again when you load it. |
| [Quanto](./quanto) | Saving isn't supported through Transformers. |
| [Quark](./quark) | Saving isn't supported through Transformers. |
| [AWQ](./awq) with an ExLlama backend | Saving isn't supported with the ExLlama backends. Load the model with another backend to save it. |

A quantized model with a [PEFT](../peft) adapter attached is an exception. [`~PreTrainedModel.save_pretrained`] saves only the adapter weights, so it works with any method.

To save a model in full precision instead, dequantize it first. See [Dequantize a model](./dequantize).
