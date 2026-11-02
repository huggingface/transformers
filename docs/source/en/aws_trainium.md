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

# AWS Trainium

[AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) instances run PyTorch natively through the `torch-neuronx` package, which registers the `neuron` device. Neuron compiles your model into one or more graphs for each input shape it sees, so keeping input shapes static avoids recompiling.

## Installation

Install `torch-neuronx` by following the [AWS Neuron setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/index.html). Then check that PyTorch sees the Neuron devices.

```py
import torch

print(torch.neuron.is_available(), torch.neuron.device_count())
```

It prints `True` and the number of Neuron devices, for example `True 4`.

## Run inference

`device_map="auto"` doesn't detect Neuron. Automatic placement relies on [Accelerate](https://huggingface.co/docs/accelerate/index) to list the available devices, and Accelerate doesn't list Neuron devices, so the model silently loads on CPU. Pass `device_map="neuron"` to [`~PreTrainedModel.from_pretrained`] explicitly, or `device="neuron"` to [`Pipeline`].

Pass the generation inputs on CPU instead of calling `.to(model.device)`. [`~GenerationMixin.generate`] moves only the tensors that `forward` consumes to the Neuron device. The tensors that grow each step (output sequence, `attention_mask`) stay on CPU, so the compiled graph stays stable. `generate` returns the output on the same device as the inputs. See [Keep generation tensors on CPU](./kv_cache#keep-generation-tensors-on-cpu) for more details.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="neuron"
)

# leave the inputs on CPU
inputs = tokenizer("The French Bread Law states", return_tensors="pt")
output = model.generate(
    **inputs, do_sample=False, max_new_tokens=50, cache_implementation="static"
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```txt
The French Bread Law states that the French government must provide a certain amount of bread for the citizens of the country. The law is passed by the French parliament. The French parliament is a body of the French government. The French government is a body of the French state. The
```

With a static cache, [`~GenerationMixin.generate`] compiles the forward pass with `torch.compile` automatically on Neuron. Pass `disable_compile=True` to [`~GenerationMixin.generate`] to opt out.

## Train with Trainer

[`TrainingArguments`] detects Neuron and sets the device to `neuron` automatically. All Trainium generations support bf16.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
```

### Compile the model

Set `torch_compile=True` to wrap the model with `torch.compile`. This compiles the model's forward pass, while the optimizer step runs eagerly. [`TrainingArguments`] sets `torch_compile_backend` to `"neuron"` automatically when it detects Neuron.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    torch_compile=True,
)
```

Each new input shape triggers a new compilation. Pad inputs to a fixed length (`padding="max_length"` in the tokenizer) and set `dataloader_drop_last=True` so the last batch doesn't have a different size.

### Distributed training

Neuron supports the Transformers distributed API. Create a [`DistributedConfig`] with the parallelism sizes you want and pass it to [`~PreTrainedModel.from_pretrained`]. Transformers shards the model at load time, and [`Trainer`] uses it as is.

```py
import torch
from transformers import AutoModelForCausalLM, DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=2, fsdp_size=2),
)
```

Launch the script with `torchrun`.

```bash
torchrun --nproc_per_node=4 train.py
```

See [Tensor parallelism](./tensor_parallelism), [Expert parallelism](./expert_parallelism), and [FSDP](./fsdp) for the available options.

## Limitations

- `device_map="auto"` loads the model on CPU instead of Neuron. Pass `device_map="neuron"` explicitly.
- Dynamic shapes trigger recompilation. Generation and training both slow down when input shapes change from step to step. Use a static cache for generation and fixed-length padding for training.
