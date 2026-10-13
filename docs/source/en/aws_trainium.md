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

[AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) and [AWS Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/) chips run PyTorch through the `torch-neuronx` package, which registers the `neuron` device. Neuron compiles your model into a graph for each input shape it sees. Keeping shapes static avoids recompilation, and that shapes most of the advice on this page.

## Installation

Install `torch-neuronx` by following the [AWS Neuron setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html). Then check that PyTorch sees the Neuron devices.

```py
import torch
import torch_neuronx

print(torch.neuron.is_available(), torch.neuron.device_count())
```

## Inference

### Run inference

`device_map="auto"` doesn't detect Neuron. Pass `device_map="neuron"` to [`~PreTrainedModel.from_pretrained`] explicitly, or `device="neuron"` to [`Pipeline`].

Pass the generation inputs on CPU instead of calling `.to(model.device)`. [`~GenerationMixin.generate`] moves only the tensors that `forward` consumes to the Neuron device. The tensors that grow each step (output sequence, `attention_mask`) stay on CPU, so the compiled graph stays stable. The output is returned on the same device as the inputs. See [Keep generation tensors on CPU](./kv_cache#keep-generation-tensors-on-cpu) for more details.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="neuron")

# leave the inputs on CPU
inputs = tokenizer("The French Bread Law states", return_tensors="pt")
output = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

<!-- TODO: add the real output from a Trainium instance. -->

With a static cache, [`~GenerationMixin.generate`] compiles the forward pass with `torch.compile` automatically on Neuron. Pass `disable_compile=True` to [`~GenerationMixin.generate`] to opt out.

<!-- Optional: ### Serve a model — fill in if `transformers serve` runs on Neuron. -->

<!-- Optional: ### Choose an attention backend — fill in if Neuron has attention backends other than SDPA (for example, Hub kernels). -->

<!-- Optional: ### Quantize models — fill in with supported methods. -->

## Training

### Train with Trainer

[`TrainingArguments`] detects Neuron and sets the device to `neuron` automatically. All Trainium generations support bf16.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
```

#### Compile the training step

Set `torch_compile=True` to compile the model. [`TrainingArguments`] sets `torch_compile_backend` to `"neuron"` automatically when it detects Neuron.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    torch_compile=True,
)
```

Each new input shape triggers a new compilation. Pad inputs to a fixed length (`padding="max_length"` in the tokenizer) and set `dataloader_drop_last=True` so the last batch doesn't have a different size.

<!-- TODO: link to Neuron docs on the compilation cache, if relevant. -->

### Distributed training

Launch multi-device training with `torchrun`. Accelerate selects the `neuron` distributed backend automatically, so you don't need to set `ddp_backend`.

```bash
torchrun --nproc_per_node=32 train.py
```

<!-- TODO: confirm the number of NeuronCores per instance type (for example, trn1.32xlarge, trn2.48xlarge) and whether FSDP / tensor parallelism are supported. -->

## Limitations

- `device_map="auto"` doesn't place models on Neuron. Pass `device_map="neuron"` explicitly.
- Dynamic shapes trigger recompilation. Generation and training both slow down when input shapes change from step to step. Use a static cache for generation and fixed-length padding for training.
- On the legacy `torch-xla` based Neuron stack, launching with `torchrun` while `torch_xla` and `torch_neuronx` are both installed logs a warning that tells you to use the `NeuronTrainer` from [Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index) instead of [`Trainer`].

## Next steps

- Read the [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) for details about the hardware and the compiler.
- Try [Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index) for Neuron-optimized models, distributed training, and deployment.
- Read [Keep generation tensors on CPU](./kv_cache#keep-generation-tensors-on-cpu) and [torch.compile](./torch_compile) for more about compiler backends.
