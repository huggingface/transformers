<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Intel Gaudi

Intel Gaudi accelerators run PyTorch on the [Intel Gaudi](https://docs.habana.ai/en/latest/index.html) stack, with the device string `hpu`. Gaudi can run PyTorch in lazy or eager mode, and the mode affects which Transformers models run.

## Installation

Install the Intel Gaudi software and its PyTorch package (`habana_frameworks`) with the [Intel Gaudi installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html), plus [Accelerate](https://huggingface.co/docs/accelerate/index) 1.5.0 or later.

Gaudi runs PyTorch in either lazy mode, which accumulates operations into a graph before running them, or eager mode, which runs each operation as it's called. Current Intel Gaudi releases default to eager mode, and lazy mode isn't compatible with all Transformers modeling code. Transformers' Gaudi CI runs in eager mode with int64 support turned on, so set both variables explicitly.

```bash
export PT_HPU_LAZY_MODE=0
export PT_ENABLE_INT64_SUPPORT=1
```

Check that Transformers can see the device.

```py
from transformers import is_torch_hpu_available

print(is_torch_hpu_available())
```

It prints `True` when Gaudi is ready to use, and `False` with Accelerate older than 1.5.0, even if a Gaudi device is present.

## Select devices

Choose which Gaudi devices a Docker container can use when you start it. Set `HABANA_VISIBLE_MODULES` to module IDs, which stay the same across reboots and driver reinstalls, or `HABANA_VISIBLE_DEVICES` to device indices, which can change. If you set neither, PyTorch uses any available Gaudi device on the server.

```bash
docker run --runtime=habana -e HABANA_VISIBLE_MODULES=0,1,2,3 ...
```

Run `hl-smi -Q index,module_id -f csv` to map device indices to module IDs. For a distributed workload on part of a server, Intel supports only 2-device and 4-device groups. See [Multiple Dockers each with a single workload](https://docs.habana.ai/en/latest/Orchestration/Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html) for the supported module ID combinations.

## Run inference

Pass `device_map="hpu"` to [`~PreTrainedModel.from_pretrained`]. [`pipeline`] picks Gaudi automatically when you don't pass `device`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="hpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

inputs = tokenizer("Intel Gaudi is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Quantize models

[bitsandbytes](./quantization/bitsandbytes) quantizes models to 4 or 8 bits on Gaudi 2 and Gaudi 3. When you don't pass `device_map`, it places the model on the current Gaudi device.

To fine-tune a model that doesn't fit on one device, load it in 4-bit with bitsandbytes and train LoRA adapters with [PEFT](./peft).

## Train with Trainer

[`TrainingArguments`] detects Gaudi and sets the device to `hpu`, so `Trainer` trains on it without any device settings. All Gaudi generations support bf16.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
)
print(training_args.device)
```

It prints an `hpu` device when `Trainer` will train on Gaudi.

### Compile the model

With `torch_compile=True`, [`TrainingArguments`] sets `torch_compile_backend` to `"hpu_backend"`. Transformers also wraps `torch.compile` itself, so a call with any backend other than `hpu_backend` or `eager`, including a `@torch.compile` decorator in modeling code, switches to `hpu_backend` and logs a warning.

```py
training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    torch_compile=True,
)
```

### Distributed training

Distributed training uses the [HCCL](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html) collective backend. Transformers selects it on Gaudi, and you can also set it explicitly with `ddp_backend="hccl"`.

```py
training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    ddp_backend="hccl",
)
```

Launch one process per Gaudi device with `torchrun` or `accelerate launch`, after [selecting devices](#select-devices). `accelerate launch` sets the number of processes to the number of Gaudi devices when you don't pass `--num_processes`.

```bash
accelerate launch train.py
```

## Next steps

- Read the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/index.html) for hardware details, the software stack, and the [Gaudi architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html).
- Try [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum-habana/index) for Gaudi-optimized model implementations for training and inference.