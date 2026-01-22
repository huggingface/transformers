<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Exporting to production

Export Transformers' models to different formats for optimized runtimes and devices. Deploy the same model to cloud providers or run it on mobile and edge devices. You don't need to rewrite the model from scratch for each deployment environment. Freely deploy across any inference ecosystem.

## ExecuTorch

[ExecuTorch](https://pytorch.org/executorch/stable/index.html) runs PyTorch models on mobile and edge devices. It exports a model into a graph of standardized operators, compiles the graph into an ExecuTorch program, and executes it on the target device. The runtime is lightweight and calculates the execution plan ahead of time.

Install [Optimum ExecuTorch](https://huggingface.co/docs/optimum-executorch/en/index) from source.

```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install '.[dev]'
```

Export a Transformers model to ExecuTorch with the CLI tool.

```bash
optimum-cli export executorch \
    --model "Qwen/Qwen3-8B" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="hf_smollm2"
```

Run the following command to view all export options.

```bash
optimum-cli export executorch --help
```

## ONNX

[ONNX](http://onnx.ai) is a shared language for describing models from different frameworks. It represents models as a graph of standardized operators with well-defined types, shapes, and metadata. Models serialize into compact protobuf files that you can deploy across optimized runtimes and engines.

[Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index) exports models to ONNX with configuration objects. It supports many [architectures](https://huggingface.co/docs/optimum-onnx/onnx/overview) and is easily extendable. Export models through the CLI tool or programmatically.

Install [Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index).

```bash
uv pip install optimum-onnx
```

### optimum-cli

Specify a model to export and the output directory with the `--model` argument.

```bash
optimum-cli export onnx --model Qwen/Qwen3-8B Qwen/Qwen3-8b-onnx/
```

Run the following command to view all available arguments or refer to the [Export a model to ONNX with optimum.exporters.onnx](https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model) guide for more details.

```bash
optimum cli export onnx --help
```

To export a local model, save the weights and tokenizer files in the same directory. Pass the directory path to the `--model` argument and use the `--task` argument to specify the [task](https://huggingface.co/docs/optimum/exporters/task_manager#transformers). If you don't provide `--task`, the system auto-infers it from the model or uses an architecture without a task-specific head.

```bash
optimum-cli export onnx --model path/to/local/model --task text-generation Qwen/Qwen3-8b-onnx/
```

Deploy the model with any [runtime](https://onnx.ai/supported-tools.html#deployModel) that supports ONNX, including ONNX Runtime.

```py
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b-onnx")
model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen3-8b-onnx")
inputs = tokenizer("Plants generate energy through a process known as ", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs))
```

### optimum.onnxruntime

Export Transformers' models programmatically with Optimum ONNX. Instantiate a [`~optimum.onnxruntime.ORTModel`] with a model and set `export=True`. Save the ONNX model with [`~optimum.onnxruntime.ORTModel.save_pretrained`].

```py
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

ort_model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen3-8b", export=True)
tokenizer = AutoTokenizer.from_pretrained("onnx/")

ort_model.save_pretrained("onnx/")
tokenizer.save_pretrained("onnx/")
```