<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is optimizes LLM inference on NVIDIA GPUs. It compiles models into a TensorRT engine with in-flight batching, paged KV caching, and tensor parallelism. [AutoDeploy](https://nvidia.github.io/TensorRT-LLM/torch/auto_deploy/auto-deploy.html) accepts Transformers models without requiring any changes. It automatically converts the model to an optimized runtime.

Pass a model id from the Hub to [build_and_run_ad.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/auto_deploy/build_and_run_ad.py) to run a Transformers model.

```bash
cd examples/auto_deploy
python build_and_run_ad.py --model meta-llama/Llama-3.2-1B
```

Under the hood, AutoDeploy creates an [LLM](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.LLM) class. It loads the model configuration with [`AutoConfig.from_pretrained`] and extracts any parallelism metadata stored in `tp_plan`. [`AutoModelForCausalLM.from_pretrained`] loads the model with the config and enables Transformers' built-in tensor parallelism.

```py
from tensorrt_llm._torch.auto_deploy import LLM

llm = LLM(model="meta-llama/Llama-3.2-1B")
```

TensorRT-LLM extracts the model graph with `torch.export` and applies optimizations. It replaces Transformers attention with TensorRT-LLM [attention kernels](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/attention_backend) and compiles the model into an optimized execution backend.

## Resources

- [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/) for more detailed usage guides.
- [AutoDeploy guide](https://nvidia.github.io/TensorRT-LLM/torch/auto_deploy/auto-deploy.html) explains how it works with advanced examples.