<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ExecuTorch

[ExecuTorch](https://pytorch.org/executorch/stable/index.html) runs PyTorch models on mobile and edge devices. Export your Transformers models to the ExecuTorch format with [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch) with the command below.

```
optimum-cli export executorch \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="hf_smollm2"
```

Run `optimum-cli export executorch --help` to see all export options. For detailed export instructions, check the [README](optimum/exporters/executorch/README.md).
