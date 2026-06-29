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

# NeMo Automodel

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) is an open-source PyTorch DTensor-native training library from NVIDIA. It supports large and small scale pretraining and fine-tuning for [LLMs](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/overview) and [VLMs](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vision-language-models/overview) for fast experimentation in research and production environments, with parallelism strategies including FSDP2, tensor, pipeline, expert, and context parallelism. For high throughput, it integrates kernels from DeepEP and TransformerEngine. 

```py
# Instantiating Nemotron V3 Nano with Expert Parallelism, FSDP2, and TransformerEngine + DeepEP kernels.
import os

import torch
import torch.distributed as dist

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
torch.manual_seed(1111)

dist_setup = create_distributed_setup_from_config(
    {
        "strategy": "fsdp2",
        "ep_size": 8,
    },
)
model = NeMoAutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    dtype=torch.bfloat16,
    distributed_setup=dist_setup,
)
print(model)
dist.destroy_process_group()
```
Launch the script with `torchrun` using the command below.

```bash
torchrun --nproc-per-node=8 /path/to/script
```

## Transformers integration

- Any LLM or VLM supported in Transformers can also be instantiated through NeMo Automodel. See the [full model coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/overview).
- Built on top of Hugging Face models with [`AutoModel.from_pretrained`], with dynamic high-performance layer swaps and support for more refined parallelisms like Expert Parallelism (EP).
- Detects the architecture field in [`AutoConfig.from_pretrained`] to automatically load custom implementations like Nemotron Nano V3.
- Follows the Transformers API closely for drop-in compatibility.

## Resources

- [NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel)
- [NeMo Transformers API](https://docs.nvidia.com/nemo/automodel/latest/get-started/hf-compatibility)
- NeMo Automodel dense models and Mixture-of-Expert (MoE) [benchmarks](https://docs.nvidia.com/nemo/automodel/latest/performance/performance-summary)
- See the NeMo [fine-tuning](./nemo_automodel_finetuning) guide to learn how to use NeMo for fine-tuning
