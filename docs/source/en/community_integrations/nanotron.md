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

# Nanotron

[Nanotron](https://github.com/huggingface/nanotron) is a distributed training framework with tensor, parallel, and data parallelism (3D parallelism). It is designed for large-scale training workloads across hundreds of GPUs.

Convert any Transformers model to an optimized Nanotron transformer model implementation for pretraining with the [convert_hf_to_nanotron.py](https://github.com/huggingface/nanotron/blob/main/examples/llama/convert_hf_to_nanotron.py) script.

```bash
torchrun --nproc_per_node=1 examples/llama/convert_hf_to_nanotron.py \
    --checkpoint_path=meta-llama/Llama-2-7b-hf \
    --save_path=./llama-7b-nanotron
```

## Transformers integration

1. Load a supported Transformers model, like [`Llama`], with the [`~LlamaForCausalLM.from_pretrained`] function. This reads the `config.json` file from the checkpoint directory and creates a [`LlamaConfig`].
2. Nanotron maps [`LlamaConfig`] to it's own config format and creates a Nanotron model.
3. Convert Transformers weights to Nanotron. A weight mapping guides how to map Nanotron parameter names to Transformers parameter names. This includes handling transformations such as fusing the QKV projections and the gate/up projections.

Nanotron also relies on [`AutoTokenizer`] for turning text into token ids during preprocessing and generation.

## Resources

- [Nanontron](https://github.com/huggingface/nanotron) repository
- [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) describes how to efficiently scale training with Nanotron