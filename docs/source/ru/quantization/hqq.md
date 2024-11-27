<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# HQQ 

Half-Quadratic Quantization (HQQ) implements on-the-fly quantization via fast robust optimization. It doesn't require calibration data and can be used to quantize any model.  
Please refer to the <a href="https://github.com/mobiusml/hqq/">official package</a> for more details.

For installation, we recommend you use the following approach to get the latest version and build its corresponding CUDA kernels:
```
pip install hqq
```

To quantize a model, you need to create an [`HqqConfig`]. There are two ways of doing it:
``` Python
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

# Method 1: all linear layers will use the same quantization config
quant_config  = HqqConfig(nbits=8, group_size=64)
```

``` Python
# Method 2: each linear layer with the same tag will use a dedicated quantization config
q4_config = {'nbits':4, 'group_size':64}
q3_config = {'nbits':3, 'group_size':32}
quant_config  = HqqConfig(dynamic_config={
  'self_attn.q_proj':q4_config,
  'self_attn.k_proj':q4_config,
  'self_attn.v_proj':q4_config,
  'self_attn.o_proj':q4_config,

  'mlp.gate_proj':q3_config,
  'mlp.up_proj'  :q3_config,
  'mlp.down_proj':q3_config,
})
```

The second approach is especially interesting for quantizing Mixture-of-Experts (MoEs) because the experts are less affected by lower quantization settings.


Then you simply quantize the model as follows
``` Python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```

## Optimized Runtime

HQQ supports various backends, including pure PyTorch and custom dequantization CUDA kernels. These backends are suitable for older gpus and peft/QLoRA training.
For faster inference, HQQ supports 4-bit fused kernels (TorchAO and Marlin), reaching up to 200 tokens/sec on a single 4090.
For more details on how to use the backends, please refer to https://github.com/mobiusml/hqq/?tab=readme-ov-file#backend
