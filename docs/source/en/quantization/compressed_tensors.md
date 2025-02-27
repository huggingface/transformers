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
# Compressed Tensors

The [`compressed-tensors`](https://github.com/neuralmagic/compressed-tensors) library provides a versatile and efficient way to store and manage compressed model checkpoints. This library supports various quantization and sparsity schemes, making it a unified format for handling different model optimizations like GPTQ, AWQ, SmoothQuant, INT8, FP8, SparseGPT, and more.

Some of the supported formats include:
1. `dense`
2. `int-quantized` ([sample](https://huggingface.co/nm-testing/tinyllama-w8a8-compressed-hf-quantizer)): INT8 quantized models
3. `float-quantized` ([sample](https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-fp8-hf_compat)): FP8 quantized models; currently support E4M3
4. `pack-quantized` ([sample](https://huggingface.co/nm-testing/tinyllama-w4a16-compressed-hf-quantizer)): INT4 or INT8 weight-quantized models, packed into INT32. For INT4, the weights have an INT4 range but are stored as INT8 and then packed into INT32.

Compressed models can be easily created using [llm-compressor](https://github.com/vllm-project/llm-compressor).
Alternatively models can be created independently and serialized with a compressed tensors config.

To find existing models on the Hugging Face Model Hub, search for the [`compressed-tensors` tag](https://huggingface.co/models?other=compressed-tensors).

#### Features:
 - Weight and activation precisions: FP8, INT4, INT8 (for Q/DQ arbitrary precision is allowed for INT)
 - Quantization scales and zero-points strategies: [tensor, channel, group, block, token](https://github.com/neuralmagic/compressed-tensors/blob/83b2e7a969d70606421a76b9a3d112646077c8de/src/compressed_tensors/quantization/quant_args.py#L43-L52)
 - Dynamic per-token activation quantization (or any static strategy)
 - Sparsity in weights (unstructured or semi-structured like 2:4) can be composed with quantization for extreme compression
 - Supports quantization of arbitrary modules, not just Linear modules
 - Targeted support or ignoring of modules by name or class

## Installation

It is recommended to install stable releases of compressed-tensors from [PyPI](https://pypi.org/project/compressed-tensors):
```bash
pip install compressed-tensors
```

Developers who want to experiment with the latest features can also install the package from source:
```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

## Quickstart Model Load
Quantized models can be easily loaded for inference as shown below. Only models that have already been quantized can be loaded at the moment. To quantize a model into the compressed-tensors format see [llm-compressor](https://github.com/vllm-project/llm-compressor).

```python
from transformers import AutoModelForCausalLM

# Load the model in compressed-tensors format
ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf")

# Measure memory usage
mem_params = sum([param.nelement()*param.element_size() for param in ct_model.parameters()])
print(f"{mem_params/2**30:.4f} GB")
# 8.4575 GB
```

We can see just above that the compressed-tensors FP8 checkpoint of Llama 3.1 8B is able to be loaded for inference using half of the memory of the unquantized reference checkpoint.

## Sample Use Cases - Load and run an FP8 model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

model_name = "nm-testing/Meta-Llama-3-8B-Instruct-fp8-hf_compat"

quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(prompt, return_tensors="pt")
generated_ids = quantized_model.generate(**inputs, max_length=50, do_sample=False)
outputs = tokenizer.batch_decode(generated_ids)

print(outputs)

"""
['<|begin_of_text|>Hello, my name is [Name]. I am a [Your Profession/Student] and I am here to learn about the [Course/Program] at [University/Institution]. I am excited to be here and I am looking forward to', '<|begin_of_text|>The capital of France is Paris, which is located in the north-central part of the country. Paris is the most populous city in France and is known for its stunning architecture, art museums, fashion, and romantic atmosphere. The city is home to', "<|begin_of_text|>The future of AI is here, and it's already changing the way we live and work. From virtual assistants to self-driving cars, AI is transforming industries and revolutionizing the way we interact with technology. But what does the future of AI hold"]
"""

```

The above shows a quick example for running generation using a `compressed-tensors`
model. Currently, once loaded the model cannot be saved.

## Deep dive into a compressed-tensors model checkpoint

In this example we will examine how the compressed-tensors model nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf is defined through its configuration entry and see how this translates to the loaded model representation. 

First, let us look at the [`quantization_config` of the model](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf/blob/main/config.json). At a glance it looks overwhelming with the number of entries but this is because compressed-tensors is a format that allows for flexible expression both during and after model compression.

In practice for checkpoint loading and inference the configuration can be simplified to not include all the default or empty entries, so we will do that here to focus on what compression is actually represented.

```yaml
"quantization_config": {
  "config_groups": {
    "group_0": {
      "input_activations": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      },
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      }
    }
  },
  "format": "naive-quantized",
  "ignore": ["lm_head"],
  "quant_method": "compressed-tensors",
  "quantization_status": "frozen"
},
```

We can see from the above configuration that it is specifying one config group that includes weight and activation quantization to FP8 with a static per-tensor strategy. It is also worth noting that in the `ignore` list there is an entry to skip quantization of the `lm_head` module, so that module should be untouched in the checkpoint.

To see the result of the configuration in practice, we can simply use the [safetensors viewer](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf?show_file_info=model.safetensors.index.json) on the model card to see the quantized weights, input_scale, and weight_scale for all of the Linear modules in the first model layer (and so on for the rest of the layers).

| Tensors | Shape |	Precision |
| ------- | ----- | --------- |
model.layers.0.input_layernorm.weight	| [4 096]	| BF16 
model.layers.0.mlp.down_proj.input_scale	| [1]	| BF16 
model.layers.0.mlp.down_proj.weight	| [4 096, 14 336] |	F8_E4M3 
model.layers.0.mlp.down_proj.weight_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.input_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.weight	| [14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.gate_proj.weight_scale	| [1] |	BF16 
model.layers.0.mlp.up_proj.input_scale|	[1]	|BF16 
model.layers.0.mlp.up_proj.weight |	[14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.up_proj.weight_scale | [1]	| BF16 
model.layers.0.post_attention_layernorm.weight |	[4 096]	|BF16 
model.layers.0.self_attn.k_proj.input_scale |	[1]	|  BF16
model.layers.0.self_attn.k_proj.weight |	[1 024, 4 096]|	F8_E4M3
model.layers.0.self_attn.k_proj.weight_scale |[1]	| BF16 
model.layers.0.self_attn.o_proj.input_scale	| [1]	| BF16
model.layers.0.self_attn.o_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.o_proj.weight_scale | [1]	| BF16 
model.layers.0.self_attn.q_proj.input_scale	| [1]	| BF16 
model.layers.0.self_attn.q_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.q_proj.weight_scale |	[1] | BF16 
model.layers.0.self_attn.v_proj.input_scale	| [1] | BF16 
model.layers.0.self_attn.v_proj.weight |	[1 024, 4 096]	| F8_E4M3 
model.layers.0.self_attn.v_proj.weight_scale |	[1] |	BF16 

When we load the model with the compressed-tensors HFQuantizer integration, we can see that all of the Linear modules that are specified within the quantization configuration have been replaced by `CompressedLinear` modules that manage the compressed weights and forward pass for inference. Note that the `lm_head` mentioned before in the ignore list is still kept as an unquantized Linear module.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf")
print(ct_model)
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (k_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (v_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (o_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (up_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (down_proj): CompressedLinear(
            in_features=14336, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""
```
