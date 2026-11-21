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

# Keep modules in full precision

Some modules lose more accuracy than others when quantized. Keeping them in their original precision costs a little memory and can recover much of the accuracy. This applies when you quantize a model while loading. A pre-quantized checkpoint already records which modules it skipped.

## Default skipped modules

If you don't pass a skip list, most methods keep these modules unquantized.

- The output embeddings, usually `lm_head`.
- Weights tied to other weights, like input embeddings that share weights with `lm_head`.
- The model's last module.
- Modules the model lists in `_keep_in_fp32_modules`.

A few methods set their own default instead. [HQQ](./hqq), [FP-Quant](./fp_quant), and [Four Over Six](./fouroversix) skip only `lm_head`. [SINQ](./sinq) skips only the modules in `_keep_in_fp32_modules`.

## Pass a skip list

Pass the module names to the skip-list argument of the method's config. The example below keeps the `lm_head` and the MLP down projections in bf16 with [bitsandbytes](./bitsandbytes).

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_skip_modules=["lm_head", "mlp.down_proj"],
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=quantization_config,
    device_map="auto",
)
```

> [!WARNING]
> A skip list replaces the default skipped modules instead of adding to them, although modules in `_keep_in_fp32_modules` stay skipped. Include `lm_head` in your list if you still want it unquantized. [AWQ](./awq) is the exception and adds your list to the defaults.

The argument's name depends on the method.

| Argument | Methods |
|---|---|
| `modules_to_not_convert` | Most methods, including AWQ, EETQ, FBGEMM FP8, Fine-grained FP8, Four Over Six, FP-Quant, HIGGS, Metal, MXFP4, NVFP4, Quanto, SINQ, and torchao |
| `llm_int8_skip_modules` | bitsandbytes, in both 4-bit and 8-bit mode |
| `skip_modules` | HQQ |

[GPTQ](./gptq) takes the opposite approach. `modules_in_block_to_quantize` lists the modules to quantize, and everything else in the block is left unquantized.

## Match module names

Most methods match each entry in the skip list against the full module names from `model.named_modules()`, such as `model.layers.3.mlp.down_proj`. [HQQ](./hqq) is the exception and matches layer names exactly. An entry matches in one of three ways.

- As a suffix. `mlp.down_proj` matches the down projection in every layer.
- As a prefix that ends at a module boundary. `model.layers.1` matches `model.layers.1.mlp` but not `model.layers.11.mlp`.
- As a regular expression with the same boundary. `model\.layers\.(0|1)` matches the first two layers.

Print the module names to find the right entries.

```py
for name, _ in model.named_modules():
    print(name)
```

## Choose which modules to skip

Skip the modules where rounding error affects accuracy the most for the least memory saved. Start with the defaults, measure accuracy on your task, and add modules only if it drops.

- `lm_head` produces the logits for every token in the vocabulary, so errors there directly change which token is picked. It's skipped by default. On models with large vocabularies it's also one of the biggest single layers, so quantizing it saves real memory if accuracy holds up.
- MoE router or gate layers, like `mlp.gate` in Qwen3-MoE or `mlp.router` in gpt-oss, decide which experts each token goes to. They're tiny, and a rounding error can send a token to the wrong expert, so they're almost always worth skipping.
- Vision encoders and multimodal projectors in vision-language models are small next to the language model, and errors there distort every image token. Skip them if image understanding degrades.
- The first and last few decoder layers often have more activation outliers than the middle layers. Skip them when the quantized model's accuracy is noticeably worse than in bf16.

Every module you skip stays in its original precision and costs memory. Check that the model still fits after you add to the skip list.
