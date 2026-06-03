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

[ExecuTorch](https://pytorch.org/executorch/stable/index.html) is a platform that enables PyTorch training and inference programs to be run on mobile and edge devices. It is powered by [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html) and [torch.export](https://pytorch.org/docs/main/export.html) for performance and deployment.

You can use ExecuTorch with Transformers with [torch.export](https://pytorch.org/docs/main/export.html). The [`~transformers.convert_and_export_with_cache`] method converts a [`PreTrainedModel`] into an exportable module. Under the hood, it uses [torch.export](https://pytorch.org/docs/main/export.html) to export the model, ensuring compatibility with ExecuTorch.

```py
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
from transformers.integrations.executorch import(
    TorchExportableModuleWithStaticCache,
    convert_and_export_with_cache
)

generation_config = GenerationConfig(
    use_cache=True,
    cache_implementation="static",
    cache_config={
        "batch_size": 1,
        "max_cache_len": 20,
    }
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", pad_token="</s>", padding_side="right")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa", generation_config=generation_config)

exported_program = convert_and_export_with_cache(model)
```

The exported PyTorch model is now ready to be used with ExecuTorch. Wrap the model with [`~transformers.TorchExportableModuleWithStaticCache`] to generate text.

```py
prompts = ["Simply put, the theory of relativity states that "]
prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
prompt_token_ids = prompt_tokens["input_ids"]

generated_ids = TorchExportableModuleWithStaticCache.generate(
    exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=20,
)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text)
['Simply put, the theory of relativity states that 1) the speed of light is the']
```
