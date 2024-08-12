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

# TorchAO

[TorchAO](https://github.com/pytorch/ao) is an architecture optimization library for PyTorch, it provides high performance dtypes, optimization techniques and kernels for inference and training, featuring composability with native PyTorch features like `torch.compile`, FSDP etc.. Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#without-intrusive-code-changes)

Before you begin, make sure the following libraries are installed with their latest version:

```bash
pip install --upgrade torch torchao
```


```py
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
# We support int4_weight_only, int8_weight_only and int8_dynamic_activation_int8_weight
# More examples and documentations for arguments can be found in https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques
quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# compile the quantizd model to get speedup
import torchao
torchao.quantization.utils.recommended_inductor_config_setter()
quantized_model = torch.compile(quantized_model, mode="max-autotune")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

torchao quantization is implemented with tensor subclasses, currently it does not work with huggingface serialization, both the safetensor option and [non-safetensor option](https://github.com/huggingface/transformers/issues/32364), we'll update here with instructions when it's working.
