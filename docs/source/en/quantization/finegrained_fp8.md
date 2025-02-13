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

# Fine-grained FP8

With FP8 quantization method, you can quantize your model in FP8 (W8A8):
- the weights will be quantized in 8bit (FP8) per 2D block (e.g. weight_block_size=(128, 128)) which is inspired from the deepseek implementation
- Activations are quantized to 8 bits (FP8) per group per token, with the group value matching that of the weights in the input channels (128 by default)

It's implemented to add support for DeepSeek-V3 and DeepSeek-R1 models, you can see the paper [here](https://arxiv.org/pdf/2412.19437), and the image below explains the quantization scheme : 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/b7b3b34bf826a6423ea82ffc57ecac80c46c3c76/transformers/quantization/quantization_deepseek.png)

> [!TIP]
> You need a GPU with compute capability>=9 (e.g. H100) 

Before you begin, make sure the following libraries are installed with their latest version:

```bash
pip install --upgrade accelerate torch
```
> [!TIP]
> You need to install a torch version compatible with the cuda version of your GPU.


By default, the weights are loaded in full precision (torch.float32) regardless of the actual data type the weights are stored in such as torch.float16. Set `torch_dtype="auto"` to load the weights in the data type defined in a model's `config.json` file to automatically load the most memory-optimal data type.

```py
from transformers import FP8Config, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = FineGrainedFP8Config()
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

A quantized model can be saved via "saved_pretrained" and be reused again via the "from_pretrained".

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```