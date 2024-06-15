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

# EETQ

The [EETQ](https://github.com/NetEase-FuXi/EETQ) library supports int8 per-channel weight-only quantization for NVIDIA GPUS. The high-performance GEMM and GEMV kernels are from FasterTransformer and TensorRT-LLM. It requires no calibration dataset and does not need to pre-quantize your model. Moreover, the accuracy degradation is negligible owing to the per-channel quantization. 

Make sure you have eetq installed from the [relase page](https://github.com/NetEase-FuXi/EETQ/releases)
```
pip install --no-cache-dir https://github.com/NetEase-FuXi/EETQ/releases/download/v1.0.0/EETQ-1.0.0+cu121+torch2.1.2-cp310-cp310-linux_x86_64.whl
```
or via the source code https://github.com/NetEase-FuXi/EETQ. EETQ requires CUDA capability <= 8.9 and >= 7.0
```
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

An unquantized model can be quantized via "from_pretrained".
```py
from transformers import AutoModelForCausalLM, EetqConfig
path = "/path/to/model"
quantization_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", quantization_config=quantization_config)
```

A quantized model can be saved via "saved_pretrained" and be reused again via the "from_pretrained".

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```