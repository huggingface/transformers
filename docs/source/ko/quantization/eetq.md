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

# EETQ [[eetq]]

[EETQ](https://github.com/NetEase-FuXi/EETQ) 라이브러리는 NVIDIA GPU에 대해 int8 채널별(per-channel) 가중치 전용 양자화(weight-only quantization)을 지원합니다. 고성능 GEMM 및 GEMV 커널은 FasterTransformer 및 TensorRT-LLM에서 가져왔습니다. 교정(calibration) 데이터셋이 필요 없으며, 모델을 사전에 양자화할 필요도 없습니다. 또한, 채널별 양자화(per-channel quantization) 덕분에 정확도 저하가 미미합니다.

[릴리스 페이지](https://github.com/NetEase-FuXi/EETQ/releases)에서 eetq를 설치했는지 확인하세요.
```
pip install --no-cache-dir https://github.com/NetEase-FuXi/EETQ/releases/download/v1.0.0/EETQ-1.0.0+cu121+torch2.1.2-cp310-cp310-linux_x86_64.whl
```
또는 소스 코드 https://github.com/NetEase-FuXi/EETQ 에서 설치할 수 있습니다. EETQ는 CUDA 기능이 8.9 이하이고 7.0 이상이어야 합니다.
```
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

비양자화 모델은 "from_pretrained"를 통해 양자화할 수 있습니다.
```py
from transformers import AutoModelForCausalLM, EetqConfig
path = "/path/to/model".
quantization_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", quantization_config=quantization_config)
```

양자화된 모델은 "save_pretrained"를 통해 저장할 수 있으며, "from_pretrained"를 통해 다시 사용할 수 있습니다.

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```