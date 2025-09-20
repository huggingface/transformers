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

# 비디오 프로세서[[video-processor]]

**Video Processor**는 비디오 모델을 위한 입력 특징을 준비하고, 출력의 후처리를 담당하는 유틸리티입니다. 크기 조정, 정규화, PyTorch로의 변환과 같은 변환을 제공합니다.

비디오 프로세서는 이미지와는 다른 인자 집합으로 비디오를 처리할 수 있도록 이미지 프로세서의 기능을 확장합니다. 원시 비디오 데이터와 모델 사이의 다리 역할을 하여 입력 특징이 VLM에 최적화되도록 보장합니다.

Hugging Face [Hub](https://hf.co) 또는 로컬 디렉터리에 있는 비디오 모델로부터 비디오 프로세서 구성(이미지 크기, 정규화 및 리스케일 여부 등)을 로드하려면 [`~BaseVideoProcessor.from_pretrained`]를 사용하세요. 각 사전 학습된 모델의 구성은 [video_preprocessor_config.json] 파일에 저장되어야 하지만, 이전 모델은 구성이 [preprocessor_config.json](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/preprocessor_config.json) 파일에 저장되어 있을 수 있습니다. 후자는 권장되지 않으며 향후 제거될 예정입니다.

### 사용 예시[[usage-example]]

다음은 [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 모델과 함께 비디오 프로세서를 로드하는 예시입니다:

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

현재, 비디오에 기본 이미지 프로세서를 사용하면, 각 프레임을 개별 이미지로 간주하고 프레임 단위로 변환을 적용하여 비디오 데이터를 처리합니다. 이는 충분히 동작은 하지만 효율이 높지 않습니다. `AutoVideoProcessor`를 사용하면 [torchvision](https://pytorch.org/vision/stable/index.html) 라이브러리를 활용하는 **fast video processors**의 이점을 누릴 수 있습니다. Fast 프로세서는 각 비디오나 프레임을 순회하지 않고 비디오 배치를 한 번에 처리합니다. 이러한 개선은 GPU 가속을 도입해, 특히 높은 처리량이 필요한 작업에서 처리 속도를 크게 향상시킵니다.

Fast video processors는 모든 모델에서 사용할 수 있으며 `AutoVideoProcessor`를 초기화하면 기본으로 로드됩니다. Fast video processors를 사용할 때는 처리 대상 디바이스를 지정하기 위해 `device` 인자를 설정할 수도 있습니다. 입력이 텐서라면 기본적으로 입력과 동일한 디바이스 혹은 CPU에서 처리가 수행됩니다. 추가적인 속도 향상을 위해 디바이스로 'cuda'를 사용할 때 프로세서를 컴파일할 수 있습니다.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```
