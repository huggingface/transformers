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

**비디오 프로세서(Video Processor)**는 비디오 모델의 입력으로 사용될 특성 값들을 준비하고, 모델 출력의 후처리를 지원하는 유틸리티입니다. 크기 조정, 정규화, PyTorch 텐서 변환 등의 기능을 제공합니다.

비디오 프로세서는 이미지 프로세서의 기능을 확장하여, 모델이 영상을 처리할 때 이미지와는 구별되는 별도의 인수를 사용할 수 있도록 합니다. 또한 원본 비디오 데이터와 모델 사이를 연결하는 역할을 하며, 입력 특성 값들이 VLM에 최적화되도록 보장합니다.

[`~BaseVideoProcessor.from_pretrained`]를 사용하면 Hugging Face [Hub](https://hf.co) 또는 로컬 디렉터리에 있는 비디오 모델로부터 비디오 프로세서 설정(이미지 크기, 정규화 및 리스케일 여부 등)을 불러올 수 있습니다. 각 사전 학습된 모델의 설정은 [video_preprocessor_config.json] 파일에 저장되어야 하지만, 일부 오래된 모델의 경우 [preprocessor_config.json](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/preprocessor_config.json) 파일에 저장되어 있을 수도 있습니다. 후자의 경우는 권장되지 않으며 향후 제거될 예정입니다.

### 사용 예시[[usage-example]]

다음은 [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 모델과 함께 비디오 프로세서를 로드하는 예시입니다.

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

현재 비디오에 기본 이미지 프로세서를 사용할 경우, 비디오 데이터의 각 프레임이 개별 이미지로 처리되어 변환이 적용됩니다. 이 방식도 잘 동작하긴 하지만 효율이 좋지 않습니다. 이 대신 `AutoVideoProcessor`를 사용하면 [torchvision](https://pytorch.org/vision/stable/index.html) 라이브러리를 활용한 **fast video processors**의 이점을 누릴 수 있습니다. Fast processors는 각 비디오나 프레임을 하나씩 처리하지 않고, 전체 비디오 배치를 한 번에 처리합니다. 이는 GPU 가속을 도입하며, 특히 높은 처리량이 필요한 작업에서 처리 속도를 크게 향상시킵니다.

Fast video processors는 모든 모델에서 사용할 수 있으며 `AutoVideoProcessor`가 초기화될 때 같이 로드됩니다. Fast video processors를 사용할 때는 `device` 인수를 설정하여 어떤 장치에서 처리를 수행할지 지정할 수 있습니다. 텐서를 입력으로 하는 경우, 기본적으로 입력과 동일한 장치에서 처리되며, 그렇지 않은 경우 CPU에서 처리됩니다. 장치로 'cuda'를 사용하는 경우, 추가적인 속도 향상을 위해 프로세서를 컴파일하여 사용할 수 있습니다.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```
