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

# 비디오 프로세서 [[video-processor]]

**비디오 프로세서(Video Processor)**는 비디오 모델의 입력 특징을 준비하고, 모델 출력에 대한 후처리를 수행하는 유틸리티입니다. 크기 조정, 정규화, PyTorch로의 변환과 같은 변환 기능을 제공합니다. 이러한 변환과 더불어 `VideoProcessor` 클래스는 로컬 경로나 URL에서 비디오를 디코딩 ([`torchcodec`](https://pypi.org/project/torchcodec/ 필요)) 하고, 모델별 전략에 따른 프레임 샘플링도 수행합니다.

비디오 프로세서는 이미지 프로세서의 기능을 확장하여 VLM(Vision Large Language Models)이 이미지를 처리할 때와는 다른 인자를 사용하여 비디오를 처리할 수 있도록 합니다.  이는 원본 비디오 데이터를 모델과 연결하는 다리 역할을 하며, 입력 특징이 VLM에 최적화되도록 보장합니다.

새로운 VLM을 추가하거나 기존 모델을 업데이트하여 별도의 비디오 전처리를 활성화할 때, 프로세서 설정을 저장하고 다시 불러오면 비디오 관련 인자가 `video_preprocessing_config.json`이라는 전용 파일에 저장됩니다. 아직 VLM을 업데이트하지 않았더라도 걱정하지 마세요. 프로세서는 `preprocessing_config.json` 파일에서 비디오 관련 설정을 자동으로 불러오려고 시도합니다. 

### 사용 예시 [[usage-example]]
다음은 [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 모델로 비디오 프로세서를 불러오는 예시입니다:

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

현재 기본 이미지 프로세서를 사용하여 비디오를 처리하면, 비디오 데이터를 각 프레임을 개별 이미지로 간주하여 프레임 단위로 변환을 적용합니다. 이 방법은 기능적으로는 작동하지만, 효율적이지 않습니다. `AutoVideoProcessor`를 사용하면 라이브러리를 [torchvision](https://pytorch.org/vision/stable/index.html) 라이브러리를 활용하는 **고속 비디오 프로세서(fast video processors)**를 활용할 수 있습니다. 고속 비디오 프로세서는 각 비디오나 프레임을 개별적으로 반복하지 않고, 전체 비디오 배치(batch)를 한 번에 처리합니다. 이러한 업데이트는 GPU 가속을 지원하며, 특히 높은 처리량이 필요한 작업에서 처리 속도를 크게 향상시킵니다.

고속 비디오 프로세서는 모든 모델에서 사용할 수 있으며, `AutoVideoProcessor`가 초기화될 때 기본적으로 로드됩니다. 고속 비디오 프로세서를 사용할 때, `device` 인자를 설정하여 처리를 수행할 장치를 지정할 수도 있습니다. 기본적으로 입력이 텐서인 경우 입력과 동일한 장치에서 처리되며, 그렇지 않은 경우에는 CPU에서 처리됩니다. 더 큰 속도 향상을 위해, 'cuda' 장치를 사용할 때 프로세서를 컴파일할 수 있습니다.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```

#### 샘플링 동작 [[sampling-behavior]]

비디오 프로세서는 주어진 모델에 가장 적합한 기법을 사용하여 비디오 프레임을 샘플링할 수도 있습니다. 샘플링 동작은 `do_sample_frames` 인자로 제어되며, `num_frames` 또는 `fps` (비디오가 샘플링될 속도)와 같은 모델별 매개변수를 통해 설정할 수 있습니다. 입력 비디오가 로컬 경로나 URL (`str`)로 주어지면 프로세서가 자동으로 디코딩합니다. 샘플링된 프레임 인덱스, 원본 크기, 길이, fps와 같은 디코딩된 비디오에 대한 메타데이터를 얻고 싶다면, 프로세서 호출 시 `return_metadata=True`를 전달하세요.

<Tip warning={false}>

- `num_frames`를 지정한다고 해서 출력이 정확히 해당 개수의 프레임을 포함한다고 보장되지는 않습니다. 모델에 따라 최소 또는 최대 프레임 수 제한이 적용될 수 있습니다.

- 기본 디코더는 [`torchcodec`](https://pypi.org/project/torchcodec/)이며, 반드시 설치해야 합니다.

</Tip>

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processed_video_inputs = processor(videos=["video_path.mp4"], return_metadata=True, do_sample_frames=True, return_tensors="pt")
video_metadata = processed_video_inputs["video_metadata"]

# 원본 비디오의 전체 프레임 수와 원본 FPS 확인
print(video_metadata.total_num_frames, video_metadata.fps)
```

이미 디코딩된 비디오 배열을 전달하면서도 모델별 프레임 샘플링을 활성화하고 싶다면, video_metadata를 제공하는 것을 강력히 권장합니다. 이를 통해 샘플러가 원본 비디오의 길이와 FPS를 알 수 있습니다. 메타데이터는 `VideoMetadata` 객체 또는 일반 딕셔너리 형태로 전달할 수 있습니다. 

```python
from transformers import AutoVideoProcessor
from transformers.video_utils import VideoMetadata

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
my_decodec_video = torch.randint(0, 255, size=(100, 3, 1280, 1280)) # 100 프레임 길이의 짧은 비디오
video_metadata = VideoMetadata(
    total_num_frames=100,
    fps=24,
    duration=4.1, # 초 단위
)
processed_video_inputs = processor(videos=["video_path.mp4"], video_metadata=video_metadata, do_sample_frames=True, num_frames=10, return_tensors="pt")
print(processed_video_inputs.pixel_values_videos.shape)
>>> [10, 3, 384, 384]
```

## 기본 비디오 프로세서 [[transformers.BaseVideoProcessor]]

[[autodoc]] video_processing_utils.BaseVideoProcessor
