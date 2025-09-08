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

# Video Processor

A **Video Processor** is a utility responsible for preparing input features for video models, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch. Along ith transformations the `VideoProcessor` class handles video decoding from local paths or URLs (requires [`torchcodec`](https://pypi.org/project/torchcodec/)) and frame sampling according to model-specific strategies.

The video processor extends the functionality of image processors by allowing Vision Large Language Models (VLMs) to handle videos with a distinct set of arguments compared to images. It serves as the bridge between raw video data and the model, ensuring that input features are optimized for the VLM.

When adding a new VLM or updating an existing one to enable distinct video preprocessing, saving and reloading the processor configuration will store the video related arguments in a dedicated file named `video_preprocessing_config.json`. Don't worry if you haven't updated your VLM, the processor will try to load video related configurations from a file named `preprocessing_config.json`.


### Usage Example
Here's an example of how to load a video processor with [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) model:

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

Currently, if using base image processor for videos, it processes video data by treating each frame as an individual image and applying transformations frame-by-frame. While functional, this approach is not highly efficient. Using `AutoVideoProcessor` allows us to take advantage of **fast video processors**, leveraging the [torchvision](https://pytorch.org/vision/stable/index.html) library. Fast processors handle the whole batch of videos at once, without iterating over each video or frame. These updates introduce GPU acceleration and significantly enhance processing speed, especially for tasks requiring high throughput.

Fast video processors are available for all models and are loaded by default when an `AutoVideoProcessor` is initialized. When using a fast video processor, you can also set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise. For even more speed improvement, we can compile the processor when using 'cuda' as device.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```

#### Sampling behavior

The video processor can also sample video frames using the technique best suited for the given model. Sampling behavior is controlled with the `do_sample_frames` argument and can be configured through model-specific parameters such as `num_frames` or `fps` (the rate at which the video will be sampled). If the input video is given as a local path or URL (`str`), the processor will decode it automatically. To obtain metadata about the decoded video, such as sampled frame indices, original dimensions, duration, and fps, pass `return_metadata=True` to the processor.

<Tip warning={false}>

- Specifying `num_frames` does not guarantee the output will contain exactly that number of frames. Depending on the model, the sampler may enforce minimum or maximum frame limits.

- The default decoder is [`torchcodec`](https://pypi.org/project/torchcodec/), which must be installed.

</Tip>


```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processed_video_inputs = processor(videos=["video_path.mp4"], return_metadata=True, do_sample_frames=True, return_tensors="pt")
video_metadata = processed_video_inputs["video_metadata"]

# See how many frames the original video had and what was the original FPS
print(video_metadata.total_num_frames, video_metadata.fps)
```

If you pass an already decoded video array but still want to enable model-specific frame sampling, it is strongly recommended to provide video_metadata. This allows the sampler to know the original video’s duration and FPS. You can pass metadata as a `VideoMetadata` object or as a plain dict.

```python
from transformers import AutoVideoProcessor
from transformers.video_utils import VideoMetadata

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
my_decodec_video = torch.randint(0, 255, size=(100, 3, 1280, 1280)) # short video of 100 frames
video_metadata = VideoMetadata(
    total_num_frames=100,
    fps=24,
    duration=4.1, # in seconds
)
processed_video_inputs = processor(videos=["video_path.mp4"], video_metadata=video_metadata, do_sample_frames=True, num_frames=10, return_tensors="pt")
print(processed_video_inputs.pixel_values_videos.shape)
>>> [10, 3, 384, 384]
```

## BaseVideoProcessor

[[autodoc]] video_processing_utils.BaseVideoProcessor

