<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Video processor

A **Video Processor** is a utility responsible for preparing input features for video models, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch. Along with transformations the `VideoProcessor` class handles video decoding from local paths or URLs (requires [`torchcodec`](https://pypi.org/project/torchcodec/)) and frame sampling according to model-specific strategies.

The video processor extends the functionality of image processors by allowing Vision Large Language Models (VLMs) to handle videos with a distinct set of arguments compared to images. It serves as the bridge between raw video data and the model, ensuring that input features are optimized for the VLM.

Video processor settings are saved in `video_preprocessor_config.json`. Older checkpoints may store them in `preprocessor_config.json`. See [Save and reload](#save-and-reload) for custom backend registration when reloading.

## Usage Example

Here's an example of how to load a video processor with [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) model:

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

Preprocessing runs on the input tensor's device, or on the CPU for other inputs. Set `device` to move videos to another device before preprocessing; this does not change the backend. You can also compile torchvision preprocessing with `torch.compile` when using an accelerator.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=device)
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```

## Video processor backends

Built-in video processors use [torchvision](https://pytorch.org/vision/stable/index.html) for batched resizing, rescaling, and normalization. Install PyTorch and torchvision to use them. Unlike image processors, built-in video processors do not provide a PIL backend.

[`AutoVideoProcessor`] selects the `torchvision` backend by default. You can also request it explicitly and inspect the loaded processor's `backend` attribute.

```python
from transformers import AutoVideoProcessor

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
video_processor = AutoVideoProcessor.from_pretrained(model_id, backend="torchvision")
print(video_processor.backend)
# torchvision
```

The `backend` argument selects the preprocessing implementation, not the video decoder. See [Sampling behavior](#sampling-behavior) for decoding and frame sampling.

### Custom backends

[`~video_processing_utils.BaseVideoProcessor`] provides torchvision-based preprocessing. For a non-torchvision implementation, you can subclass [`~video_processing_utils.VideoProcessorMixin`] and override `preprocess` to return a [`BatchFeature`] containing the inputs expected by the model.

Register custom classes with [`AutoVideoProcessor.register`], and use `video_processor_classes` to map backend names to classes. The example below assumes `NewCustomVideoProcessor` is defined with the preprocessing required by the LLaVA-OneVision checkpoint.

```python
from transformers import LlavaOnevisionConfig

AutoVideoProcessor.register(
    LlavaOnevisionConfig,
    video_processor_classes={"custom": NewCustomVideoProcessor},
    exist_ok=True,
)
custom_processor = AutoVideoProcessor.from_pretrained(model_id, backend="custom")
```

Use `exist_ok=True` because `LlavaOnevisionConfig` is already registered. The supplied entries are merged into its backend mapping where matching names are replaced, and entries not supplied, such as `torchvision`, are preserved.

### Save and reload

[`~video_processing_utils.VideoProcessorMixin.save_pretrained`] saves the preprocessing configuration, not the backend mappings added with [`AutoVideoProcessor.register`]. In the same Python process, reload the saved processor with `backend="custom"`:

```python
custom_processor.save_pretrained("./saved-video-processor")
custom_processor = AutoVideoProcessor.from_pretrained("./saved-video-processor", backend="custom")
```

Custom registrations apply to the current Python process. In a new process, define or import the custom class and register it again before loading.

## Sampling behavior

The video processor can also sample video frames using the technique best suited for the given model. Sampling behavior is controlled with the `do_sample_frames` argument and can be configured through model-specific parameters such as `num_frames` or `fps` (the rate at which the video will be sampled). If the input video is given as a local path or URL (`str`), the processor will decode it automatically. To obtain metadata about the decoded video, such as sampled frame indices, original dimensions, duration, and fps, pass `return_metadata=True` to the processor.

<Tip warning={false}>

- Specifying `num_frames` does not guarantee the output will contain exactly that number of frames. Depending on the model, the sampler may enforce minimum or maximum frame limits.

- The default decoder is [`torchcodec`](https://pypi.org/project/torchcodec/), which must be installed.

</Tip>

```python
import torch
from transformers import AutoVideoProcessor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=device)
processed_video_inputs = processor(videos=["video_path.mp4"], return_metadata=True, do_sample_frames=True, return_tensors="pt")
video_metadata = processed_video_inputs["video_metadata"]

# See how many frames the original video had and what was the original FPS
print(video_metadata.total_num_frames, video_metadata.fps)
```

If you pass an already decoded video array but still want to enable model-specific frame sampling, it is strongly recommended to provide video_metadata. This allows the sampler to know the original video’s duration and FPS. You can pass metadata as a `VideoMetadata` object or as a plain dict.

```python
import torch
from transformers import AutoVideoProcessor
from transformers.video_utils import VideoMetadata

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=device)
decoded_video = torch.randint(0, 255, size=(100, 3, 1280, 1280)) # short video of 100 frames
video_metadata = VideoMetadata(
    total_num_frames=100,
    fps=24,
    duration=4.1, # in seconds
)
processed_video_inputs = processor(videos=[decoded_video], video_metadata=video_metadata, do_sample_frames=True, num_frames=10, return_tensors="pt")
print(processed_video_inputs.pixel_values_videos.shape)
[10, 3, 384, 384]
```

## VideoProcessorMixin

[[autodoc]] video_processing_utils.VideoProcessorMixin


## BaseVideoProcessor

[[autodoc]] video_processing_utils.BaseVideoProcessor
