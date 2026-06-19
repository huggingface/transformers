<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Add vision processing components

Adding a vision model requires image or video processing components on top of the standard [modular](./modular_transformers) approach. Image-only models need image processors and video models need a video processor, both of which are accessible behind the [`AutoImageProcessor`] and [`AutoVideoProcessor`] entry points.

> [!NOTE]
> For the modeling and config steps, follow the [modular](./modular_transformers) guide first.

## Image processors

Create image processors when the model consumes images. The [torchvision](https://docs.pytorch.org/vision/stable/index.html) backend is the default and supports GPU acceleration. [PIL](https://pillow.readthedocs.io/en/stable/index.html) is the fallback when torchvision isn't available.

Both image processor classes share the same preprocessing logic but have different backends. Their constructor signatures and default values must be identical. [`AutoImageProcessor.from_pretrained`] selects the backend at load time and falls back to PIL when torchvision isn't available. Mismatched signatures cause the same saved config to behave differently across environments.

### torchvision

Create `image_processing_<model_name>.py` with a class that inherits from [`TorchvisionBackend`]. If your processor needs custom parameters beyond the standard [ImagesKwargs], define a kwargs class.

```py
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring

class MyModelImageProcessorKwargs(ImagesKwargs, total=False):
    tile_size: int  # any model-specific kwargs

@auto_docstring
class MyModelImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[MyModelImageProcessorKwargs]):
        super().__init__(**kwargs)
```

> [!TIP]
> See [`LlavaOnevisionImageProcessor`] for reference.

### PIL

Create `image_processing_pil_<model_name>.py` with a class that inherits from [`PilBackend`]. Duplicate the kwargs class here instead of importing it from the torchvision file because it can fail when torchvision isn't installed. Add an `# Adapted from` comment so the two stay in sync. For processors with no custom parameters, use [`ImagesKwargs`] directly.

```py
from ...image_processing_backends import PilBackend
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring

# Adapted from transformers.models.my_model.image_processing_my_model.MyModelImageProcessorKwargs
class MyModelImageProcessorKwargs(ImagesKwargs, total=False):
    tile_size: int  # any model-specific kwargs

@auto_docstring
class MyModelImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[MyModelImageProcessorKwargs]):
        super().__init__(**kwargs)
```

> [!TIP]
> See [`LlavaOnevisionImageProcessorPil`] for reference.

### Post-processing

Add post-processing methods directly to the processor class. Post-processing methods are called with the model outputs (`outputs`) and any additional arguments required for the specific post-processing method.

```py
class MyModelImageProcessor(TorchvisionBackend):
    ...

    def post_process_my_task(self, outputs, ...):
        ...
```

Post-processors return either a list of simple objects (`list[str]` or `list[torch.Tensor]`) or a list of complex objects (`list[MyTaskPostProcessorOutput]` or `list[dict]`). Post-processor outputs are defined in `src/transformers/image_processing_outputs.py` and inherit from [`BatchFeature`].

```py
class MyTaskPostProcessorOutput(BatchFeature):
    predictions: torch.Tensor
    scores: torch.Tensor
```


## Video processor

Add a video processor when the model consumes videos or sampled video frames.

Create `video_processing_<model_name>.py` in the model directory. [`BaseVideoProcessor`] inherits from the [`TorchvisionBackend`] and provides shared decoding, frame sampling, resizing, rescaling, normalization, saving, and loading behavior.

The class attributes are the default preprocessing values. Users can override them at initialization or call time. Use the same names as [`VideosKwargs`] when possible, such as `size`, `crop_size`, `do_resize`, `do_sample_frames`, `num_frames`, and `fps`.

Define a kwargs class if your video processor needs custom parameters beyond the standard [`VideosKwargs`]. Set it as `valid_kwargs` and use it to annotate `__init__` for both runtime validation and the auto-generated docstring.

```py
from ...processing_utils import Unpack, VideosKwargs
from ...utils import auto_docstring
from ...video_processing_utils import BaseVideoProcessor

class MyModelVideoProcessorKwargs(VideosKwargs, total=False):
    min_frames: int
    max_frames: int

@auto_docstring
class MyModelVideoProcessor(BaseVideoProcessor):
    size = {"shortest_edge": 224}
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_normalize = True
    do_sample_frames = True
    num_frames = 16
    model_input_names = ["pixel_values_videos"]
    valid_kwargs = MyModelVideoProcessorKwargs

    def __init__(self, **kwargs: Unpack[MyModelVideoProcessorKwargs]):
        super().__init__(**kwargs)
```

Override [`~BaseVideoProcessor.sample_frames`] only when the model requires a sampling rule that the base uniform sampler can't express. For example, some models enforce a minimum or maximum number of frames, or sample based on model-specific constraints.

If the model's forward method expects a legacy input name, override `preprocess` and rename the key after calling the base implementation.

```py
class MyModelVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values"]

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch
```

Save the video processor with the checkpoint by instantiating it in the conversion script and calling [`~BaseVideoProcessor.save_pretrained`]. If a [`ProcessorMixin`] wraps the video processor, call [`~ProcessorMixin.save_pretrained`] instead. Do not manually create or edit preprocessing config files.

> [!TIP]
> See [`Qwen3VLVideoProcessor`] for reference.


## Register the classes

Expose the processing classes from the model package `__init__.py`. Follow the lazy import pattern used by nearby models and guard imports with the same optional dependencies required by each backend.

Map the new classes to the model config so the `Auto` classes can load them. The generated auto mapping file has a warning at the top. Do not edit it by hand. Add or update the model config, then run:

```bash
python utils/check_auto.py --fix_and_overwrite
```

After the mapping is generated, verify the model type appears in the relevant mappings in `src/transformers/models/auto/auto_mappings.py`.

- `IMAGE_PROCESSOR_MAPPING_NAMES` for [`AutoImageProcessor`]
- `VIDEO_PROCESSOR_MAPPING_NAMES` for [`AutoVideoProcessor`]

## Testing

Add tests for each vision processing component in the model test directory. Image and video processor tests follow the same pattern. Inherit from the shared mixin, indicate the fast and slow processing classes when automatic discovery isn't enough, provide model-specific init kwargs, and override the input name when the model uses a non-default output key.

### Image processor tests

Image processor tests usually live in `tests/models/<model_name>/test_image_processing_<model_name>.py` and inherit from [`ImageProcessingTestMixin`].

The image processing mixin finds the image processor classes from `IMAGE_PROCESSOR_MAPPING_NAMES`. Expose model-specific defaults through `image_processor_dict`. Add a tester object only when you need reusable dummy inputs or helper methods for focused tests.

```py
from transformers.testing_utils import require_torch, require_vision
from ...test_image_processing_common import ImageProcessingTestMixin

@require_torch
@require_vision
class MyModelImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    @property
    def image_processor_dict(self):
        return {"size": {"shortest_edge": 224}, "do_resize": True}
```

Add focused tests for behavior the mixin can't infer, such as custom resizing rules or model-specific kwargs.

Post-processing test mixins are available in `tests/test_image_processing_common.py` and are added on top of [`ImageProcessingTestMixin`].

```py
class MyModelImageProcessingTest(ImageProcessingTestMixin, MyTaskPostProcessTestMixin, unittest.TestCase):
```

The tests automatically verify that the correct mixins are used for your model. Mixins for new tasks must be added to `tests/test_image_processing_common.py`.


### Video processor tests

Video processor tests usually live in `tests/models/<model_name>/test_video_processing_<model_name>.py` and inherit from [`VideoProcessingTestMixin`]. Set `fast_video_processing_class`, define `video_processor_dict`, and override `input_name` if the model uses a key other than `pixel_values_videos`.

```py
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available
from ...test_video_processing_common import VideoProcessingTestMixin

@require_torch
@require_vision
class MyModelVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MyModelVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    @property
    def video_processor_dict(self):
        return {"size": {"shortest_edge": 224}, "num_frames": 16}
```

Add focused video tests for frame sampling, metadata handling, decoded video inputs, list-of-frame inputs, and output shapes. If your processor renames `pixel_values_videos`, assert the renamed key is returned.

If the model also has a [`ProcessorMixin`] that wraps the image or video processor, add `tests/models/<model_name>/test_processing_<model_name>.py` and inherit from [`ProcessorTesterMixin`]. Set `processor_class` and override `_setup_<component>()` class methods for components that can't be constructed without arguments. Use `_setup_test_attributes()` to expose placeholder tokens used by the common processor tests.

```py
from ...test_processing_common import ProcessorTesterMixin

class MyModelProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    @classmethod
    def _setup_image_processor(cls):
        return cls._get_component_class_from_processor("image_processor")(size={"shortest_edge": 224})

    @classmethod
    def _setup_video_processor(cls):
        return cls._get_component_class_from_processor("video_processor")(num_frames=2)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = getattr(processor, "image_token", "")
        cls.video_token = getattr(processor, "video_token", "")
```

## Next steps

- Read the [Auto-generating docstrings](./auto_docstring) guide to auto-generate consistent docstrings with `@auto_docstring`.
- Read the [Image processors](./image_processors) and [Video processors](./video_processors) guides for user-facing preprocessing behavior.
