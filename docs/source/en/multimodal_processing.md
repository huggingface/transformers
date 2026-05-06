<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Multimodal Processors

A processor combines multiple input processing components — an image processor, tokenizer, feature extractor, video processor — into a single object with a unified `__call__` interface. For generative multimodal models this means encoding inputs along with placeholder tokens; for other models like [`CLIP`] it means encoding each modality independently and returning a merged input dictionary. In both cases the processor handles modality-specific preprocessing, batching, and padding so the model receives consistently formatted tensors.

For models that interleave visual inputs with text, [`ProcessorMixin`] additionally handles the placeholder token pattern: special tokens embedded in the prompt — e.g. `<image>`, `<video>`, `<audio>` — are replaced by expanded strings of N repeated soft tokens, optionally wrapped in begin/end-of-modality boundary tokens. The number of repetitions is model-specific and depends on the input itself (image resolution, video frame count, audio duration) — this is what `replace_image_token` / `replace_video_token` / `replace_audio_token` compute. Each processor exposes the placeholder token and its vocabulary ID via `self.image_token` / `self.image_token_id`, with video and audio following the same pattern.


## Adding a new processor

Define your processor class first by creating `src/transformers/models/<model>/processing_<my_model_name>.py` and subclass `ProcessorMixin`. Make sure to define a `TypedDict` object with default values and assign it as `cls.valid_processor_kwargs`

```python
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack

class MyModelProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MyModelImageProcessorKwargs
    _defaults = {
        "text_kwargs": {"padding": True},
        "images_kwargs": {"do_convert_rgb": True},
    }

class MyModelProcessor(ProcessorMixin):
    valid_processor_kwargs = MyModelProcessorKwargs

    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )
```

Then implement `replace_<modality>_token` if needed. It receives the full output dict from the subprocessor and the index of the current input, and returns the expanded replacement string for that input. The replacement string is whatever your model expects in the input sequence.

If your model does not use placeholder repetition at all (no `image_token` defined), you do not need to override this method — just leave `self.image_token` unset and the base class skips replacement entirely.

```python
def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
    num_crops = image_inputs["num_crops"][image_idx]
    return f"{self.boi_token}{self.image_token * self.num_image_tokens * num_crops}{self.eoi_token}"
```


Optionally override `prepare_inputs_layout` and `validate_inputs` methods. If your model requires a specific input structure before processing begins, such as re-ordering images as a nested list, or a model-specific validation on top of the common checks.

```python
def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
    images, text, videos, audio = super().prepare_inputs_layout(
        images=images, text=text, videos=videos, audio=audio, **kwargs
    )
    # Call `super()` to apply common preparation steps first 
    images, text, videos, audio = super().prepare_inputs_layout(images, text, videos, audio)
    if images is not None:
        images = make_nested_list_of_images(images)
    return images, text, videos, audio

def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
    super().validate_inputs(images=images, text=text, **kwargs)
    if text is not None and images is not None:
        n_tokens = [s.count(self.image_token) for s in text]
        n_images = [len(img_list) for img_list in images]
        if n_tokens != n_images:
            raise ValueError(
                f"Number of {self.image_token} tokens in text {n_tokens} does not match "
                f"number of images {n_images}."
            )
```

> [!TIP]
> See [`Gemma4Processor`] and [`Qwen2VLProcessor`] for reference.

## Testing

All multimodal processors should have a test class that inherits from [`ProcessorTestMixin`]. This mixin provides a standard suite covering tokenization, image processing, batching, and round-trip encoding.

```python
# tests/models/my_model_name/test_processor_{{my_model_name}}.py

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available
from ...test_processing_common import ProcessorTestMixin

if is_vision_available():
    from transformers import MyModelProcessor

@require_vision
class MyModelProcessorTest(ProcessorTestMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    def get_processor(self):
        return MyModelProcessor.from_pretrained("hf-internal-testing/my-model-test")
```
