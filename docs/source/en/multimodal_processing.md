<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Multimodal processors

A processor combines a tokenizer with one or more modality processors, such as an image processor, video processor, or feature extractor. It exposes a single `__call__` method that routes each input to the right component and merges the outputs into one dictionary.

Some multimodal models interleave text with images, videos, or audio. For these models, [`ProcessorMixin`] can replace placeholder tokens like `<image>`, `<video>`, and `<audio>` with the token pattern expected by the model.


## Adding a new processor

Define a processor class by creating `src/transformers/models/<model>/processing_<my_model_name>.py` and subclass `ProcessorMixin`. Make sure to define a `TypedDict` object with default values and assign it as `cls.valid_processor_kwargs`

```python
from ...processing_utils import ProcessorMixin

class MyModelProcessor(ProcessorMixin):

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

Implement `replace_<modality>_token` if needed. It receives the full output dict from the subprocessor and the index of the current input, and returns the expanded replacement string for that input. The replacement string is whatever the model expects in the input sequence.

If the model does not use placeholder repetition at all (no `image_token` defined), you do not need to override this method. Leave `self.image_token` unset and the base class skips replacement entirely.

```python
def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
    num_crops = image_inputs["num_crops"][image_idx]
    return f"{self.boi_token}{self.image_token * self.num_image_tokens * num_crops}{self.eoi_token}"
```


Optionally override `prepare_inputs_layout` and `validate_inputs` methods. If the model requires a specific input structure before processing begins, such as re-ordering images as a nested list, or a model-specific validation on top of the common checks.

```python
def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
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

### Adding custom arguments

There are three ways to add custom arguments to a processor:

| Where to add | Use for | Example |
|---|---|---|
| `__init__` | Settings saved in the config (model properties) | `image_seq_length` in [`Gemma3Processor`] |
| `__call__` | Inputs only known at runtime | `images` or `text` |
| Custom `MyModelProcessingKwargs` class | Settings that can be configured at runtime | `use_image_id` in [`MiniCPMV4_7Processor`] |

Arguments passed to `__init__` are saved in the config, have typically a default, and are not configurable at runtime.

Arguments passed to `__call__` are inputs only known at runtime. They must either have no default value or `None`.

Runtime arguments which can have a default must be added to a custom `MyModelProcessingKwargs` and will be passed as part of `**kwargs` to `__call__`. Defaults for these values can be set as processor class attributes. Defaults for modality-specific processors, for example the tokenizer or image processor, can be defined in the same way. This is usually not needed as modality-specific processors already have defaults defined in their respective classes. Only use this option if the processor must override a modality-specific processor setting. Of course, arguments passed at runtime to `__call__` have always higher priority than any defaults.

```python
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack

class MyModelProcessorKwargs(ProcessingKwargs, total=False):
    """
    use_image_id (`bool`, defaults to `False`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder.
    """
    use_image_id: bool


class MyModelProcessor(ProcessingMixin):
    valid_processor_kwargs = MyModelProcessorKwargs
    use_image_id = False

    text_kwargs = {"padding": True}   # default passed to tokenizer

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, image_seq_length: int = 256, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_seq_length = image_seq_length

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[MyModelProcessorKwargs],
    ):
        images, text, videos, audio = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)

        merged_kwargs = self._merge_kwargs(tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)

        merged_kwargs["use_image_id"] # is `False` as default from class attribute is added by _merge_kwargs
        merged_kwargs["text_kwargs"]["padding"] # is `True`
```



## Testing

All multimodal processors should have a test class that inherits from [`ProcessorTesterMixin`]. This mixin provides a standard suite covering tokenization, image processing, batching, and round-trip encoding.

```python
# tests/models/my_model_name/test_processor_<my_model_name>.py

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available
from ...test_processing_common import ProcessorTesterMixin

if is_vision_available():
    from transformers import MyModelProcessor

@require_vision
class MyModelProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    def get_processor(self):
        return MyModelProcessor.from_pretrained("hf-internal-testing/my-model-test")
```
