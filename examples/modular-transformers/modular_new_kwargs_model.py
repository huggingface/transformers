from typing import Optional

from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers.models.paligemma.processing_paligemma import (
    PaliGemmaImagesKwargs,
    PaliGemmaProcessor,
    PaliGemmaProcessorKwargs,
)


# class NewKwargsModelProcessor(PaliGemmaProcessor):
#     pass
class NewKwargsModelImagesKwargs(PaliGemmaImagesKwargs):
    do_convert_rgb: Optional[bool]
    do_something_else: Optional[bool]


class NewKwargsModelProcessorKwargs(PaliGemmaProcessorKwargs, total=False):
    images_kwargs: NewKwargsModelImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class NewKwargsModelForConditionalGeneration(PaliGemmaForConditionalGeneration):
    pass


class NewKwargsModelProcessor(PaliGemmaProcessor):
    pass
