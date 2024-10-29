from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers.models.paligemma.processing_paligemma import (
    PaliGemmaImagesKwargs,
    PaliGemmaProcessor,
    PaliGemmaProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
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
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[NewKwargsModelProcessorKwargs],
    ) -> BatchFeature:
        super().__call__(images=images, text=text, audio=audio, videos=videos, **kwargs)
