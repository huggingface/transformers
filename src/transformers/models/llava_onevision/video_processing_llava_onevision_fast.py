from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
)
from ...video_processing_utils_fast import (
    BaseVideoProcessorFast,
)


class LlavaOnevisionVideoProcessorFast(BaseVideoProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = ["LlavaOnevisionVideoProcessorFast"]
