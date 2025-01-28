from transformers.models.llava_next.image_processing_llava_next_fast import LlavaNextImageProcessorFast

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
)
from ...utils import add_start_docstrings, logging


logger = logging.get_logger(__name__)


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        image_grid_pinpoints (`List[List[int]]`, *optional*):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processing videos.
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """,
)
class LlavaOnevisionImageProcessorFast(LlavaNextImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    crop_size = None
    default_to_square = False
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    image_grid_pinpoints = [[384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304], [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920], [768, 2304], [1152, 384], [1152, 768], [1152, 1152], [1152, 1536], [1152, 1920], [1152, 2304], [1536, 384], [1536, 768], [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304], [1920, 384], [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304], [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], [2304, 2304]]  # fmt: skip
    valid_extra_kwargs = ["image_grid_pinpoints", "do_pad"]
    model_input_names = ["pixel_values_videos"]


__all__ = ["LlavaOnevisionImageProcessorFast"]
