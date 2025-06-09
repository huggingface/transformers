"""Fast Video processor class for VJEPA2."""

from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    is_vision_available,
)
from ...utils.import_utils import requires
from ...video_processing_utils import (
    BaseVideoProcessor,
)


if is_vision_available():
    from ...image_utils import PILImageResampling


class VJEPA2VideoProcessorInitKwargs(VideosKwargs): ...


@requires(backends=("torchvision",))
class VJEPA2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    crop_size = 256
    do_resize = True
    do_rescale = False
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = VJEPA2VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[VJEPA2VideoProcessorInitKwargs]):
        self.size = {"height": self.crop_size, "width": self.crop_size}
        super().__init__(**kwargs)


__all__ = ["VJEPA2VideoProcessor"]
