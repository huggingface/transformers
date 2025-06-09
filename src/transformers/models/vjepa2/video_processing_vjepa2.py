"""Fast Video processor class for VJEPA2."""

from ...image_processing_utils import BatchFeature
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

    def _preprocess(self, **kwargs) -> BatchFeature:
        out = super()._preprocess(**kwargs)
        pixel_values_videos = out.pop("pixel_values_videos")
        pixel_values = pixel_values_videos.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
        return_tensors = out.get("return_tensors", None)
        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)


__all__ = ["VJEPA2VideoProcessor"]
