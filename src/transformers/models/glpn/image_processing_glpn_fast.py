from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling


class GLPNImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for GLPN using torch/torchvision backend.
    """
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN  # [0.485, 0.456, 0.406]
    image_std = IMAGENET_STANDARD_STD    # [0.229, 0.224, 0.225]
    size = {"height": 480, "width": 640}
    do_resize = True
    do_rescale = True
    do_normalize = False
    do_convert_rgb = True
