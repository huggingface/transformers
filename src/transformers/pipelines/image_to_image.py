from typing import List, Union

import numpy as np

from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES

if is_torch_available():
    from ..modeling_outputs import ImageSuperResolutionOutput
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageToImagePipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES
        )

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        if "head_mask" in kwargs:
            forward_params["head_mask"] = kwargs["head_mask"]

        return preprocess_params, forward_params, postprocess_params

    def __call__(
        self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        return super().__call__(images, **kwargs)

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors=self.framework)
        return inputs

    def postprocess(self, model_outputs):
        images = []
        for output in model_outputs:
            if hasattr(self.image_processor, "post_process"):
                images.append(self.image_processor.post_process(output))
            elif isinstance(output, ImageSuperResolutionOutput):
                output = output.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            else:
                raise ValueError(f"Output {type(output)} is not yet supported.")
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            images.append(Image.fromarray(output))

        return images
