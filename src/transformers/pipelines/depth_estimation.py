from typing import List, Union

import numpy as np

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class DepthEstimationPipeline(Pipeline):
    """
    Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
    >>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> # This is a tensor with the values being the depth expressed in meters for each pixel
    >>> output["predicted_depth"].shape
    torch.Size([1, 384, 384])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This depth estimation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"depth-estimation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Predict the depth(s) of the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **predicted_depth** (`torch.Tensor`) -- The predicted depth by the model as a `torch.Tensor`.
            - **depth** (`PIL.Image`) -- The predicted depth by the model as a `PIL.Image`.
        """
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, timeout=None, **kwargs):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        return preprocess_params, {}, {}

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout)
        self.image_size = image.size
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs):
        predicted_depth = model_outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=self.image_size[::-1], mode="bicubic", align_corners=False
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        output_dict = {}
        output_dict["predicted_depth"] = predicted_depth
        output_dict["depth"] = depth
        return output_dict
