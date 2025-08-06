# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union

from typing_extensions import overload

from ..image_utils import is_pil_image
from ..utils import is_vision_available, requires_backends
from .base import Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image


def validate_image_pairs(images: Union[list, list[list]]):
    error_message = (
        "Input images must be a one of the following :",
        " - A pair of images.",
        " - A list of pairs of images.",
    )

    def _is_valid_image(image):
        """images is a PIL Image or a string."""
        return is_pil_image(image) or isinstance(image, str)

    if isinstance(images, list):
        if len(images) == 2 and all((_is_valid_image(image)) for image in images):
            return [images]
        if all(
            isinstance(image_pair, list)
            and len(image_pair) == 2
            and all(_is_valid_image(image) for image in image_pair)
            for image_pair in images
        ):
            return images
    raise ValueError(error_message)


class KeypointMatchingPipeline(Pipeline):
    """
    Keypoint matching pipeline using any `AutoModelForKeypointMatching`. This pipeline matches keypoints between two images.
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")

    def _sanitize_parameters(self, threshold=None, timeout=None):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        postprocess_params = {}
        if threshold is not None:
            postprocess_params["threshold"] = threshold
        return preprocess_params, {}, postprocess_params

    @overload
    def __call__(self, inputs: Union[str, "Image.Image"], **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(self, inputs: Union[list[str], list["Image.Image"]], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(
        self, inputs: Union[str, list[str], "Image.Image", list["Image.Image"]], **kwargs: Any
    ) -> Union[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """
        Find matches between keypoints in two images.

        Args:
            inputs (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single pair of images or a batch of image pairs, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.

            kwargs:
                - `threshold` (`float`, *optional*, defaults to 0.0):
                    The threshold to use for keypoint matching.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        if inputs is None:
            raise ValueError("Cannot call the keypoint-matching pipeline without an inputs argument!")
        formatted_inputs = validate_image_pairs(inputs)
        return super().__call__(formatted_inputs, **kwargs)

    def preprocess(self, images, timeout=None):
        images = [load_image(image, timeout=timeout) for image in images]
        model_inputs = self.image_processor(images=images, return_tensors=self.framework)
        if self.framework == "pt":
            model_inputs = model_inputs.to(self.torch_dtype)
        target_sizes = [image.size for image in images]
        preprocess_outputs = {"model_inputs": model_inputs, "target_sizes": target_sizes}
        return preprocess_outputs

    def _forward(self, preprocess_outputs):
        model_inputs = preprocess_outputs["model_inputs"]
        model_outputs = self.model(**model_inputs)
        forward_outputs = {"model_outputs": model_outputs, "target_sizes": [preprocess_outputs["target_sizes"]]}
        return forward_outputs

    def postprocess(self, forward_outputs, threshold=0.0):
        model_outputs = forward_outputs["model_outputs"]
        target_sizes = forward_outputs["target_sizes"]
        postprocess_outputs = self.image_processor.post_process_keypoint_matching(
            model_outputs, target_sizes=target_sizes, threshold=threshold
        )
        postprocess_outputs = postprocess_outputs[0]
        dict_result = {
            "keypoints0": postprocess_outputs["keypoints0"].tolist(),
            "keypoints1": postprocess_outputs["keypoints1"].tolist(),
            "matching_scores": postprocess_outputs["matching_scores"].tolist(),
        }
        return dict_result
