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

from collections.abc import Sequence
from typing import Any, TypedDict, Union

from typing_extensions import TypeAlias, overload

from ..image_utils import is_pil_image
from ..utils import is_vision_available, requires_backends
from .base import Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image


ImagePair: TypeAlias = Sequence[Union["Image.Image", str]]


class Keypoint(TypedDict):
    x: float
    y: float


class Match(TypedDict):
    keypoint_image_0: Keypoint
    keypoint_image_1: Keypoint
    score: float


def validate_image_pairs(images: Any) -> Sequence[Sequence[ImagePair]]:
    error_message = (
        "Input images must be a one of the following :",
        " - A pair of images.",
        " - A list of pairs of images.",
    )

    def _is_valid_image(image):
        """images is a PIL Image or a string."""
        return is_pil_image(image) or isinstance(image, str)

    if isinstance(images, Sequence):
        if len(images) == 2 and all((_is_valid_image(image)) for image in images):
            return [images]
        if all(
            isinstance(image_pair, Sequence)
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
        if self.framework != "pt":
            raise ValueError("Keypoint matching pipeline only supports PyTorch (framework='pt').")

    def _sanitize_parameters(self, threshold=None, timeout=None):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        postprocess_params = {}
        if threshold is not None:
            postprocess_params["threshold"] = threshold
        return preprocess_params, {}, postprocess_params

    @overload
    def __call__(self, inputs: ImagePair, threshold: float = 0.0, **kwargs: Any) -> list[Match]: ...

    @overload
    def __call__(self, inputs: list[ImagePair], threshold: float = 0.0, **kwargs: Any) -> list[list[Match]]: ...

    def __call__(
        self,
        inputs: Union[list[ImagePair], ImagePair],
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> Union[list[Match], list[list[Match]]]:
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

            threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for keypoint matching. Keypoints matched with a lower matching score will be filtered out.
                A value of 0 means that all matched keypoints will be returned.

            kwargs:
                `timeout (`float`, *optional*, defaults to None)`
                    The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                    the call may block forever.

        Return:
            Union[list[Match], list[list[Match]]]:
                A list of matches or a list if a single image pair is provided, or of lists of matches if a batch
                of image pairs is provided. Each match is a dictionary containing the following keys:

                - **keypoint_image_0** (`Keypoint`): The keypoint in the first image (x, y coordinates).
                - **keypoint_image_1** (`Keypoint`): The keypoint in the second image (x, y coordinates).
                - **score** (`float`): The matching score between the two keypoints.
        """
        if inputs is None:
            raise ValueError("Cannot call the keypoint-matching pipeline without an inputs argument!")
        formatted_inputs = validate_image_pairs(inputs)
        outputs = super().__call__(formatted_inputs, threshold=threshold, **kwargs)
        if len(formatted_inputs) == 1:
            return outputs[0]
        return outputs

    def preprocess(self, images, timeout=None):
        images = [load_image(image, timeout=timeout) for image in images]
        model_inputs = self.image_processor(images=images, return_tensors=self.framework)
        model_inputs = model_inputs.to(self.dtype)
        target_sizes = [image.size for image in images]
        preprocess_outputs = {"model_inputs": model_inputs, "target_sizes": target_sizes}
        return preprocess_outputs

    def _forward(self, preprocess_outputs):
        model_inputs = preprocess_outputs["model_inputs"]
        model_outputs = self.model(**model_inputs)
        forward_outputs = {"model_outputs": model_outputs, "target_sizes": [preprocess_outputs["target_sizes"]]}
        return forward_outputs

    def postprocess(self, forward_outputs, threshold=0.0) -> list[Match]:
        model_outputs = forward_outputs["model_outputs"]
        target_sizes = forward_outputs["target_sizes"]
        postprocess_outputs = self.image_processor.post_process_keypoint_matching(
            model_outputs, target_sizes=target_sizes, threshold=threshold
        )
        postprocess_outputs = postprocess_outputs[0]
        pair_result = []
        for kp_0, kp_1, score in zip(
            postprocess_outputs["keypoints0"],
            postprocess_outputs["keypoints1"],
            postprocess_outputs["matching_scores"],
        ):
            kp_0 = Keypoint(x=kp_0[0].item(), y=kp_0[1].item())
            kp_1 = Keypoint(x=kp_1[0].item(), y=kp_1[1].item())
            pair_result.append(Match(keypoint_image_0=kp_0, keypoint_image_1=kp_1, score=score.item()))
        pair_result = sorted(pair_result, key=lambda x: x["score"], reverse=True)
        return pair_result
