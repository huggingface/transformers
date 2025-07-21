# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Any, Union, overload

import numpy as np

from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


# Copied from transformers.pipelines.text_classification.sigmoid
def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


# Copied from transformers.pipelines.text_classification.softmax
def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


# Copied from transformers.pipelines.text_classification.ClassificationFunction
class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=True),
    r"""
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.""",
)
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    >>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    """

    function_to_apply: ClassificationFunction = ClassificationFunction.NONE
    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
        )

    def _sanitize_parameters(self, top_k=None, function_to_apply=None, timeout=None):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction(function_to_apply.lower())
        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        return preprocess_params, {}, postprocess_params

    @overload
    def __call__(self, inputs: Union[str, "Image.Image"], **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(self, inputs: Union[list[str], list["Image.Image"]], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(
        self, inputs: Union[str, list[str], "Image.Image", list["Image.Image"]], **kwargs: Any
    ) -> Union[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            inputs (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
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
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs:
            inputs = kwargs.pop("images")
        if inputs is None:
            raise ValueError("Cannot call the image-classification pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        if self.framework == "pt":
            model_inputs = model_inputs.to(self.torch_dtype)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, function_to_apply=None, top_k=5):
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE

        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        outputs = model_outputs["logits"][0]
        if self.framework == "pt" and outputs.dtype in (torch.bfloat16, torch.float16):
            outputs = outputs.to(torch.float32).numpy()
        else:
            outputs = outputs.numpy()

        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]
        dict_scores.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            dict_scores = dict_scores[:top_k]

        return dict_scores
