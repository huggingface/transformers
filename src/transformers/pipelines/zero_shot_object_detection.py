from typing import Dict, List, Union

import numpy as np

from ..tokenization_utils_base import BatchEncoding
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

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotObjectDetectionPipeline(Pipeline):
    """
    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING)

    def __call__(
        self,
        images: Union[str, List[str], "Image.Image", List["Image.Image"]],
        text_queries: Union[str, List[str], List[List[str]]] = None,
        **kwargs
    ):
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an http url pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

            text_queries (`str` or `List[str]` or `List[List[str]]`): Text queries to query the target image with.
            If given multiple images, `text_queries` should be provided as a list of lists, where each nested list
            contains the text queries for the corresponding image.

            threshold (`float`, *optional*, defaults to 0.1):
                The probability necessary to make a prediction.

            top_k (`int`, *optional*, defaults to None):
                The number of top predictions that will be returned by the pipeline. If the provided number is `None`
                or higher than the number of predictions available, it will default to the number of predictions.


        Return:
            A list of lists containing prediction results, one list per input image. Each list contains dictionaries
            with the following keys:

            - **label** (`str`) -- Text query corresponding to the found object.
            - **score** (`float`) -- Score corresponding to the object (between 0 and 1).
            - **box** (`Dict[str,int]`) -- Bounding box of the detected object in image's original size. It is a
              dictionary with `x_min`, `x_max`, `y_min`, `y_max` keys.
        """
        if isinstance(text_queries, str) or (isinstance(text_queries, List) and not isinstance(text_queries[0], List)):
            if isinstance(images, (str, Image.Image)):
                inputs = {"images": images, "text_queries": text_queries}
            elif isinstance(images, List):
                assert len(images) == 1, "Input text_queries and images must have correspondance"
                inputs = {"images": images[0], "text_queries": text_queries}
            else:
                raise TypeError(f"Innapropriate type of images: {type(images)}")

        elif isinstance(text_queries, str) or (isinstance(text_queries, List) and isinstance(text_queries[0], List)):
            if isinstance(images, (Image.Image, str)):
                images = [images]
            assert len(images) == len(text_queries), "Input text_queries and images must have correspondance"
            inputs = {"images": images, "text_queries": text_queries}
        else:
            """
            Supports the following format
            - {"images": images, "text_queries": text_queries}
            """
            inputs = images
        results = super().__call__(inputs, **kwargs)
        return results

    def _sanitize_parameters(self, **kwargs):
        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]
        return {}, {}, postprocess_params

    def preprocess(self, inputs):
        if not isinstance(inputs["images"], List):
            inputs["images"] = [inputs["images"]]
        images = [load_image(img) for img in inputs["images"]]
        text_queries = inputs["text_queries"]
        if isinstance(text_queries, str) or isinstance(text_queries[0], str):
            text_queries = [text_queries]

        target_sizes = [torch.IntTensor([[img.height, img.width]]) for img in images]
        target_sizes = torch.cat(target_sizes)
        inputs = self._processor(text=inputs["text_queries"], images=images, return_tensors="pt")
        return {"target_sizes": target_sizes, "text_queries": text_queries, **inputs}

    def _forward(self, model_inputs):
        target_sizes = model_inputs.pop("target_sizes")
        text_queries = model_inputs.pop("text_queries")
        outputs = self.model(**model_inputs)

        model_outputs = outputs.__class__({"target_sizes": target_sizes, "text_queries": text_queries, **outputs})
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.1, top_k=None):
        texts = model_outputs["text_queries"]

        outputs = self.feature_extractor.post_process(
            outputs=model_outputs, target_sizes=model_outputs["target_sizes"]
        )

        results = []
        for i in range(len(outputs)):
            keep = outputs[i]["scores"] >= threshold
            labels = outputs[i]["labels"][keep].tolist()
            scores = outputs[i]["scores"][keep].tolist()
            boxes = [self._get_bounding_box(box) for box in outputs[i]["boxes"][keep]]

            result = [
                {"score": score, "label": texts[i][label], "box": box}
                for score, label, box in zip(scores, labels, boxes)
            ]

            result = sorted(result, key=lambda x: x["score"], reverse=True)
            if top_k:
                result = result[:top_k]
            results.append(result)

        return results

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox

    # Replication of OwlViTProcessor __call__ method, since pipelines don't auto infer processor's yet!
    def _processor(self, text=None, images=None, padding="max_length", return_tensors="np", **kwargs):
        """
        Main method to prepare for the model one or several text(s) and image(s). This method forwards the `text` and
        `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode:
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPFeatureExtractor's [`~CLIPFeatureExtractor.__call__`] if `images` is not `None`. Please refer to the
        doctsring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
            `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None and images is None:
            raise ValueError("You have to specify at least one text or image. Both cannot be none.")

        if text is not None:
            if isinstance(text, str) or (isinstance(text, List) and not isinstance(text[0], List)):
                encodings = [self.tokenizer(text, padding=padding, return_tensors=return_tensors, **kwargs)]

            elif isinstance(text, List) and isinstance(text[0], List):
                encodings = []

                # Maximum number of queries across batch
                max_num_queries = max([len(t) for t in text])

                # Pad all batch samples to max number of text queries
                for t in text:
                    if len(t) != max_num_queries:
                        t = t + [" "] * (max_num_queries - len(t))

                    encoding = self.tokenizer(t, padding=padding, return_tensors=return_tensors, **kwargs)
                    encodings.append(encoding)
            else:
                raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

            if return_tensors == "np":
                input_ids = np.concatenate([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = np.concatenate([encoding["attention_mask"] for encoding in encodings], axis=0)

            elif return_tensors == "pt" and is_torch_available():
                import torch

                input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
                attention_mask = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)

            elif return_tensors == "tf" and is_tf_available():
                import tensorflow as tf

                input_ids = tf.stack([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = tf.stack([encoding["attention_mask"] for encoding in encodings], axis=0)

            else:
                raise ValueError("Target return tensor type could not be returned")

            encoding = BatchEncoding()
            encoding["input_ids"] = input_ids
            encoding["attention_mask"] = attention_mask

        if images is not None:
            image_features = self.feature_extractor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
