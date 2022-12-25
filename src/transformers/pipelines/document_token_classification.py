# Copyright 2022 The Loop Team and the HuggingFace Team. All rights reserved.
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

import re
from typing import List, Optional, Tuple, Union, Dict

import numpy as np

from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
    is_torch_available,
    is_vision_available,
    logging,
)
from .base import PIPELINE_INIT_ARGS, Pipeline, ArgumentHandler, Dataset, types

if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_DOCUMENT_TOKEN_CLASSIFICATION_MAPPING

TESSERACT_LOADED = False
if is_pytesseract_available():
    TESSERACT_LOADED = True
    import pytesseract

logger = logging.get_logger(__name__)


class ModelType(ExplicitEnum):
    LayoutLMv3 = "layoutlmv3"
    LayoutLMv2 = "layoutlmv2"


class DocumentTokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):

        if inputs is not None and (
            (isinstance(inputs, (list, tuple)) and len(inputs) > 0) or isinstance(inputs, types.GeneratorType)
        ):
            inputs = list(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
        elif Dataset is not None and isinstance(inputs, Dataset) or isinstance(inputs, types.GeneratorType):
            return inputs
        else:
            raise ValueError("At least one input is required.")
        return inputs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class DocumentTokenClassificationPipeline(Pipeline):
    # TODO: Update task_summary docs to include an example with document token classification
    """
    Document Token Classification pipeline using any `AutoModelForDocumentTokenClassification`. The inputs/outputs are
    similar to the Token Classification pipeline; however, the pipeline takes an image (and optional OCR'd
    words/boxes) as input instead of text context.

    This Document Token Classification pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-token-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on a Document Token Classification task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-token-classification).
    """

    def __init__(self, args_parser=DocumentTokenClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_DOCUMENT_TOKEN_CLASSIFICATION_MAPPING)
        self.image_processor = self.feature_extractor
        self.image_processor.apply_ocr = False
        self._args_parser = args_parser
        if self.model.config.model_type == "layoutlmv3":
            self.model_type = ModelType.LayoutLMv3
        elif self.model.config.model_type == "layoutlmv2":
            self.model_type = ModelType.LayoutLMv2
        else:
            raise ValueError(f"Model type {self.model.config.model_type} is not supported by this pipeline.")

    def _sanitize_parameters(
        self,
        padding=None,
        doc_stride=None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        max_seq_len=None,
        **kwargs,
    ):
        preprocess_params, postprocess_params = {}, {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_seq_len is not None:
            preprocess_params["max_seq_len"] = max_seq_len
        if lang is not None:
            preprocess_params["lang"] = lang
        if tesseract_config is not None:
            preprocess_params["tesseract_config"] = tesseract_config

        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        inputs: Union["Image.Image", List["Image.Image"], str, Dict, List[dict]],
        **kwargs,
    ):
        """
        Classifies the list of tokens (word_boxes) given a document. A document is defined as an image and an
        optional list of (word, box) tuples which represent the text in the document. If the `word_boxes` are not
        provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for
        LayoutLM-like models which require them as input.

        You can invoke the pipeline several ways:

        - `pipeline(inputs=image)`
        - `pipeline(inputs=[image])`
        - `pipeline(inputs={"image": image})`
        - `pipeline(inputs={"image": image, "word_boxes": word_boxes})`
        - `pipeline(inputs={"image": image, "words": words, "boxes": boxes})`
        - `pipeline(inputs=[{"image": image}])`
        - `pipeline(inputs=[{"image": image, "word_boxes": word_boxes}])`
        - `pipeline(inputs=[{"image": image, "words": words, "boxes": boxes}])`

        Args:
            inputs (:obj:`str`, :obj:`List[str]`, :obj:`PIL.Image`, :obj:`List[PIL.Image]`, :obj:`Dict`, :obj:`List[Dict]`):

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **words** (:obj:`List[str]`) -- The words in the document.
            - **boxes** (:obj:`List[List[int]]`) -- The boxes of the words in the document.
            - **word_labels** (:obj:`List[str]`) -- The predicted labels for each word.
        """
        inputs = self._args_parser(inputs)
        return super().__call__(inputs, **kwargs)

    def preprocess(self, input, lang=None, tesseract_config="", **kwargs):
        image = None
        if isinstance(input, str) or isinstance(input, Image.Image):
            image = load_image(input)
            input = {"image": image}
        elif input.get("image", None) is not None:
            image = load_image(input["image"])

        words, boxes = None, None
        self.image_processor.apply_ocr = False
        if "words" in input and "boxes" in input:
            words = input["words"]
            boxes = input["boxes"]
        elif "word_boxes" in input:
            words = [x[0] for x in input["word_boxes"]]
            boxes = [x[1] for x in input["word_boxes"]]
        elif image is not None and not TESSERACT_LOADED:
            raise ValueError(
                "If you provide an image without word_boxes, then the pipeline will run OCR using Tesseract,"
                " but pytesseract is not available"
            )
        else:
            self.image_processor.apply_ocr = True

        # first, apply the image processor
        features = self.image_processor(
            images=image,
            return_tensors=self.framework,
            **kwargs,
        )

        encoded_inputs = self.tokenizer(
            text=words if words is not None else features["words"],
            boxes=boxes if boxes is not None else features["boxes"],
            return_tensors=self.framework,
            **kwargs,
        )

        if self.model_type == ModelType.LayoutLMv3:
            image_field = "pixel_values"
        elif self.model_type == ModelType.LayoutLMv2:
            image_field = "image"
        encoded_inputs[image_field] = features.pop("pixel_values")

        # Fields that help with post-processing
        encoded_inputs["word_ids"] = encoded_inputs.word_ids()
        encoded_inputs["words"] = words if words is not None else features["words"]
        encoded_inputs["boxes"] = boxes if boxes is not None else features["boxes"]

        return encoded_inputs

    def _forward(self, model_inputs):
        word_ids = model_inputs.pop("word_ids", None)
        words = model_inputs.pop("words", None)
        boxes = model_inputs.pop("boxes", None)

        model_outputs = self.model(**model_inputs)

        model_outputs["word_ids"] = word_ids
        model_outputs["words"] = words
        model_outputs["boxes"] = boxes
        return model_outputs

    def postprocess(self, model_outputs, **kwargs):
        model_outputs = dict(model_outputs)
        logits = np.asarray(model_outputs.pop("logits", None))
        words = model_outputs["words"]
        boxes = model_outputs["boxes"]

        # if first dimension is 1, remove it
        if logits.shape[0] == 1:
            logits = logits[0]

        # if words is a list of list of strings, get the first one
        if isinstance(words, list) and len(words) != 0 and isinstance(words[0], list):
            words = words[0]
            model_outputs["words"] = words

        if isinstance(boxes, list) and len(boxes) != 0 and isinstance(boxes[0], list):
            boxes = boxes[0]
            model_outputs["boxes"] = boxes

        token_predictions = logits.argmax(-1)

        word_ids = model_outputs.pop("word_ids", None)

        # Map Token predictions to word predictions
        word_predictions = [None] * len(words)
        for word_id, token_prediction in zip(word_ids, token_predictions):
            if word_id is not None and word_predictions[word_id] is None:
                word_predictions[word_id] = token_prediction
            elif word_id is not None and word_predictions[word_id] != token_prediction:
                # If conflict, we take the first prediction
                pass

        word_labels = [self.model.config.id2label[prediction] for prediction in word_predictions]
        model_outputs["word_labels"] = word_labels
        return model_outputs
