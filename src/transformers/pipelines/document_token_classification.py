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
from typing import List, Optional, Tuple, Union

import numpy as np

from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
    is_torch_available,
    is_vision_available,
    logging,
)
from .base import PIPELINE_INIT_ARGS, Pipeline

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


@add_end_docstrings(PIPELINE_INIT_ARGS)
class DocumentTokenClassificationPipeline(Pipeline):
    # TODO: Update task_summary docs to include an example with document QA and then update the first sentence
    """
    Document Token Classification pipeline using any `AutoModelForDocumentTokenClassification`. The inputs/outputs are
    similar to the (extractive) Token Classification pipeline; however, the pipeline takes an image (and optional OCR'd
    words/boxes) as input instead of text context.

    This Document Token Classification pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-token-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on a Document Token Classification task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-token-classification).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_DOCUMENT_TOKEN_CLASSIFICATION_MAPPING)
        self.image_processor = self.feature_extractor
        self.image_processor.apply_ocr = False
        if self.model.config.model_type == "layoutlmv3":
            self.model_type = ModelType.LayoutLMv3

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
        image: Union["Image.Image", str],
        word_boxes: Tuple[str, List[float]] = None,
        words: List[str] = None,
        boxes: List[List[float]] = None,
        **kwargs,
    ):
        """
        Classifies the list of tokens (word_boxes) given a document. A document is defined as an image and an
        optional list of (word, box) tuples which represent the text in the document. If the `word_boxes` are not
        provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for
        LayoutLM-like models which require them as input.

        You can invoke the pipeline several ways:

        - `pipeline(image=image, word_boxes=word_boxes)`
        - `pipeline(image=image, words=words, boxes=boxes)`
        - `pipeline(image=image)`

        Args:
            image (`str` or `PIL.Image`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. If given a single image, it can be
                broadcasted to multiple questions.
            word_boxes (`List[str, Tuple[float, float, float, float]]`, *optional*):
                A list of words and bounding boxes (normalized 0->1000). If you provide this optional input, then the
                pipeline will use these words and boxes instead of running OCR on the image to derive them for models
                that need them (e.g. LayoutLM). This allows you to reuse OCR'd results across many invocations of the
                pipeline without having to re-run it each time.
            words (`List[str]`, *optional*):
                A list of words. If you provide this optional input, then the pipeline will use these words instead of
                running OCR on the image to derive them for models that need them (e.g. LayoutLM). This allows you to
                reuse OCR'd results across many invocations of the pipeline without having to re-run it each time.
            boxes (`List[Tuple[float, float, float, float]]`, *optional*):
                A list of bounding boxes (normalized 0->1000). If you provide this optional input, then the pipeline will
                use these boxes instead of running OCR on the image to derive them for models that need them (e.g.
                LayoutLM). This allows you to reuse OCR'd results across many invocations of the pipeline without having
                to re-run it each time.
            TODO doc_stride (`int`, *optional*, defaults to 128):
                If the words in the document are too long to fit with the question for the model, it will be split in
                several chunks with some overlap. This argument controls the size of that overlap.
            TODO max_seq_len (`int`, *optional*, defaults to 384):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using `doc_stride` as overlap) if needed.
            TODO max_question_len (`int`, *optional*, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            lang (`str`, *optional*):
                Language to use while running OCR. Defaults to english.
            tesseract_config (`str`, *optional*):
                Additional flags to pass to tesseract while running OCR.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **logits** - (`List[float]`) - The list of raw logits for each word in the document.
            - **labels** - (`List[str]`) - The list of predicted labels for each word in the document.
        """
        if word_boxes is not None:
            inputs = {
                "image": image,
                "word_boxes": word_boxes,
            }
        else:
            inputs = image
            self.image_processor.apply_ocr = True
        return super().__call__(inputs, **kwargs)

    def preprocess(self, input, lang=None, tesseract_config="", **kwargs):
        image = None
        if isinstance(input, str) or isinstance(input, Image.Image):
            image = load_image(input)
            input = {"image": image}
        elif input.get("image", None) is not None:
            image = load_image(input["image"])

        words, boxes = None, None
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

        # add pixel values
        images = features.pop("pixel_values")
        encoded_inputs["pixel_values"] = images
        encoded_inputs["word_ids"] = encoded_inputs.word_ids()
        encoded_inputs["words"] = words if words is not None else features["words"]

        return encoded_inputs

    def _forward(self, model_inputs):
        word_ids = model_inputs.pop("word_ids", None)
        words = model_inputs.pop("words", None)

        model_outputs = self.model(**model_inputs)

        model_outputs["word_ids"] = word_ids
        model_outputs["words"] = words
        model_outputs["attention_mask"] = model_inputs.get("attention_mask", None)
        return model_outputs

    def postprocess(self, model_outputs, **kwargs):
        logits = model_outputs["logits"]
        if self.framework == "pt":
            logits = logits.detach().cpu().numpy()
        words = model_outputs["words"]
        # if first dimension is 1, remove it
        if logits.shape[0] == 1:
            logits = logits[0]
        # if words is a list of list of strings, get the first one
        if isinstance(words, list) and isinstance(words[0], list):
            words = words[0]
            model_outputs["words"] = words
        token_predictions = logits.argmax(-1)

        word_ids = model_outputs["word_ids"]

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
