# coding=utf-8
# Copyright 2024 Microsoft and The HuggingFace Inc. team.
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
"""
Processor class for Florence-2.
"""

import re
from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, is_valid_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging


if is_torch_available():
    import torch

if is_vision_available():
    from ...image_utils import PILImageResampling

logger = logging.get_logger(__name__)


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


class Florence2Processor(ProcessorMixin):
    r"""
    Constructs a Florence2 processor which wraps a Florence2 image processor and a Florence2 tokenizer into a single processor.

    [`Florence2Processor`] offers all the functionalities of [`CLIPImageProcessor`] and [`BartTokenizerFast`]. See the
    [`~Florence2Processor.__call__`] and [`~Florence2Processor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`BartTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        tokens_to_add = {
            "additional_special_tokens": tokenizer.additional_special_tokens
            + ["<od>", "</od>", "<ocr>", "</ocr>"]
            + [f"<loc_{x}>" for x in range(1000)]
            + [
                "<cap>",
                "</cap>",
                "<ncap>",
                "</ncap>",
                "<dcap>",
                "</dcap>",
                "<grounding>",
                "</grounding>",
                "<seg>",
                "</seg>",
                "<sep>",
                "<region_cap>",
                "</region_cap>",
                "<region_to_desciption>",
                "</region_to_desciption>",
                "<proposal>",
                "</proposal>",
                "<poly>",
                "</poly>",
                "<and>",
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)

        self.tasks_answer_post_processing_type = {
            "<OCR>": "pure_text",
            "<OCR_WITH_REGION>": "ocr",
            "<CAPTION>": "pure_text",
            "<DETAILED_CAPTION>": "pure_text",
            "<MORE_DETAILED_CAPTION>": "pure_text",
            "<OD>": "description_with_bboxes",
            "<DENSE_REGION_CAPTION>": "description_with_bboxes",
            "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
            "<REFERRING_EXPRESSION_SEGMENTATION>": "polygons",
            "<REGION_TO_SEGMENTATION>": "polygons",
            "<OPEN_VOCABULARY_DETECTION>": "description_with_bboxes_or_polygons",
            "<REGION_TO_CATEGORY>": "pure_text",
            "<REGION_TO_DESCRIPTION>": "pure_text",
            "<REGION_TO_OCR>": "pure_text",
            "<REGION_PROPOSAL>": "bboxes",
        }

        self.task_prompts_without_inputs = {
            "<OCR>": "What is the text in the image?",
            "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
            "<CAPTION>": "What does the image describe?",
            "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
            "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
            "<OD>": "Locate the objects with category name in the image.",
            "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
            "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
        }

        self.task_prompts_with_input = {
            "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
            "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
            "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
            "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
            "<REGION_TO_CATEGORY>": "What is the region {input}?",
            "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
            "<REGION_TO_OCR>": "What text is in the region {input}?",
        }

        self.post_processor = Florence2PostProcesser(tokenizer=tokenizer)

        super().__init__(image_processor, tokenizer)

    def _construct_prompts(self, text):
        # replace the task tokens with the task prompts if task token is in the text
        prompts = []
        for _text in text:
            # 1. fixed task prompts without additional inputs
            for task_token, task_prompt in self.task_prompts_without_inputs.items():
                if task_token in _text:
                    assert _text == task_token, f"Task token {task_token} should be the only token in the text."
                    _text = task_prompt
                    break
            # 2. task prompts with additional inputs
            for task_token, task_prompt in self.task_prompts_with_input.items():
                if task_token in _text:
                    _text = task_prompt.format(input=_text.replace(task_token, ""))
                    break
            prompts.append(_text)
        return prompts

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        tokenize_newline_separately: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        do_resize: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = "channels_first",  # noqa: F821
        input_data_format: Optional[
            Union[str, "ChannelDimension"]  # noqa: F821
        ] = None,
        resample: "PILImageResampling" = None,  # noqa: F821
        do_convert_rgb: bool = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_rescale: bool = None,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to BartTokenizerFast's [`~BartTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            tokenize_newline_separately (`bool`, defaults to `True`):
                Adds a separately tokenized '\n' at the end of the prompt.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
              is provided, the `input_ids` will also contain the suffix input ids.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **labels** -- Labels compatible with training if `suffix` is not None
        """

        return_token_type_ids = False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `Florence2Processor` instance.")
        if text is None:
            logger.warning_once("You are using Florence-2 without a text prompt.")
            text = ""

        if isinstance(text, List) and isinstance(images, List):
            if len(images) < len(text):
                raise ValueError(
                    f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
                )
        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        pixel_values = self.image_processor(
            images,
            do_resize=do_resize,
            do_normalize=do_normalize,
            return_tensors=return_tensors,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
            data_format=data_format,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
        )["pixel_values"]

        if max_length is not None:
            max_length -= self.image_seq_length  # max_length has to account for the image tokens

        text = self._construct_prompts(text)

        inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_token_type_ids=return_token_type_ids,
        )

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def post_process_generation(self, text, task, image_size):
        """
        Post-process the output of the model to each of the task outputs.

        Args:
            text (`str`): The text to post-process.
            task (`str`): The task to post-process the text for.
            image_size (`Tuple[int, int]`): The size of the image. height x width.
        """

        task_answer_post_processing_type = self.tasks_answer_post_processing_type.get(task, "pure_text")
        task_answer = self.post_processor(
            text=text,
            image_size=image_size,
            parse_tasks=task_answer_post_processing_type,
        )[task_answer_post_processing_type]

        if task_answer_post_processing_type == "pure_text":
            final_answer = task_answer
            # remove the special tokens
            final_answer = final_answer.replace("<s>", "").replace("</s>", "")
        elif task_answer_post_processing_type in ["od", "description_with_bboxes", "bboxes"]:
            od_instances = task_answer
            bboxes_od = [_od_instance["bbox"] for _od_instance in od_instances]
            labels_od = [str(_od_instance["cat_name"]) for _od_instance in od_instances]
            final_answer = {"bboxes": bboxes_od, "labels": labels_od}
        elif task_answer_post_processing_type in ["ocr"]:
            bboxes = [_od_instance["quad_box"] for _od_instance in task_answer]
            labels = [str(_od_instance["text"]) for _od_instance in task_answer]
            final_answer = {"quad_boxes": bboxes, "labels": labels}
        elif task_answer_post_processing_type in ["phrase_grounding"]:
            bboxes = []
            labels = []
            for _grounded_phrase in task_answer:
                for _bbox in _grounded_phrase["bbox"]:
                    bboxes.append(_bbox)
                    labels.append(_grounded_phrase["cat_name"])
            final_answer = {"bboxes": bboxes, "labels": labels}
        elif task_answer_post_processing_type in ["description_with_polygons", "polygons"]:
            labels = []
            polygons = []
            for result in task_answer:
                label = result["cat_name"]
                _polygons = result["polygons"]
                labels.append(label)
                polygons.append(_polygons)
            final_answer = {"polygons": polygons, "labels": labels}
        elif task_answer_post_processing_type in ["description_with_bboxes_or_polygons"]:
            bboxes = []
            bboxes_labels = []
            polygons = []
            polygons_labels = []
            for result in task_answer:
                label = result["cat_name"]
                if "polygons" in result:
                    _polygons = result["polygons"]
                    polygons.append(_polygons)
                    polygons_labels.append(label)
                else:
                    _bbox = result["bbox"]
                    bboxes.append(_bbox)
                    bboxes_labels.append(label)
            final_answer = {
                "bboxes": bboxes,
                "bboxes_labels": bboxes_labels,
                "polygons": polygons,
                "polygons_labels": polygons_labels,
            }
        else:
            raise ValueError("Unknown task answer post processing type: {}".format(task_answer_post_processing_type))

        final_answer = {task: final_answer}
        return final_answer


class BoxQuantizer(object):
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            quantized_xmin = (xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        quantized_boxes = torch.cat((quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1).int()

        return quantized_boxes

    def dequantize(self, boxes, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_xmin = (xmin + 0.5) * size_per_bin_w
            dequantized_ymin = (ymin + 0.5) * size_per_bin_h
            dequantized_xmax = (xmax + 0.5) * size_per_bin_w
            dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        dequantized_boxes = torch.cat((dequantized_xmin, dequantized_ymin, dequantized_xmax, dequantized_ymax), dim=-1)

        return dequantized_boxes


class CoordinatesQuantizer(object):
    """
    Quantize coornidates (Nx2)
    """

    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, coordinates, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            quantized_x = (x / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_y = (y / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        quantized_coordinates = torch.cat((quantized_x, quantized_y), dim=-1).int()

        return quantized_coordinates

    def dequantize(self, coordinates, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_x = (x + 0.5) * size_per_bin_w
            dequantized_y = (y + 0.5) * size_per_bin_h

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        dequantized_coordinates = torch.cat((dequantized_x, dequantized_y), dim=-1)

        return dequantized_coordinates


class Florence2PostProcesser(object):
    r"""
    Florence-2 post process for converting text prediction to various tasks results.

    Args:
        config: A dict of configs.
        tokenizer: A tokenizer for decoding text to spans.
        sample config:
            UNIFIED_POST_PROCESS:
                # commom configs
                NUM_BBOX_HEIGHT_BINS: 1000
                NUM_BBOX_WIDTH_BINS: 1000
                COORDINATES_HEIGHT_BINS: 1000
                COORDINATES_WIDTH_BINS: 1000
                # task specific configs, override the common configs
                PRASE_TASKS:
                    - TASK_NAME: 'video_dense_caption'
                      PATTERN: 'r<time_(\d+)><time_(\d+)>([a-zA-Z0-9 ]+)'
                      SCORE_MODE: 'avg_cat_name_scores'
                      NUM_BINS: 100
                    - TASK_NAME: 'od'
                      PATTERN: 'r<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>([a-zA-Z0-9 ]+)'
                      SCORE_MODE: 'avg_cat_name_scores'

    Returns:
        parsed_dict (dict): A dict of parsed results.
    """

    def __init__(self, tokenizer=None):
        parse_tasks = []
        parse_task_configs = {}
        config = self._create_default_config()
        for task in config["PARSE_TASKS"]:
            parse_tasks.append(task["TASK_NAME"])
            parse_task_configs[task["TASK_NAME"]] = task

        self.config = config
        self.parse_tasks = parse_tasks
        self.parse_tasks_configs = parse_task_configs

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.all_special_tokens = set(self.tokenizer.all_special_tokens)

        self.init_quantizers()
        self.black_list_of_phrase_grounding = self._create_black_list_of_phrase_grounding()

    def _create_black_list_of_phrase_grounding(self):
        black_list = {}

        if (
            "phrase_grounding" in self.parse_tasks
            and self.parse_tasks_configs["phrase_grounding"]["FILTER_BY_BLACK_LIST"]
        ):
            black_list = {
                "it",
                "I",
                "me",
                "mine",
                "you",
                "your",
                "yours",
                "he",
                "him",
                "his",
                "she",
                "her",
                "hers",
                "they",
                "them",
                "their",
                "theirs",
                "one",
                "oneself",
                "we",
                "us",
                "our",
                "ours",
                "you",
                "your",
                "yours",
                "they",
                "them",
                "their",
                "theirs",
                "mine",
                "yours",
                "his",
                "hers",
                "its",
                "ours",
                "yours",
                "theirs",
                "myself",
                "yourself",
                "himself",
                "herself",
                "itself",
                "ourselves",
                "yourselves",
                "themselves",
                "this",
                "that",
                "these",
                "those",
                "who",
                "whom",
                "whose",
                "which",
                "what",
                "who",
                "whom",
                "whose",
                "which",
                "that",
                "all",
                "another",
                "any",
                "anybody",
                "anyone",
                "anything",
                "each",
                "everybody",
                "everyone",
                "everything",
                "few",
                "many",
                "nobody",
                "none",
                "one",
                "several",
                "some",
                "somebody",
                "someone",
                "something",
                "each other",
                "one another",
                "myself",
                "yourself",
                "himself",
                "herself",
                "itself",
                "ourselves",
                "yourselves",
                "themselves",
                "the image",
                "image",
                "images",
                "the",
                "a",
                "an",
                "a group",
                "other objects",
                "lots",
                "a set",
            }

        return black_list

    def _create_default_config(self):
        config = {
            "NUM_BBOX_HEIGHT_BINS": 1000,
            "NUM_BBOX_WIDTH_BINS": 1000,
            "BOX_QUANTIZATION_MODE": "floor",
            "COORDINATES_HEIGHT_BINS": 1000,
            "COORDINATES_WIDTH_BINS": 1000,
            "COORDINATES_QUANTIZATION_MODE": "floor",
            "PARSE_TASKS": [
                {"TASK_NAME": "od", "PATTERN": r"([a-zA-Z0-9 ]+)<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>"},
                {
                    "TASK_NAME": "ocr",
                    "PATTERN": r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
                    "AREA_THRESHOLD": 0.00,
                },
                {"TASK_NAME": "phrase_grounding", "FILTER_BY_BLACK_LIST": True},
                {
                    "TASK_NAME": "pure_text",
                },
                {
                    "TASK_NAME": "description_with_bboxes",
                },
                {
                    "TASK_NAME": "description_with_polygons",
                },
                {
                    "TASK_NAME": "polygons",
                },
                {
                    "TASK_NAME": "bboxes",
                },
                {
                    "TASK_NAME": "description_with_bboxes_or_polygons",
                },
            ],
        }

        return config

    def init_quantizers(self):
        # we have box_quantizer (od, grounding) and coordinates_quantizer (ocr, referring_segmentation)
        num_bbox_height_bins = self.config.get("NUM_BBOX_HEIGHT_BINS", 1000)
        num_bbox_width_bins = self.config.get("NUM_BBOX_WIDTH_BINS", 1000)
        box_quantization_mode = self.config.get("BOX_QUANTIZATION_MODE", "floor")
        self.box_quantizer = BoxQuantizer(
            box_quantization_mode,
            (num_bbox_width_bins, num_bbox_height_bins),
        )

        num_bbox_height_bins = (
            self.config["COORDINATES_HEIGHT_BINS"]
            if "COORDINATES_HEIGHT_BINS" in self.config
            else self.config.get("NUM_BBOX_HEIGHT_BINS", 1000)
        )
        num_bbox_width_bins = (
            self.config["COORDINATES_WIDTH_BINS"]
            if "COORDINATES_WIDTH_BINS" in self.config
            else self.config.get("NUM_BBOX_WIDTH_BINS", 1000)
        )
        box_quantization_mode = (
            self.config.get("COORDINATES_QUANTIZATION_MODE")
            if "COORDINATES_QUANTIZATION_MODE" in self.config
            else self.config.get("BOX_QUANTIZATION_MODE", "floor")
        )
        self.coordinates_quantizer = CoordinatesQuantizer(
            box_quantization_mode,
            (num_bbox_width_bins, num_bbox_height_bins),
        )

    def parse_od_from_text_and_spans(self, text, pattern, image_size, phrase_centric=False):
        parsed = list(re.finditer(pattern, text))

        instances = []
        for i in range(len(parsed)):
            # Prepare instance.
            instance = {}

            if phrase_centric:
                bbox_bins = [int(parsed[i].group(j)) for j in range(2, 6)]
            else:
                bbox_bins = [int(parsed[i].group(j)) for j in range(1, 5)]
            instance["bbox"] = self.box_quantizer.dequantize(boxes=torch.tensor(bbox_bins), size=image_size).tolist()

            if phrase_centric:
                instance["cat_name"] = parsed[i].group(1).lower().strip()
            else:
                instance["cat_name"] = parsed[i].group(5).lower().strip()
            instances.append(instance)

        return instances

    def parse_ocr_from_text_and_spans(
        self,
        text,
        pattern,
        image_size,
        area_threshold=-1.0,
    ):
        bboxes = []
        labels = []
        text = text.replace("<s>", "")
        # ocr with regions
        parsed = re.findall(pattern, text)
        instances = []
        image_width, image_height = image_size

        for ocr_line in parsed:
            ocr_content = ocr_line[0]
            quad_box = ocr_line[1:]
            quad_box = [int(i) for i in quad_box]
            quad_box = (
                self.coordinates_quantizer.dequantize(torch.tensor(np.array(quad_box).reshape(-1, 2)), size=image_size)
                .reshape(-1)
                .tolist()
            )

            if area_threshold > 0:
                x_coords = list(quad_box[0::2])
                y_coords = list(quad_box[1::2])

                # apply the Shoelace formula
                area = 0.5 * abs(
                    sum(x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i] for i in range(4 - 1))
                )

                if area < (image_width * image_height) * area_threshold:
                    continue

            bboxes.append(quad_box)
            labels.append(ocr_content)
            instances.append(
                {
                    "quad_box": quad_box,
                    "text": ocr_content,
                }
            )
        return instances

    def parse_phrase_grounding_from_text_and_spans(self, text, pattern, image_size):
        # ignore <s> </s> and <pad>
        cur_span = 0
        if text.startswith("<s>"):
            cur_span += 3

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)

        # pattern should be text pattern and od pattern
        pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for pharse_text in phrases:
            phrase_text_strip = pharse_text.replace("<ground>", "", 1)
            phrase_text_strip = pharse_text.replace("<obj>", "", 1)

            if phrase_text_strip == "":
                cur_span += len(pharse_text)
                continue

            # Prepare instance.
            instance = {}

            # parse phrase, get string
            phrase = re.search(pattern, phrase_text_strip)
            if phrase is None:
                cur_span += len(pharse_text)
                continue

            # parse bboxes by box_pattern
            bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
            if len(bboxes_parsed) == 0:
                cur_span += len(pharse_text)
                continue

            phrase = phrase.group()
            # remove leading and trailing spaces
            phrase = phrase.strip()

            if phrase in self.black_list_of_phrase_grounding:
                cur_span += len(pharse_text)
                continue

            # a list of list
            bbox_bins = [[int(_bboxes_parsed.group(j)) for j in range(1, 5)] for _bboxes_parsed in bboxes_parsed]
            instance["bbox"] = self.box_quantizer.dequantize(boxes=torch.tensor(bbox_bins), size=image_size).tolist()

            # exclude non-ascii characters
            phrase = phrase.encode("ascii", errors="ignore").decode("ascii")
            instance["cat_name"] = phrase

            instances.append(instance)

        return instances

    def parse_description_with_bboxes_from_text_and_spans(self, text, pattern, image_size, allow_empty_phrase=False):
        # temporary parse solution, split by '.'
        # ignore <s> </s> and <pad>

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        if allow_empty_phrase:
            pattern = r"(?:(?:<loc_\d+>){4,})"
        else:
            pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)

        # pattern should be text pattern and od pattern
        pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for pharse_text in phrases:
            phrase_text_strip = pharse_text.replace("<ground>", "", 1)
            phrase_text_strip = pharse_text.replace("<obj>", "", 1)

            if phrase_text_strip == "" and not allow_empty_phrase:
                continue

            # parse phrase, get string
            phrase = re.search(pattern, phrase_text_strip)
            if phrase is None:
                continue

            phrase = phrase.group()
            # remove leading and trailing spaces
            phrase = phrase.strip()

            # parse bboxes by box_pattern
            bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
            if len(bboxes_parsed) == 0:
                continue

            # a list of list
            bbox_bins = [[int(_bboxes_parsed.group(j)) for j in range(1, 5)] for _bboxes_parsed in bboxes_parsed]

            bboxes = self.box_quantizer.dequantize(boxes=torch.tensor(bbox_bins), size=image_size).tolist()

            phrase = phrase.encode("ascii", errors="ignore").decode("ascii")
            for _bboxes in bboxes:
                # Prepare instance.
                instance = {}
                instance["bbox"] = _bboxes
                # exclude non-ascii characters
                instance["cat_name"] = phrase
                instances.append(instance)

        return instances

    def parse_description_with_polygons_from_text_and_spans(
        self,
        text,
        pattern,
        image_size,
        allow_empty_phrase=False,
        polygon_sep_token="<sep>",
        polygon_start_token="<poly>",
        polygon_end_token="</poly>",
        with_box_at_start=False,
    ):
        # ref_seg format: '<expression><x1><y1><x2><y2><><><sep><><><><>'
        # ignore <s> </s> and <pad>

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        if allow_empty_phrase:
            pattern = rf"(?:(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        else:
            # [^<]+: This part matches one or more characters that are not the < symbol.
            # The ^ inside the square brackets [] is a negation, meaning it matches anything except <.
            #
            pattern = rf"([^<]+(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        phrases = re.findall(pattern, text)

        phrase_string_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)"
        box_pattern = rf"((?:<loc_\d+>)+)(?:{re.escape(polygon_sep_token)}|$)"

        # one polygons instance is separated by polygon_start_token and polygon_end_token
        polygons_instance_pattern = rf"{re.escape(polygon_start_token)}(.*?){re.escape(polygon_end_token)}"

        instances = []
        for phrase_text in phrases:
            # exclude loc_\d+>
            # need to get span if want to include category score
            phrase_text_strip = re.sub(r"^loc_\d+>", "", phrase_text, count=1)

            # phrase = phrase.replace('<poly>', '')
            # phrase = phrase.replace('poly>', '')

            if phrase_text_strip == "" and not allow_empty_phrase:
                continue

            # parse phrase, get string
            phrase = re.search(phrase_string_pattern, phrase_text_strip)
            if phrase is None:
                continue
            phrase = phrase.group()
            # remove leading and trailing spaces
            phrase = phrase.strip()

            # parse bboxes by box_pattern

            # split by polygon_start_token and polygon_end_token first using polygons_instance_pattern
            if polygon_start_token in phrase_text and polygon_end_token in phrase_text:
                polygons_instances_parsed = list(re.finditer(polygons_instance_pattern, phrase_text))
            else:
                polygons_instances_parsed = [phrase_text]

            for _polygons_instances_parsed in polygons_instances_parsed:
                # Prepare instance.
                instance = {}

                # polygons_parsed= list(re.finditer(box_pattern, phrase_text))
                if isinstance(_polygons_instances_parsed, str):
                    polygons_parsed = list(re.finditer(box_pattern, _polygons_instances_parsed))
                else:
                    polygons_parsed = list(re.finditer(box_pattern, _polygons_instances_parsed.group(1)))
                if len(polygons_parsed) == 0:
                    continue

                # a list of list (polygon)
                bbox = []
                polygons = []
                for _polygon_parsed in polygons_parsed:
                    # group 1: whole <loc_\d+>...</loc_\d+>
                    _polygon = _polygon_parsed.group(1)
                    # parse into list of int
                    _polygon = [int(_loc_parsed.group(1)) for _loc_parsed in re.finditer(r"<loc_(\d+)>", _polygon)]
                    if with_box_at_start and len(bbox) == 0:
                        if len(_polygon) > 4:
                            # no valid bbox prediction
                            bbox = _polygon[:4]
                            _polygon = _polygon[4:]
                        else:
                            bbox = [0, 0, 0, 0]
                    # abandon last element if is not paired
                    if len(_polygon) % 2 == 1:
                        _polygon = _polygon[:-1]

                    # reshape into (n, 2)
                    _polygon = (
                        self.coordinates_quantizer.dequantize(
                            torch.tensor(np.array(_polygon).reshape(-1, 2)), size=image_size
                        )
                        .reshape(-1)
                        .tolist()
                    )
                    # reshape back
                    polygons.append(_polygon)

                instance["cat_name"] = phrase
                instance["polygons"] = polygons
                if len(bbox) != 0:
                    instance["bbox"] = self.box_quantizer.dequantize(
                        boxes=torch.tensor([bbox]), size=image_size
                    ).tolist()[0]

                instances.append(instance)

        return instances

    def __call__(
        self,
        text=None,
        image_size=None,
        parse_tasks=None,
    ):
        """
        Args:
            text: model outputs
            image_size: (width, height)
            parse_tasks: a list of tasks to parse, if None, parse all tasks.

        """
        if parse_tasks is not None:
            if isinstance(parse_tasks, str):
                parse_tasks = [parse_tasks]
            for _parse_task in parse_tasks:
                assert _parse_task in self.parse_tasks, f"parse task {_parse_task} not supported"

        # sequence or text should be provided
        assert text is not None, "text should be provided"

        parsed_dict = {"text": text}

        for task in self.parse_tasks:
            if parse_tasks is not None and task not in parse_tasks:
                continue

            pattern = self.parse_tasks_configs[task].get("PATTERN", None)

            if task == "ocr":
                instances = self.parse_ocr_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                    area_threshold=self.parse_tasks_configs[task].get("AREA_THRESHOLD", 0.0),
                )
                parsed_dict["ocr"] = instances
            elif task == "phrase_grounding":
                instances = self.parse_phrase_grounding_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                )
                parsed_dict["phrase_grounding"] = instances
            elif task == "pure_text":
                parsed_dict["pure_text"] = text
            elif task == "description_with_bboxes":
                instances = self.parse_description_with_bboxes_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                )
                parsed_dict["description_with_bboxes"] = instances
            elif task == "description_with_polygons":
                instances = self.parse_description_with_polygons_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                )
                parsed_dict["description_with_polygons"] = instances
            elif task == "polygons":
                instances = self.parse_description_with_polygons_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                    allow_empty_phrase=True,
                )
                parsed_dict["polygons"] = instances
            elif task == "bboxes":
                instances = self.parse_description_with_bboxes_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                    allow_empty_phrase=True,
                )
                parsed_dict["bboxes"] = instances
            elif task == "description_with_bboxes_or_polygons":
                if "<poly>" in text:
                    # only support either polygons or bboxes, not both at the same time
                    instances = self.parse_description_with_polygons_from_text_and_spans(
                        text,
                        pattern=pattern,
                        image_size=image_size,
                    )
                else:
                    instances = self.parse_description_with_bboxes_from_text_and_spans(
                        text,
                        pattern=pattern,
                        image_size=image_size,
                    )
                parsed_dict["description_with_bboxes_or_polygons"] = instances
            else:
                raise ValueError("task {} is not supported".format(task))

        return parsed_dict
