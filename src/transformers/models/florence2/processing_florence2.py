# coding=utf-8
# Copyright 2025 Microsoft and the HuggingFace Team. All rights reserved.
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
import math
import re
from typing import Any, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Florence2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_token_type_ids": False},
        "images_kwargs": {},
    }


class Florence2Processor(ProcessorMixin):
    r"""
    Constructs a Florence2 processor which wraps a Florence2 image processor and a Florence2 tokenizer into a single processor.

    [`Florence2Processor`] offers all the functionalities of [`AutoImageProcessor`] and [`BartTokenizerFast`]. See the
    [`~Florence2Processor.__call__`] and [`~Florence2Processor.decode`] for more information.

    Args:
        image_processor (`AutoImageProcessor`, *optional*):
            The image processor is a required input.
        tokenizer (`Union[BartTokenizer, BartTokenizerFast]`, *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
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

        self.post_processor = Florence2PostProcessor(tokenizer=tokenizer)

        super().__init__(image_processor, tokenizer, **kwargs)

    def _construct_prompts(self, text: Union[str, list[str]]) -> list[str]:
        """
        Construct prompts by replacing task tokens with corresponding prompt strings.
        """
        if isinstance(text, str):
            text = [text]

        prompts = []
        for prompt in text:
            # Check for tasks without inputs
            for task_token, task_prompt in self.task_prompts_without_inputs.items():
                if task_token in prompt:
                    if prompt != task_token:
                        raise ValueError(f"Task token {task_token} should be the only content in the prompt.")
                    prompt = task_prompt
                    break
            # Check for tasks with inputs
            for task_token, task_prompt in self.task_prompts_with_input.items():
                if task_token in prompt:
                    input_text = prompt.replace(task_token, "").strip()
                    prompt = task_prompt.format(input=input_text)
                    break
            prompts.append(prompt)
        return prompts

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[Florence2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for the model.

        Args:
            images (`ImageInput`, *optional*):
                The image or batch of images to process.
            text (`str` or `List[str]`, *optional*):
                The text prompts to process.
            return_tensors (`str`, *optional*):
                The tensor type to return ("pt", "np", etc.).

        Returns:
            `BatchFeature`: Encoded inputs.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            Florence2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is None:
            raise ValueError("`images` must be provided.")

        if text is None:
            logger.warning_once("You are using Florence-2 without a text prefix.")
            text = [""] * (1 if not isinstance(images, list) else len(images))
        elif isinstance(text, str):
            text = [text]

        if not isinstance(text, list) or not all(isinstance(t, str) for t in text):
            raise ValueError("`text` must be a string or list of strings.")

        if isinstance(images, list) and len(images) != len(text):
            raise ValueError(f"Number of images ({len(images)}) must match number of texts ({len(text)}).")

        # Process images
        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        pixel_values = image_inputs["pixel_values"]

        # Construct and tokenize prompts
        prompt_strings = self._construct_prompts(text)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)

        return BatchFeature(data={"pixel_values": pixel_values, **text_inputs}, tensor_type=return_tensors)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Bart
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Bart
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def post_process_generation(
        self, text=None, sequence=None, transition_scores=None, task=None, image_size=None
    ) -> dict[str, Any]:
        """
        Post-process generation outputs based on the task.

        Args:
            text (`str`, *optional*):
                Generated text.
            sequence (`Union[List[int], torch.Tensor]`, *optional*):
                Generated token sequence.
            transition_scores (`Union[List[float], torch.Tensor]`, *optional*):
                Transition scores.
            task (`str`, *optional*):
                The task for post-processing.
            image_size (`Tuple[int, int]`, *optional*):
                Image size for dequantization.

        Returns:
            `Dict[str, Any]`: Post-processed results keyed by task.
        """
        if task is None:
            raise ValueError("`task` must be provided for post-processing.")

        post_proc_type = self.tasks_answer_post_processing_type.get(task, "pure_text")
        parsed = self.post_processor(
            text=text,
            sequence=sequence,
            transition_scores=transition_scores,
            image_size=image_size,
            parse_tasks=[post_proc_type],
        )[post_proc_type]

        if post_proc_type == "pure_text":
            final_answer = parsed.replace("<s>", "").replace("</s>", "").strip()
        elif post_proc_type in ["description_with_bboxes", "bboxes"]:
            bboxes = [inst["bbox"] for inst in parsed]
            labels = [inst["cat_name"] for inst in parsed]
            final_answer = {"bboxes": bboxes, "labels": labels}
            if parsed and "score" in parsed[0]:
                final_answer["scores"] = [inst["score"] for inst in parsed]
        elif post_proc_type == "ocr":
            quad_boxes = [inst["quad_box"] for inst in parsed]
            labels = [inst["text"] for inst in parsed]
            final_answer = {"quad_boxes": quad_boxes, "labels": labels}
        elif post_proc_type == "phrase_grounding":
            bboxes = []
            labels = []
            for inst in parsed:
                for bbox in inst["bbox"]:
                    bboxes.append(bbox)
                    labels.append(inst["cat_name"])
            final_answer = {"bboxes": bboxes, "labels": labels}
        elif post_proc_type in ["description_with_polygons", "polygons"]:
            polygons = [inst["polygons"] for inst in parsed]
            labels = [inst["cat_name"] for inst in parsed]
            final_answer = {"polygons": polygons, "labels": labels}
        elif post_proc_type == "description_with_bboxes_or_polygons":
            bboxes = []
            bboxes_labels = []
            polygons = []
            polygons_labels = []
            for inst in parsed:
                label = inst["cat_name"]
                if "polygons" in inst:
                    polygons.append(inst["polygons"])
                    polygons_labels.append(label)
                else:
                    bboxes.append(inst["bbox"])
                    bboxes_labels.append(label)
            final_answer = {
                "bboxes": bboxes,
                "bboxes_labels": bboxes_labels,
                "polygons": polygons,
                "polygons_labels": polygons_labels,
            }
        else:
            raise ValueError(f"Unknown post-processing type: {post_proc_type}")

        return {task: final_answer}


class Quantizer:
    """
    A general quantizer for locations (bounding boxes or coordinates/points).
    Supports quantization modes like 'floor'. Can handle bounding boxes (xmin, ymin, xmax, ymax)
    or arbitrary points/coordinates (Nx2).
    """

    def __init__(self, mode: str = "floor", bins: tuple[int, int] = (1000, 1000)):
        if mode not in ["floor"]:
            raise ValueError(f"Unsupported quantization mode: {mode}. Currently only 'floor' is implemented.")
        self.mode = mode
        self.bins = bins  # (width_bins, height_bins)

    def quantize(self, locations, size: tuple[int, int]):
        """
        Quantize locations.

        Args:
            locations (`torch.Tensor`):
                Tensor of shape (N, 4) for boxes or (N, 2) for points/coordinates.
            size (`tuple[int, int]`):
                Original image size (width, height).

        Returns:
            `torch.Tensor`: Quantized locations as integers.
        """
        bins_w, bins_h = self.bins
        size_w, size_h = size
        per_bin_w = size_w / bins_w
        per_bin_h = size_h / bins_h

        if locations.shape[-1] == 4:  # Bounding boxes: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = locations.split(1, dim=-1)
            q_xmin = (xmin / per_bin_w).floor().clamp(0, bins_w - 1)
            q_ymin = (ymin / per_bin_h).floor().clamp(0, bins_h - 1)
            q_xmax = (xmax / per_bin_w).floor().clamp(0, bins_w - 1)
            q_ymax = (ymax / per_bin_h).floor().clamp(0, bins_h - 1)
            return torch.cat([q_xmin, q_ymin, q_xmax, q_ymax], dim=-1).int()

        elif locations.shape[-1] == 2:  # Points/coordinates: [x, y]
            x, y = locations.split(1, dim=-1)
            q_x = (x / per_bin_w).floor().clamp(0, bins_w - 1)
            q_y = (y / per_bin_h).floor().clamp(0, bins_h - 1)
            return torch.cat([q_x, q_y], dim=-1).int()

        else:
            raise ValueError(f"Unsupported location shape: last dim must be 2 or 4, got {locations.shape[-1]}.")

    def dequantize(self, locations, size: tuple[int, int]):
        """
        Dequantize locations back to original scale.

        Args:
            locations (`torch.Tensor`):
                Quantized tensor of shape (N, 4) for boxes or (N, 2) for points/coordinates.
            size (`tuple[int, int]`):
                Original image size (width, height).

        Returns:
            `torch.Tensor`: Dequantized locations as floats.
        """
        bins_w, bins_h = self.bins
        size_w, size_h = size
        per_bin_w = size_w / bins_w
        per_bin_h = size_h / bins_h

        if locations.shape[-1] == 4:  # Bounding boxes
            xmin, ymin, xmax, ymax = locations.split(1, dim=-1)
            dq_xmin = (xmin + 0.5) * per_bin_w
            dq_ymin = (ymin + 0.5) * per_bin_h
            dq_xmax = (xmax + 0.5) * per_bin_w
            dq_ymax = (ymax + 0.5) * per_bin_h
            return torch.cat([dq_xmin, dq_ymin, dq_xmax, dq_ymax], dim=-1)

        elif locations.shape[-1] == 2:  # Points/coordinates
            x, y = locations.split(1, dim=-1)
            dq_x = (x + 0.5) * per_bin_w
            dq_y = (y + 0.5) * per_bin_h
            return torch.cat([dq_x, dq_y], dim=-1)

        else:
            raise ValueError(f"Unsupported location shape: last dim must be 2 or 4, got {locations.shape[-1]}.")


class Florence2PostProcessor:
    """
    Post-processor for Florence-2 model outputs. Parses generated text into structured results for various tasks
    like object detection, OCR, phrase grounding, etc.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer used for decoding model outputs.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.all_special_tokens = set(self.tokenizer.all_special_tokens)

        self.config = self._create_default_config()
        self.parse_tasks = [task["TASK_NAME"] for task in self.config["PARSE_TASKS"]]
        self.parse_task_configs = {task["TASK_NAME"]: task for task in self.config["PARSE_TASKS"]}

        self.black_list_of_phrase_grounding = self._create_black_list_of_phrase_grounding()
        self.init_quantizers()

    def _create_default_config(self) -> dict[str, Any]:
        return {
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
                {"TASK_NAME": "pure_text"},
                {"TASK_NAME": "description_with_bboxes"},
                {"TASK_NAME": "description_with_polygons"},
                {"TASK_NAME": "polygons"},
                {"TASK_NAME": "bboxes"},
                {"TASK_NAME": "description_with_bboxes_or_polygons"},
            ],
        }

    def _create_black_list_of_phrase_grounding(self) -> set:
        black_list = set()
        if "phrase_grounding" in self.parse_tasks and self.parse_task_configs["phrase_grounding"].get(
            "FILTER_BY_BLACK_LIST", False
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

    def init_quantizers(self):
        num_bbox_height_bins = self.config.get("NUM_BBOX_HEIGHT_BINS", 1000)
        num_bbox_width_bins = self.config.get("NUM_BBOX_WIDTH_BINS", 1000)
        box_quantization_mode = self.config.get("BOX_QUANTIZATION_MODE", "floor")
        self.box_quantizer = Quantizer(mode=box_quantization_mode, bins=(num_bbox_width_bins, num_bbox_height_bins))

        coordinates_height_bins = self.config.get("COORDINATES_HEIGHT_BINS", num_bbox_height_bins)
        coordinates_width_bins = self.config.get("COORDINATES_WIDTH_BINS", num_bbox_width_bins)
        coordinates_quantization_mode = self.config.get("COORDINATES_QUANTIZATION_MODE", box_quantization_mode)
        self.coordinates_quantizer = Quantizer(
            mode=coordinates_quantization_mode, bins=(coordinates_width_bins, coordinates_height_bins)
        )

    def decode_with_spans(self, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        """
        Decode token IDs to text and compute character spans.

        Args:
            token_ids (`list[int]`):
                list of token IDs to decode.

        Returns:
            `tuple[str, list[tuple[int, int]]]`: Decoded text and list of spans (start, end) for each token.
        """
        filtered_tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        text = ""
        spans = []
        for token in filtered_tokens:
            if token in self.all_special_tokens:
                sub_text = token
            else:
                sub_text = self.tokenizer.convert_tokens_to_string([token])
            span = (len(text), len(text) + len(sub_text))
            text += sub_text
            spans.append(span)
        return text, spans

    def parse_od_from_text_and_spans(
        self, text: str, pattern: str, image_size: tuple[int, int], phrase_centric: bool = False
    ) -> list[dict[str, Any]]:
        """
        Parse object detection results from text.

        Args:
            text (`str`):
                The generated text.
            pattern (`str`):
                Regex pattern for matching.
            image_size (`tuple[int, int]`):
                Image size (width, height).
            phrase_centric (`bool`, *optional*, defaults to `False`):
                Whether parsing is phrase-centric.

        Returns:
            `list[dict[str, Any]]`: list of instances with 'bbox' and 'cat_name'.
        """
        matches = list(re.finditer(pattern, text))
        instances = []
        for match in matches:
            if phrase_centric:
                bbox_bins = [int(match.group(j)) for j in range(2, 6)]
                cat_name = match.group(1).lower().strip()
            else:
                bbox_bins = [int(match.group(j)) for j in range(1, 5)]
                cat_name = match.group(5).lower().strip()
            bbox = self.box_quantizer.dequantize(torch.tensor([bbox_bins]), size=image_size)[0].tolist()
            instances.append({"bbox": bbox, "cat_name": cat_name})
        return instances

    def parse_ocr_from_text_and_spans(
        self, text: str, pattern: str, image_size: tuple[int, int], area_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        Parse OCR results with quadrilateral boxes.

        Args:
            text (`str`):
                The generated text.
            pattern (`str`):
                Regex pattern for matching.
            image_size (`tuple[int, int]`):
                Image size (width, height).
            area_threshold (`float`, *optional*, defaults to 0.0):
                Minimum area threshold for filtering boxes.

        Returns:
            `list[dict[str, Any]]`: list of instances with 'quad_box' and 'text'.
        """
        text = text.replace("<s>", "")
        matches = re.findall(pattern, text)
        instances = []
        width, height = image_size
        for content, *quad_str in matches:
            quad_bins = [int(i) for i in quad_str]
            quad_box = (
                self.coordinates_quantizer.dequantize(torch.tensor(quad_bins).reshape(-1, 2), size=image_size)
                .flatten()
                .tolist()
            )

            if area_threshold > 0:
                x_coords = quad_box[0::2]
                y_coords = quad_box[1::2]
                area = 0.5 * abs(
                    sum(x_coords[i] * y_coords[(i + 1) % 4] - x_coords[(i + 1) % 4] * y_coords[i] for i in range(4))
                )
                if area < (width * height) * area_threshold:
                    continue

            instances.append({"quad_box": quad_box, "text": content})
        return instances

    def parse_phrase_grounding_from_text_and_spans(
        self, text: str, image_size: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """
        Parse phrase grounding results.

        Args:
            text (`str`):
                The generated text.
            image_size (`tuple[int, int]`):
                Image size (width, height).

        Returns:
            `list[dict[str, Any]]`: list of instances with 'bbox' and 'cat_name'.
        """
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        phrase_pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(phrase_pattern, text)
        text_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for phrase_text in phrases:
            phrase_text = phrase_text.replace("<ground>", "", 1).replace("<obj>", "", 1)
            if not phrase_text:
                continue
            match = re.search(text_pattern, phrase_text)
            if not match:
                continue
            phrase = match.group().strip()
            if phrase in self.black_list_of_phrase_grounding:
                continue
            boxes_matches = list(re.finditer(box_pattern, phrase_text))
            if not boxes_matches:
                continue
            bbox_bins = [[int(m.group(j)) for j in range(1, 5)] for m in boxes_matches]
            bboxes = self.box_quantizer.dequantize(torch.tensor(bbox_bins), size=image_size).tolist()
            phrase = phrase.encode("ascii", "ignore").decode("ascii")
            instances.append({"bbox": bboxes, "cat_name": phrase})
        return instances

    def _find_matched_token_indices(self, cur_span: tuple[int, int], token_spans: list[tuple[int, int]]) -> list[int]:
        return [i for i, span in enumerate(token_spans) if not (span[1] <= cur_span[0] or span[0] >= cur_span[1])]

    def parse_description_with_bboxes_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        spans: Optional[list[tuple[int, int]]] = None,
        scores: Optional[list[float]] = None,
        score_mode: Optional[str] = None,
        allow_empty_phrase: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Parse descriptions with bounding boxes.

        Args:
            text (`str`):
                The generated text.
            image_size (`tuple[int, int]`):
                Image size (width, height).
            spans (`Optional[list[tuple[int, int]]]`, *optional*):
                Token spans for scoring.
            scores (`Optional[list[float]]`, *optional*):
                Transition scores for scoring.
            score_mode (`Optional[str]`, *optional*):
                Scoring mode ('avg_loc_scores' or 'avg_cat_name_scores').
            allow_empty_phrase (`bool`, *optional*, defaults to `False`):
                Allow phrases without text.

        Returns:
            `list[dict[str, Any]]`: list of instances with 'bbox', 'cat_name', and optional 'score'.
        """
        cur_span = 3 if text.startswith("<s>") else 0
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

        pattern = r"(?:(?:<loc_\d+>){4,})" if allow_empty_phrase else r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)
        text_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for phrase_text in phrases:
            phrase_text = phrase_text.replace("<ground>", "", 1).replace("<obj>", "", 1)
            if not phrase_text and not allow_empty_phrase:
                cur_span += len(phrase_text)
                continue
            match = re.search(text_pattern, phrase_text)
            if not match:
                cur_span += len(phrase_text)
                continue
            phrase_span = match.span()
            phrase = match.group().strip()
            boxes_matches = list(re.finditer(box_pattern, phrase_text))
            if not boxes_matches:
                cur_span += len(phrase_text)
                continue
            bbox_bins = [[int(m.group(j)) for j in range(1, 5)] for m in boxes_matches]
            bboxes = self.box_quantizer.dequantize(torch.tensor(bbox_bins), size=image_size).tolist()

            all_scores = None
            if score_mode == "avg_loc_scores" and spans and scores:
                bbox_spans = [m.span(0) for m in boxes_matches]
                all_scores = []
                for b_span in bbox_spans:
                    token_inds = self._find_matched_token_indices((b_span[0] + cur_span, b_span[1] + cur_span), spans)
                    loc_scores = [scores[i] for i in token_inds]
                    all_scores.append(sum(loc_scores) / len(loc_scores) if loc_scores else 0)
            elif score_mode == "avg_cat_name_scores" and spans and scores:
                token_inds = self._find_matched_token_indices(
                    (phrase_span[0] + cur_span, phrase_span[1] + cur_span), spans
                )
                cat_scores = [scores[i] for i in token_inds]
                score = sum(cat_scores) / len(cat_scores) if cat_scores else 0
                all_scores = [score] * len(bboxes)

            phrase = phrase.encode("ascii", "ignore").decode("ascii")
            for idx, bbox in enumerate(bboxes):
                instance = {"bbox": bbox, "cat_name": phrase}
                if all_scores is not None:
                    instance["score"] = math.exp(all_scores[idx])
                instances.append(instance)

            cur_span += len(phrase_text)
        return instances

    def parse_description_with_polygons_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        allow_empty_phrase: bool = False,
        polygon_sep_token: str = "<sep>",
        polygon_start_token: str = "<poly>",
        polygon_end_token: str = "</poly>",
        with_box_at_start: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Parse descriptions with polygons.

        Args:
            text (`str`):
                The generated text.
            image_size (`tuple[int, int]`):
                Image size (width, height).
            allow_empty_phrase (`bool`, *optional*, defaults to `False`):
                Allow phrases without text.
            polygon_sep_token (`str`, *optional*, defaults to "<sep>"):
                Token separating polygons.
            polygon_start_token (`str`, *optional*, defaults to "<poly>"):
                Start token for polygons.
            polygon_end_token (`str`, *optional*, defaults to "</poly>"):
                End token for polygons.
            with_box_at_start (`bool`, *optional*, defaults to `False`):
                Whether a bounding box is at the start of polygons.

        Returns:
            `list[dict[str, Any]]`: list of instances with 'polygons', 'cat_name', and optional 'bbox'.
        """
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

        pattern = (
            rf"(?:(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
            if allow_empty_phrase
            else rf"([^<]+(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        )
        phrases = re.findall(pattern, text)
        phrase_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)"
        poly_instance_pattern = rf"{re.escape(polygon_start_token)}(.*?){re.escape(polygon_end_token)}"
        box_pattern = rf"((?:<loc_\d+>)+)(?:{re.escape(polygon_sep_token)}|$)"

        instances = []
        for phrase_text in phrases:
            phrase_text = re.sub(r"<loc_\d+>", "", phrase_text, count=1)  # Remove potential leading loc
            if not phrase_text and not allow_empty_phrase:
                continue
            match = re.search(phrase_pattern, phrase_text)
            if not match:
                continue
            phrase = match.group().strip()

            poly_instances = (
                re.findall(poly_instance_pattern, phrase_text)
                if polygon_start_token in phrase_text and polygon_end_token in phrase_text
                else [phrase_text]
            )

            for poly_inst in poly_instances:
                poly_matches = re.finditer(box_pattern, poly_inst)
                if not poly_matches:
                    continue
                bbox = []
                polygons = []
                for poly_match in poly_matches:
                    poly_str = poly_match.group(1)
                    poly_bins = [int(m.group(1)) for m in re.finditer(r"<loc_(\d+)>", poly_str)]
                    if with_box_at_start and not bbox:
                        if len(poly_bins) > 4:
                            bbox = poly_bins[:4]
                            poly_bins = poly_bins[4:]
                        else:
                            bbox = [0, 0, 0, 0]
                    if len(poly_bins) % 2 == 1:
                        poly_bins = poly_bins[:-1]
                    poly_coords = (
                        self.coordinates_quantizer.dequantize(torch.tensor(poly_bins).reshape(-1, 2), size=image_size)
                        .flatten()
                        .tolist()
                    )
                    polygons.append(poly_coords)

                instance = {"cat_name": phrase, "polygons": polygons}
                if bbox:
                    instance["bbox"] = self.box_quantizer.dequantize(torch.tensor([bbox]), size=image_size)[0].tolist()
                instances.append(instance)
        return instances

    def __call__(
        self, text=None, sequence=None, transition_scores=None, image_size=None, parse_tasks=None
    ) -> dict[str, Any]:
        """
        Process model output and parse into task-specific results.

        Args:
            text (`Optional[str]`, *optional*):
                Generated text. Either this or `sequence` must be provided.
            sequence (`Optional[Union[list[int], torch.Tensor]]`, *optional*):
                Token sequence. Either this or `text` must be provided.
            transition_scores (`Optional[Union[list[float], torch.Tensor]]`, *optional*):
                Transition scores for computing instance scores.
            image_size (`Optional[tuple[int, int]]`, *optional*):
                Image size (width, height) required for dequantization.
            parse_tasks (`Optional[Union[str, list[str]]]`, *optional*):
                Specific tasks to parse. If None, parse all supported tasks.

        Returns:
            `dict[str, Any]`: Parsed results for each task, including the raw 'text'.
        """
        if parse_tasks is not None:
            parse_tasks = [parse_tasks] if isinstance(parse_tasks, str) else parse_tasks
            for task in parse_tasks:
                if task not in self.parse_tasks:
                    raise ValueError(f"Unsupported parse task: {task}")

        if (text is None and sequence is None) or (text is not None and sequence is not None):
            raise ValueError("Exactly one of 'text' or 'sequence' must be provided.")

        spans = None
        scores = None
        if sequence is not None:
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.tolist()
            sequence = sequence[1:] if sequence[0] == self.tokenizer.bos_token_id else sequence  # Skip BOS if present
            text, spans = self.decode_with_spans(sequence)
            if transition_scores is not None:
                if isinstance(transition_scores, torch.Tensor):
                    transition_scores = transition_scores.tolist()
                if len(sequence) != len(transition_scores):
                    raise ValueError("Sequence and transition_scores must have the same length.")
                scores = transition_scores

        parsed_dict = {"text": text}

        tasks_to_parse = parse_tasks or self.parse_tasks
        for task in tasks_to_parse:
            config = self.parse_task_configs[task]
            pattern = config.get("PATTERN")

            if task == "ocr":
                parsed_dict["ocr"] = self.parse_ocr_from_text_and_spans(
                    text, pattern=pattern, image_size=image_size, area_threshold=config.get("AREA_THRESHOLD", 0.0)
                )
            elif task == "phrase_grounding":
                parsed_dict["phrase_grounding"] = self.parse_phrase_grounding_from_text_and_spans(
                    text, image_size=image_size
                )
            elif task == "pure_text":
                parsed_dict["pure_text"] = text
            elif task == "description_with_bboxes":
                parsed_dict["description_with_bboxes"] = self.parse_description_with_bboxes_from_text_and_spans(
                    text, image_size=image_size, spans=spans, scores=scores, score_mode=config.get("SCORE_MODE")
                )
            elif task == "description_with_polygons":
                parsed_dict["description_with_polygons"] = self.parse_description_with_polygons_from_text_and_spans(
                    text, image_size=image_size
                )
            elif task == "polygons":
                parsed_dict["polygons"] = self.parse_description_with_polygons_from_text_and_spans(
                    text, image_size=image_size, allow_empty_phrase=True
                )
            elif task == "bboxes":
                parsed_dict["bboxes"] = self.parse_description_with_bboxes_from_text_and_spans(
                    text, image_size=image_size, allow_empty_phrase=True
                )
            elif task == "description_with_bboxes_or_polygons":
                if "<poly>" in text:
                    instances = self.parse_description_with_polygons_from_text_and_spans(text, image_size=image_size)
                else:
                    instances = self.parse_description_with_bboxes_from_text_and_spans(text, image_size=image_size)
                parsed_dict["description_with_bboxes_or_polygons"] = instances
            elif task == "od":
                parsed_dict["od"] = self.parse_od_from_text_and_spans(text, pattern=pattern, image_size=image_size)

        return parsed_dict


__all__ = ["Florence2Processor"]
