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
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import MultiModalData, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_available,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig
from ..bart.modeling_bart import eager_attention_forward
from ..beit.modeling_beit import BeitDropPath
from ..llama4.modeling_llama4 import Llama4VisionMLP
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel, LlavaPreTrainedModel
from ..llava.processing_llava import LlavaProcessorKwargs


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


logger = logging.get_logger(__name__)


class Florence2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2VisionModel`]. It is used to instantiate a Florence2VisionModel
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Florence2VisionModel architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        depths (`Tuple[int]`, *optional*, defaults to `(1, 1, 9, 1)`):
            The depth of the model.
        patch_size (`Tuple[int]`, *optional*, defaults to `(7, 3, 3, 3)`):
            The patch size of the image.
        patch_stride (`Tuple[int]`, *optional*, defaults to `(4, 2, 2, 2)`):
            The patch stride of the image.
        patch_padding (`Tuple[int]`, *optional*, defaults to `(3, 1, 1, 1)`):
            The patch padding of the image.
        patch_prenorm (`Tuple[bool]`, *optional*, defaults to `(False, True, True, True)`):
            Whether to apply layer normalization before the patch embedding layer.
        embed_dim (`Tuple[int]`, *optional*, defaults to `(128, 256, 512, 1024)`):
            The dimension of the embedding layer.
        num_heads (`Tuple[int]`, *optional*, defaults to `(4, 8, 16, 32)`):
            The number of attention heads.
        num_groups (`Tuple[int]`, *optional*, defaults to `(4, 8, 16, 32)`):
            The number of groups.
        window_size (`int`, *optional*, defaults to 12):
            The window size of the model.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout rate of the drop path layer.
        mlp_ratio (`int`, *optional*, defaults to 4.0):
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            If True, add a learnable bias to query, key, value.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        projection_dim (`int`, *optional*, defaults to 1024):
            The dimension of the projection layer.
        max_temporal_embeddings (`int`, *optional*, defaults to 100):
            The configuration of the visual temporal embedding.
        max_position_embeddings (`int`, *optional*, defaults to 50):
            The configuration of the image position embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    Example:

    ```python
    >>> from transformers import Florence2VisionConfig, Florence2VisionModel

    >>> # Initializing a Florence2 Vision style configuration
    >>> configuration = Florence2VisionConfig()

    >>> # Initializing a model (with random weights)
    >>> model = Florence2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence_vision"

    def __init__(
        self,
        in_channels=3,
        depths=(1, 1, 9, 1),
        patch_size=(7, 3, 3, 3),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 1, 1, 1),
        patch_prenorm=(False, True, True, True),
        embed_dim=(128, 256, 512, 1024),
        num_heads=(4, 8, 16, 32),
        num_groups=(4, 8, 16, 32),
        window_size=12,
        drop_path_rate=0.1,
        mlp_ratio=4.0,
        qkv_bias=True,
        activation_function="gelu",
        projection_dim=1024,
        max_temporal_embeddings=100,
        max_position_embeddings=50,
        initializer_range=0.02,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.depths = list(depths)
        self.patch_size = list(patch_size)
        self.patch_stride = list(patch_stride)
        self.patch_padding = list(patch_padding)
        self.patch_prenorm = list(patch_prenorm)
        self.embed_dim = list(embed_dim)
        self.num_heads = list(num_heads)
        self.num_groups = list(num_groups)
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.projection_dim = projection_dim
        self.max_temporal_embeddings = max_temporal_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.activation_function = activation_function

        super().__init__(**kwargs)


class Florence2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2ForConditionalGeneration`]. It is used to instantiate an
    Florence-2 model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the Florence-2
    [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AutoConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Florence2VisionConfig`].
        image_token_id (`int`, *optional*, defaults to 51289):
            The image token index to encode the image prompt.
        is_encoder_decoder (bool, optional, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.

    Example:

    ```python
    >>> from transformers import Florence2ForConditionalGeneration, Florence2Config, CLIPVisionConfig, BartConfig

    >>> # Initializing a clip-like vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Bart config
    >>> text_config = BartConfig()

    >>> # Initializing a Florence-2 configuration
    >>> configuration = Florence2Config(vision_config, text_config)

    >>> # Initializing a model from the florence-2 configuration
    >>> model = Florence2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence2"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": Florence2VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=51289,
        is_encoder_decoder=True,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "bart")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["bart"]()

        if isinstance(vision_config, dict):
            vision_config = Florence2VisionConfig(**vision_config)
        elif vision_config is None:
            logger.info("vision_config is None. Initializing the Florence2VisionConfig with default values.")
            vision_config = Florence2VisionConfig()

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class Florence2ProcessorKwargs(LlavaProcessorKwargs):
    pass


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
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
        post_processor_config (`dict`,  *optional*, defaults to 0):
            Task-specific parsing rules for [`Florence2PostProcessor`], e.g. regex patterns,
            thresholds, or banned tokens.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        num_additional_image_tokens: int = 0,
        post_processor_config: Optional[dict] = None,
        **kwargs,
    ):
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

        self.num_image_tokens = image_processor.image_seq_length
        self.num_additional_image_tokens = num_additional_image_tokens
        self.post_processor_config = post_processor_config
        self.post_processor = Florence2PostProcessor(config=post_processor_config, tokenizer=tokenizer)
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id

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
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to BartTokenizerFast's [`~BartTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            Florence2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

        if text is None:
            logger.warning_once("You are using Florence-2 without a text prefix.")
            text = [""] * (1 if not isinstance(images, list) else len(images))
        elif isinstance(text, str):
            text = [text]

        if not isinstance(text, list) or not all(isinstance(token, str) for token in text):
            raise ValueError("`text` must be a string or list of strings.")

        if isinstance(images, list) and len(images) != len(text):
            raise ValueError(f"Number of images ({len(images)}) must match number of texts ({len(text)}).")

        prompt_strings = self._construct_prompts(text)

        # Add image tokens and special tokens if images are provided
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            expanded_image_prompts = []
            for sample in prompt_strings:
                sample = (
                    self.image_token * self.num_image_tokens
                    + self.tokenizer.bos_token
                    + sample
                    + self.tokenizer.eos_token
                )
                expanded_image_prompts.append(sample)
            prompt_strings = expanded_image_prompts

        # Construct and tokenize prompts
        output_kwargs["text_kwargs"].pop("add_special_tokens", None)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(
            prompt_strings, **output_kwargs["text_kwargs"], add_special_tokens=False, return_tensors=None
        )
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**image_inputs, **text_inputs}, tensor_type=return_tensors)

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

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=False, **kwargs):
        """
        Post-processes the output of `FuyuForConditionalGeneration` to only return the text output.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                containing the token ids of the generated sequences.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text output.
        """
        return self.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)

    def post_process_generation(self, text=None, sequence=None, task=None, image_size=None) -> dict[str, Any]:
        """
        Post-process generation outputs based on the task.

        Args:
            text (`str`, *optional*):
                Generated text.
            sequence (`Union[List[int], torch.Tensor]`, *optional*):
                Generated token sequence.
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


class Florence2PostProcessor:
    """
    Post-processor for Florence-2 model outputs. Parses generated text into structured results for various tasks
    like object detection, OCR, phrase grounding, etc.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer used for decoding model outputs.
    """

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.parse_task_config = config or {}
        self.banned_grounding_tokens = set(
            self.parse_task_config.get("phrase_grounding", {}).get("banned_grounding_tokens", [])
        )
        self.all_special_tokens = set(self.tokenizer.all_special_tokens)
        self.quantize_bins = (1000, 1000)

    def quantize(self, locations: "torch.Tensor", size: tuple[int, int]) -> "torch.Tensor":
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
        bins_w, bins_h = self.quantize_bins
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

    def dequantize(self, locations: "torch.Tensor", size: tuple[int, int]) -> "torch.Tensor":
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
        bins_w, bins_h = self.quantize_bins
        size_w, size_h = size
        per_bin_w = size_w / bins_w
        per_bin_h = size_h / bins_h

        # Add 0.5 to use the center position of the bin as the coordinate.
        if locations.shape[-1] == 4:  # Bounding boxes
            xmin, ymin, xmax, ymax = locations.split(1, dim=-1)
            dq_xmin = (xmin + 0.5) * per_bin_w
            dq_ymin = (ymin + 0.5) * per_bin_h
            dq_xmax = (xmax + 0.5) * per_bin_w
            dq_ymax = (ymax + 0.5) * per_bin_h
            return torch.cat([dq_xmin, dq_ymin, dq_xmax, dq_ymax], dim=-1).int()

        elif locations.shape[-1] == 2:  # Points/coordinates
            x, y = locations.split(1, dim=-1)
            dq_x = (x + 0.5) * per_bin_w
            dq_y = (y + 0.5) * per_bin_h
            return torch.cat([dq_x, dq_y], dim=-1).int()

        else:
            raise ValueError(f"Unsupported location shape: last dim must be 2 or 4, got {locations.shape[-1]}.")

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

    def parse_ocr_from_text_and_spans(
        self, text: str, pattern: Optional[str], image_size: tuple[int, int], area_threshold: float = 0.0
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
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        if pattern is None:
            pattern = r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        matches = re.findall(pattern, text)
        instances = []
        width, height = image_size

        for content, *quad_str in matches:
            quad_bins = [int(i) for i in quad_str]
            quad_box = self.dequantize(torch.tensor(quad_bins).reshape(-1, 2), size=image_size).flatten().tolist()

            if area_threshold > 0:
                x_coords = quad_box[0::2]
                y_coords = quad_box[1::2]
                # Apply the Shoelace formula
                area = 0.5 * abs(
                    sum(x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i] for i in range(4 - 1))
                )

                if area < (width * height) * area_threshold:
                    continue

            instances.append({"quad_box": quad_box, "text": content.strip()})
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
            if phrase in self.banned_grounding_tokens:
                continue
            boxes_matches = list(re.finditer(box_pattern, phrase_text))
            if not boxes_matches:
                continue
            bbox_bins = [[int(m.group(j)) for j in range(1, 5)] for m in boxes_matches]
            bboxes = self.dequantize(torch.tensor(bbox_bins), size=image_size).tolist()
            phrase = phrase.encode("ascii", "ignore").decode("ascii")
            instances.append({"bbox": bboxes, "cat_name": phrase})
        return instances

    def _find_matched_token_indices(self, cur_span: tuple[int, int], token_spans: list[tuple[int, int]]) -> list[int]:
        return [i for i, span in enumerate(token_spans) if not (span[1] <= cur_span[0] or span[0] >= cur_span[1])]

    def parse_description_with_bboxes_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        allow_empty_phrase: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Parse descriptions with bounding boxes.

        Args:
            text (`str`):
                The generated text.
            image_size (`tuple[int, int]`):
                Image size (width, height).
            allow_empty_phrase (`bool`, *optional*, defaults to `False`):
                Allow phrases without text.

        Returns:
            `list[dict[str, Any]]`: list of instances with 'bbox', 'cat_name', and optional 'score'.
        """
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

        if allow_empty_phrase:
            pattern = r"(?:(?:<loc_\d+>){4,})"
        else:
            pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)

        text_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for phrase_text in phrases:
            phrase_text = phrase_text.replace("<ground>", "", 1).replace("<obj>", "", 1)
            if not phrase_text and not allow_empty_phrase:
                continue
            match = re.search(text_pattern, phrase_text)
            if not match:
                continue
            phrase = match.group().strip()
            boxes_matches = list(re.finditer(box_pattern, phrase_text))
            if not boxes_matches:
                continue
            bbox_bins = [[int(m.group(j)) for j in range(1, 5)] for m in boxes_matches]
            bboxes = self.dequantize(torch.tensor(bbox_bins), size=image_size).tolist()

            phrase = phrase.encode("ascii", "ignore").decode("ascii")
            for bbox in bboxes:
                instance = {"bbox": bbox, "cat_name": phrase}
                instances.append(instance)

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

        if allow_empty_phrase:
            pattern = rf"(?:(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        else:
            pattern = rf"([^<]+(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        phrases = re.findall(pattern, text)
        phrase_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)"
        poly_instance_pattern = rf"{re.escape(polygon_start_token)}(.*?){re.escape(polygon_end_token)}"
        box_pattern = rf"((?:<loc_\d+>)+)(?:{re.escape(polygon_sep_token)}|$)"

        instances = []
        for phrase_text in phrases:
            phrase_text_strip = re.sub(r"^<loc_\d+>", "", phrase_text, count=1)
            if not phrase_text_strip and not allow_empty_phrase:
                continue
            match = re.search(phrase_pattern, phrase_text_strip)
            if not match:
                continue
            phrase = match.group().strip()

            if polygon_start_token in phrase_text and polygon_end_token in phrase_text:
                poly_instances = [m.group(1) for m in re.finditer(poly_instance_pattern, phrase_text)]
            else:
                poly_instances = [phrase_text]

            for poly_inst in poly_instances:
                poly_matches = list(re.finditer(box_pattern, poly_inst))
                if len(poly_matches) == 0:
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
                        self.dequantize(torch.tensor(poly_bins).reshape(-1, 2), size=image_size).flatten().tolist()
                    )
                    polygons.append(poly_coords)

                instance = {"cat_name": phrase, "polygons": polygons}
                if bbox:
                    instance["bbox"] = self.dequantize(torch.tensor([bbox]), size=image_size)[0].tolist()
                instances.append(instance)
        return instances

    def __call__(self, text=None, sequence=None, image_size=None, parse_tasks=None) -> dict[str, Any]:
        """
        Process model output and parse into task-specific results.

        Args:
            text (`Optional[str]`, *optional*):
                Generated text. Either this or `sequence` must be provided.
            sequence (`Optional[Union[list[int], torch.Tensor]]`, *optional*):
                Token sequence. Either this or `text` must be provided.
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
                if task not in self.parse_task_config.keys():
                    raise ValueError(f"Unsupported parse task: {task}")

        if (text is None and sequence is None) or (text is not None and sequence is not None):
            raise ValueError("Exactly one of 'text' or 'sequence' must be provided.")

        if sequence is not None:
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.tolist()
            sequence = sequence[1:] if sequence[0] == self.tokenizer.bos_token_id else sequence  # Skip BOS if present
            text, _ = self.decode_with_spans(sequence)

        parsed_dict = {"text": text}

        tasks_to_parse = parse_tasks or self.parse_task_config.keys()
        for task in tasks_to_parse:
            config = self.parse_task_config[task]
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
                    text, image_size=image_size
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
            else:
                raise ValueError("task {} is not supported".format(task))

        return parsed_dict


class Florence2VisionDropPath(BeitDropPath):
    pass


class Florence2VisionLearnedAbsolutePositionEmbedding2D(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, config: Florence2Config):
        super().__init__()
        num_pos = config.vision_config.max_position_embeddings
        embedding_dim = config.vision_config.embed_dim[-1]
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - (embedding_dim // 2))

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    """
    This module generates 1D cosine positional embeddings using precomputed sinusoidal functions.
    """

    def __init__(self, config: Florence2Config):
        super().__init__()
        self.embed_dim = config.vision_config.embed_dim[-1]
        self.max_seq_len = config.vision_config.max_temporal_embeddings
        pos_idx_to_embed = torch.empty((self.max_seq_len, self.embed_dim))
        sine, cosine = self.get_sinusoid_embeddings(
            max_positions=self.max_seq_len,
            embed_dim=self.embed_dim,
        )
        pos_idx_to_embed[:, 0::2] = sine
        pos_idx_to_embed[:, 1::2] = cosine
        # Save the positional embeddings in a constant buffer.
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embed_dim: int):
        half_dim = embed_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        len_seq = seq_embeds.size(1)
        if len_seq > self.max_seq_len:
            raise ValueError(f"Maximum sequence length {self.max_seq_len}, got {len_seq}")
        pos_embeds = self.pos_idx_to_embed[0:len_seq, :]
        return pos_embeds


class Florence2VisionMLP(Llama4VisionMLP):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int):
        super().__init__(config)
        self.fc1 = nn.Linear(config.embed_dim[stage_idx], int(config.embed_dim[stage_idx] * config.mlp_ratio))
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc2 = nn.Linear(int(config.embed_dim[stage_idx] * config.mlp_ratio), config.embed_dim[stage_idx])


class Florence2VisionConvEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config: Florence2VisionConfig, stage_idx: int):
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        self.patch_size = config.patch_size[stage_idx]
        self.in_channels = config.in_channels if stage_idx == 0 else config.embed_dim[stage_idx - 1]
        self.embed_dim = config.embed_dim[stage_idx]
        self.stride = config.patch_stride[stage_idx]
        self.padding = config.patch_padding[stage_idx]
        self.pre_norm = config.patch_prenorm[stage_idx]

        self.conv = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
        )

        dim_norm = self.in_channels if self.pre_norm else self.embed_dim
        self.norm = nn.LayerNorm(dim_norm)

    def forward(self, hidden_states: torch.Tensor):
        if self.norm and self.pre_norm:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)

        hidden_states = self.conv(hidden_states)

        if self.norm and not self.pre_norm:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)
        return hidden_states


class Florence2VisionChannelAttention(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int):
        super().__init__()
        self.config = config
        self.dim = config.embed_dim[stage_idx]
        self.groups = config.num_groups[stage_idx]
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.is_causal = False

    def forward(self, hidden_states: torch.Tensor):
        batch_size, num_tokens, hidden_size = hidden_states.shape

        # Reshape for grouped channel attention
        qkv = self.qkv(hidden_states).reshape(batch_size, num_tokens, 3, self.groups, hidden_size // self.groups)
        qkv = qkv.permute(2, 0, 3, 4, 1)
        query, key, value = qkv.unbind(0)

        scale = num_tokens**-0.5
        # Channel-to-channel attention within groups:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        hidden_states, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            scaling=scale,
        )
        hidden_states = hidden_states.permute(0, 3, 2, 1)
        hidden_states = hidden_states.reshape(batch_size, num_tokens, hidden_size)

        # Final projection
        hidden_states = self.proj(hidden_states)
        return hidden_states


class Florence2VisionChannelBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        drop_path_rate: float,
    ):
        super().__init__()

        self.config = config
        dim_in = config.embed_dim[stage_idx]

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=3,
            padding=1,
            groups=dim_in,
        )
        self.norm1 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.channel_attn = Florence2VisionChannelAttention(config=config, stage_idx=stage_idx)
        self.drop_path1 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=3,
            padding=1,
            groups=dim_in,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.ffn = Florence2VisionMLP(config=config, stage_idx=stage_idx)
        self.drop_path2 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, embed_dim, height, width = hidden_states.shape

        # First channel block: Depthwise Conv + Channel Attention
        hidden_states = self.conv1(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # Channel group attention self-attention mechanism
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.channel_attn(hidden_states)
        hidden_states = residual + self.drop_path1(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        # Second channel block: Depthwise Conv + FFN
        hidden_states = self.conv2(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # FFN
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.drop_path2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        return hidden_states


class Florence2VisionWindowAttention(nn.Module):
    def __init__(self, config: Florence2VisionConfig, stage_idx: int):
        super().__init__()
        self.config = config
        self.dim = config.embed_dim[stage_idx]
        self.window_size = config.window_size
        self.num_heads = config.num_heads[stage_idx]
        head_dim = self.dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.is_causal = False

    def forward(self, hidden_states: torch.Tensor):
        batch_size, height, width, embed_dim = hidden_states.shape

        # Pad the input if necessary
        pad_left = pad_top = 0
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        hidden_states = F.pad(hidden_states, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, padded_height, padded_width, _ = hidden_states.shape

        # Partition input into non-overlapping windows (for local spatial attention in DaViT)
        hidden_states = hidden_states.view(
            batch_size,
            padded_height // self.window_size,
            self.window_size,
            padded_width // self.window_size,
            self.window_size,
            embed_dim,
        )
        windowed_hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous()
        windowed_hidden_states = windowed_hidden_states.view(-1, self.window_size * self.window_size, embed_dim)

        # Generate Q, K, V for each window
        num_windows_per_batch, num_tokens_per_window, embed_dim = windowed_hidden_states.shape
        qkv = self.qkv(windowed_hidden_states).reshape(
            num_windows_per_batch, num_tokens_per_window, 3, self.num_heads, embed_dim // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        windowed_hidden_states, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            scaling=self.scale,
        )
        windowed_hidden_states = windowed_hidden_states.view(num_windows_per_batch, num_tokens_per_window, embed_dim)
        windowed_hidden_states = self.proj(windowed_hidden_states)

        # Merge windows back to original spatial layout
        windowed_hidden_states = windowed_hidden_states.view(-1, self.window_size, self.window_size, embed_dim)
        hidden_states = windowed_hidden_states.view(
            -1,
            padded_height // self.window_size,
            padded_width // self.window_size,
            self.window_size,
            self.window_size,
            embed_dim,
        )
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous()
        hidden_states = hidden_states.view(-1, padded_height, padded_width, embed_dim)
        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        hidden_states = hidden_states.view(batch_size, height * width, embed_dim)

        return hidden_states


class Florence2VisionSpatialBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        drop_path_rate: float,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            config.embed_dim[stage_idx],
            config.embed_dim[stage_idx],
            kernel_size=3,
            padding=1,
            groups=config.embed_dim[stage_idx],
        )
        self.norm1 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.window_attn = Florence2VisionWindowAttention(config=config, stage_idx=stage_idx)
        self.drop_path1 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            config.embed_dim[stage_idx],
            config.embed_dim[stage_idx],
            kernel_size=3,
            padding=1,
            groups=config.embed_dim[stage_idx],
        )
        self.norm2 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.ffn = Florence2VisionMLP(config=config, stage_idx=stage_idx)
        self.drop_path2 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, embed_dim, height, width = hidden_states.shape

        # First spatial mixing block: Conv + Window Attention
        hidden_states = self.conv1(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # Spatial Window-based self-attention mechanism
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, embed_dim)
        hidden_states = self.window_attn(hidden_states)
        hidden_states = residual + self.drop_path1(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        # Second spatial mixing block: Conv + FFN
        hidden_states = self.conv2(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # FFN
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.drop_path2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        return hidden_states


class Florence2VisionBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        spatial_drop_path_rate: float,
        channel_drop_path_rate: float,
    ):
        super().__init__()
        self.spatial_block = Florence2VisionSpatialBlock(
            config=config,
            stage_idx=stage_idx,
            drop_path_rate=spatial_drop_path_rate,
        )
        self.channel_block = Florence2VisionChannelBlock(
            config=config,
            stage_idx=stage_idx,
            drop_path_rate=channel_drop_path_rate,
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.spatial_block(hidden_states)
        hidden_states = self.channel_block(hidden_states)
        return hidden_states


@auto_docstring
class Florence2VisionPreTrainedModel(PreTrainedModel):
    config_class = Florence2VisionConfig
    main_input_name = "pixel_values"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True


@auto_docstring
class Florence2VisionBackbone(Florence2VisionPreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        self.config = config

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.num_stages = len(self.embed_dim)

        if not (self.num_stages == len(self.num_heads) == len(self.num_groups)):
            raise ValueError(
                f"Expected self.num_stages ({self.num_stages}) == "
                f"len(self.num_heads) ({len(self.num_heads)}) == "
                f"len(self.num_groups) ({len(self.num_groups)})"
            )

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths) * 2, device="cpu")]
        depth_offset = 0

        convs = []
        blocks = []
        for stage_idx in range(self.num_stages):
            conv_embed = Florence2VisionConvEmbed(
                config=config,
                stage_idx=stage_idx,
            )
            convs.append(conv_embed)

            block = nn.ModuleList(
                Florence2VisionBlock(
                    config=config,
                    stage_idx=stage_idx,
                    spatial_drop_path_rate=dpr[depth_offset + block_idx * 2],
                    channel_drop_path_rate=dpr[depth_offset + block_idx * 2 + 1],
                )
                for block_idx in range(config.depths[stage_idx])
            )
            blocks.append(block)
            depth_offset += config.depths[stage_idx] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, hidden_states: torch.Tensor):
        for conv, block in zip(self.convs, self.blocks):
            hidden_states = conv(hidden_states)
            for layer in block:
                hidden_states = layer(hidden_states)
        return hidden_states


class Florence2MultiModalProjector(nn.Module):
    def __init__(self, config: Florence2Config):
        super().__init__()
        self.vision_embedding_dim = config.vision_config.embed_dim[-1]
        self.vision_projection_dim = config.vision_config.projection_dim
        self.image_projection = nn.Linear(self.vision_embedding_dim, self.vision_projection_dim, bias=False)
        self.image_proj_norm = nn.LayerNorm(self.vision_projection_dim)
        self.image_position_embed = Florence2VisionLearnedAbsolutePositionEmbedding2D(config=config)
        self.visual_temporal_embed = Florence2VisionPositionalEmbeddingCosine1D(config=config)

    def forward(self, image_features):
        position_features = image_features + self.image_position_embed(image_features)
        position_features = position_features.flatten(2).transpose(1, 2)
        temporal_features = self.visual_temporal_embed(position_features[:, :1, :])
        temporal_features = temporal_features.unsqueeze(1)
        visual_token_features = position_features + temporal_features
        visual_token_features = visual_token_features.unsqueeze(1)
        spatial_image_features = visual_token_features.mean(dim=2)
        temporal_image_features = visual_token_features.mean(dim=1)
        image_features = torch.cat([spatial_image_features, temporal_image_features], dim=1)
        image_features = self.image_projection(image_features)
        image_features = self.image_proj_norm(image_features)
        return image_features


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Florence-2 base model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    """
)
class Florence2Seq2SeqModelOutput(Seq2SeqModelOutput):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Florence-2 model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    """
)
class Florence2Seq2SeqLMOutput(Seq2SeqLMOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


@auto_docstring
class Florence2PreTrainedModel(LlavaPreTrainedModel):
    config_class = Florence2Config

    _supports_attention_backend = False


@auto_docstring(
    custom_intro="""
    Florence-2 is a vision model for captioning, detection, and segmentation.
    """
)
class Florence2Model(LlavaModel):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = [
        "language_model.encoder.embed_tokens.weight",
        "language_model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config: Florence2Config):
        super().__init__(config)
        self.vision_tower = Florence2VisionBackbone(config=config.vision_config)

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_features = self.vision_tower(pixel_values, **kwargs)
        image_embeds = self.multi_modal_projector(image_features)
        return image_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Florence2Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values)
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                special_image_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            encoder_outputs = self.language_model.encoder(
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if decoder_input_ids is None:
            decoder_start_token_id = self.config.text_config.decoder_start_token_id
            decoder_input_ids = torch.ones((inputs_embeds.size()[0], 1), dtype=torch.long, device=inputs_embeds.device)
            decoder_input_ids *= decoder_start_token_id

        decoder_outputs = self.language_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        return Florence2Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@auto_docstring(
    custom_intro="""
    Florence-2 is a vision model for captioning, detection, and segmentation.
    """
)
class Florence2ForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = [
        "model.language_model.encoder.embed_tokens.weight",
        "model.language_model.decoder.embed_tokens.weight",
        "lm_head.weight",
    ]

    def get_encoder(self):
        return self.model.get_encoder()

    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):
        return self.model.get_image_features(pixel_values=pixel_values, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Florence2Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Florence2ForConditionalGeneration

        >>> model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large")
        >>> processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

        >>> prompt = "<CAPTION>"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=100)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A green car parked in front of a yellow building."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Florence2Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        return self.model.get_placeholder_mask(
            input_ids=input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config,
    ) -> dict[str, Any]:
        # override to handle merging image and text embeddings before passing to language encoder
        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        pixel_values = model_kwargs.pop("pixel_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(inputs_tensor)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                inputs_tensor, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        model_kwargs["inputs_embeds"] = inputs_embeds
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            None, model_kwargs, model_input_name, generation_config
        )
        model_kwargs.pop("inputs_embeds", None)
        return model_kwargs


__all__ = [
    "Florence2Config",
    "Florence2Processor",
    "Florence2VisionConfig",
    "Florence2Model",
    "Florence2ForConditionalGeneration",
    "Florence2PreTrainedModel",
    "Florence2VisionBackbone",
    "Florence2VisionPreTrainedModel",
]
