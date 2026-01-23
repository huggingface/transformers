# MIT License
#
# Copyright 2026 Illuin Technology, and contributors, and The HuggingFace Inc. team.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import AddedToken, BatchEncoding, PreTokenizedInput, TextInput
from ...utils import ModelOutput, auto_docstring, can_return_tuple, is_torch_available, logging
from ..auto.modeling_auto import AutoModel
from ..colpali.modeling_colpali import ColPaliForRetrieval, ColPaliPreTrainedModel
from ..colqwen2.configuration_colqwen2 import ColQwen2Config
from ..colqwen2.processing_colqwen2 import ColQwen2Processor
from ..modernvbert.configuration_modernvbert import ModernVBertConfig, ModernVBertTextConfig, ModernVBertVisionConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class ModernVBertTextConfig(ModernVBertTextConfig):
    pass


class ModernVBertVisionConfig(ModernVBertVisionConfig):
    pass


class ModernVBertConfig(ModernVBertConfig):
    model_type = "modernvbert"
    sub_configs: dict[str, Any] = {"text_config": ModernVBertTextConfig, "vision_config": ModernVBertVisionConfig}


class ColModernVBertConfig(ColQwen2Config):
    r"""
    Configuration class to store the configuration of a [`ColModernVBertForRetrieval`]. It is used to instantiate an instance
    of `ColModernVBertForRetrieval` according to the specified arguments, defining the model architecture following the methodology
    from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Instantiating a configuration with the defaults will yield a similar configuration to the vision encoder used by the pre-trained
    ColModernVBert model, e.g. [ModernVBERT/colmodernvbert-merged](https://huggingface.co/ModernVBERT/colmodernvbert-merged).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vlm_config (`PreTrainedConfig`, *optional*):
            Configuration of the VLM backbone model.
        embedding_dim (`int`, *optional*, defaults to 128):
            Dimension of the multi-vector embeddings produced by the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    Example:

    ```python
    from transformers.models.ColModernVBert import ColModernVBertConfig, ColModernVBertForRetrieval

    config = ColModernVBertConfig()
    model = ColModernVBertForRetrieval(config)
    ```
    """

    model_type = "colmodernvbert"
    default_vlm_config_class = ModernVBertConfig


class ColModernVBertProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
            "add_special_tokens": True,
            "is_split_into_words": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}" + f"<row_{n_h + 1}_col_{n_w + 1}>" + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_img_token
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token
    )


class ColModernVBertProcessor(ColQwen2Processor):
    r"""
    Constructs a ColModernVBert processor which wraps a ModernVBertProcessor and special methods to process images and queries, as
    well as to compute the late-interaction retrieval score.

    [`ColModernVBertProcessor`] offers all the functionalities of [`ModernVBertProcessor`]. See the [`~ModernVBertProcessor.__call__`]
    for more information.

    Args:
            image_processor ([`ModernVBertImageProcessor`]): An instance of [`ModernVBertImageProcessor`]. The image processor is a required input.
            tokenizer (`PreTrainedTokenizerBase`, *optional*): An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
            image_seq_len (`int`, *optional*, defaults to 64): The length of the image sequence i.e. the number of <image> tokens per image in the input.
            visual_prompt_prefix (`Optional`, *optional*): A prefix to be prepended to visual prompts.
            query_prefix (`Optional`, *optional*): A prefix to be prepended to query prompts.
    """

    def __init__(
        self,
        image_processor,
        tokenizer=None,
        chat_template=None,
        image_seq_len: int = 64,
        visual_prompt_prefix: str | None = None,
        query_prefix: str | None = None,
        **kwargs,
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
        visual_prompt_prefix (`str`, *optional*):
            A string that gets tokenized and prepended to the image tokens.
        query_prefix (`str`, *optional*):
            A prefix to be used for the query.
        """
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            visual_prompt_prefix=visual_prompt_prefix,
            query_prefix=query_prefix,
            **kwargs,
        )

        self.chat_template = None  # ColModernVBert does not use chat templates
        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True).content
        self.image_token = AddedToken("<image>", normalized=False, special=True).content
        self.video_token = None  #    ColModernVBert does not process video inputs
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True).content
        self.global_image_tag = "<global-img>"  # https://github.com/huggingface/transformers/pull/32473/files/8063e5e17362571b693f1db95167f5443a3be1b2#r1734825341
        self.image_seq_len = image_seq_len
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(self.fake_image_token)
        self.global_image_token_id = tokenizer.convert_tokens_to_ids(self.global_image_tag)
        self.row_col_ids = [
            tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>") for i in range(6) for j in range(6)
        ]

        # This regex matches one or more occurrences of <global-img> tags (optionally surrounded by newline characters)
        # or <row_x_col_y> tags (where x and y are digits, also optionally surrounded by newline characters).
        self._regex_to_remove_extra_special_tokens = re.compile(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+")

        tokens_to_add = {
            "additional_special_tokens": [
                self.fake_image_token,
                self.image_token,
                self.end_of_utterance_token,
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)

        self.visual_prompt_prefix = visual_prompt_prefix or (
            f"<|begin_of_text|>User:{self.image_token}Describe the image.<end_of_utterance>\nAssistant:"
        )

        self.query_prefix = query_prefix or ""

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.

        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.end_of_utterance_token

    def _extract_images_from_prompts(self, prompts):
        prompt_images = []
        for prompt in prompts:
            images = []
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))
            prompt_images.append(images)
        return prompt_images

    def _process_elements(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        image_seq_len: int | None = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchEncoding:
        """Processes the input prompts and returns a BatchEncoding."""
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            ColModernVBertProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {self.image_token} tokens and {len(images)} images."
                        )
                    # Reorganize the images to match the prompts
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    images = [images]
            elif (
                not isinstance(images, (list, tuple))
                and not isinstance(images[0], (list, tuple))
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            n_images_in_images = [len(sample) for sample in images]

            # Load images if they are URLs
            images = [[load_image(im) if is_url(im) else im for im in sample] for sample in images]

            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                image_rows = inputs.pop("rows", [[0] * n_images for n_images in n_images_in_text])
                image_cols = inputs.pop("cols", [[0] * n_images for n_images in n_images_in_text])

                fake_image_token = self.fake_image_token
                image_token = self.image_token
                global_img_token = self.global_image_tag

                prompt_strings = []
                batch_image_seq_lengths = []
                for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                    # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
                    image_prompt_strings = []
                    image_seq_lengths = []
                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = get_image_prompt_string(
                            n_rows,
                            n_cols,
                            image_seq_len,
                            image_token=image_token,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        # Add +2 and +3 for special BOI/EOI/fake_image_wrapper tokens
                        row_length = (self.image_seq_len + 2) * n_cols + 1
                        image_seq_lengths.append((self.image_seq_len + 3) + row_length * n_rows)
                        image_prompt_strings.append(image_prompt_string)

                    batch_image_seq_lengths.append(image_seq_lengths)
                    split_sample = sample.split(image_token)
                    if len(split_sample) == 0:
                        raise ValueError("The image token should be present in the text.")

                    # Place in the image prompt strings where the image tokens are
                    sample = split_sample[0]
                    for i, image_prompt_string in enumerate(image_prompt_strings):
                        sample += image_prompt_string + split_sample[i + 1]
                    prompt_strings.append(sample)

                text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
                self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])
                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        if return_mm_token_type_ids:
            array_ids = np.array(inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            for i, seq_lengths in enumerate(batch_image_seq_lengths):
                image_start_positions = np.where(array_ids[i] == self.fake_image_token_id)[0]
                j = 0
                for seq_len in seq_lengths:
                    if j >= len(image_start_positions):
                        break
                    start = image_start_positions[j]
                    end = start + seq_len
                    mm_token_type_ids[i, start:end] = 1
                    j = np.searchsorted(image_start_positions, end)

            inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=inputs, tensor_type=return_tensors)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model either (1) one or several texts, either (2) one or several image(s). This method is a custom
        wrapper around the ModernVBertProcessor's [`~ModernVBertProcessor.__call__`] method adapted for the ColModernVBert model. It cannot process
        both text and images at the same time.

        When preparing the the text(s), this method forwards the `text` and `kwargs` arguments to ModernVBertTokenizerFast's
        [`~ModernVBertTokenizerFast.__call__`].
        When preparing the the image(s), this method forwards the `images` and `kwargs` arguments to ModernVBertImageProcessor's
        [`~ModernVBertImageProcessor.__call__`].
        Please refer to the doctsring of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.vlm_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            ColModernVBertProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = suffix is not None

        if text is None and images is None:
            raise ValueError("Either text or images must be provided")
        if text is not None and images is not None:
            raise ValueError("Only one of text or images can be processed at a time")

        if images is not None:
            if is_valid_image(images):
                images = [images]
            elif isinstance(images, list) and is_valid_image(images[0]):
                pass
            elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                raise ValueError("images must be an image, list of images or list of list of images")

            images = [image.convert("RGB") for image in images]

            batch_doc = self._process_elements(
                text=[self.visual_prompt_prefix] * len(images),
                images=images,
                images_kwargs=output_kwargs["images_kwargs"],
                text_kwargs=output_kwargs["text_kwargs"],
            )

            if return_token_type_ids:
                labels = batch_doc["input_ids"].masked_fill(batch_doc["token_type_ids"] == 0, -100)
                batch_doc.update({"labels": labels})

            return batch_doc

        elif text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, list) and isinstance(text[0], str)):
                raise ValueError("Text must be a string or a list of strings")

            if suffix is None:
                suffix = self.query_augmentation_token * 10

            texts_query: list[str] = []

            for query in text:
                augmented_query = self.query_prefix + query + suffix
                texts_query.append(augmented_query)

            batch_query = self._process_elements(
                text=texts_query,
                return_token_type_ids=False,
                text_kwargs=output_kwargs["text_kwargs"],
            )

            return batch_query


@auto_docstring
class ColModernVBertPreTrainedModel(ColPaliPreTrainedModel):
    config: ColModernVBertConfig
    _no_split_modules = ["ModernVBertEncoderLayer", "ModernVBertEncoderLayerVision"]


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for ColModernVBert embeddings output.
    """
)
class ColModernVBertForRetrievalOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        The embeddings of the model.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape
        `(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    image_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True` and `pixel_values` are provided):
        Tuple of `torch.FloatTensor` (one for the output of the image modality projection + one for the output of each layer) of shape
        `(batch_size, num_channels, image_size, image_size)`.
        Hidden-states of the image encoder at the output of each layer plus the initial modality projection outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    loss: torch.FloatTensor | None = None
    embeddings: torch.Tensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    image_hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Following the ColPali approach, ColModernVBert leverages VLMs to construct efficient multi-vector embeddings directly
    from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
    between these document embeddings and the corresponding query embeddings, using the late interaction method
    introduced in ColBERT.

    Using ColModernVBert removes the need for potentially complex and brittle layout recognition and OCR pipelines with
    a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

    ColModernVBert is trained on top of ModernVBert, and was introduced in the following paper:
    [*ModernVBERT: Towards Smaller Visual Document Retrievers*](https://arxiv.org/abs/2510.01149).

    ColModernVBert is part of the ColVision model family, which was introduced with ColPali in the following paper:
    [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://huggingface.co/papers/2407.01449).
    """
)
class ColModernVBertForRetrieval(ColPaliForRetrieval):
    def __init__(self, config: ColModernVBertConfig):
        super().__init__(config)
        self.vlm = AutoModel.from_config(config.vlm_config)
        del self._tied_weights_keys
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> ColModernVBertForRetrievalOutput:
        output_attentions = kwargs.pop("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)

        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        vlm_output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        last_hidden_states = vlm_output[0]  # (batch_size, sequence_length, hidden_size)
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))  # (batch_size, sequence_length, dim)

        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=embeddings.dtype, device=embeddings.device)
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColModernVBertForRetrievalOutput(
            embeddings=embeddings,
            hidden_states=vlm_output.hidden_states,
            attentions=vlm_output.attentions,
            image_hidden_states=vlm_output.image_hidden_states,
        )


__all__ = [
    "ColModernVBertConfig",
    "ColModernVBertForRetrieval",
    "ColModernVBertPreTrainedModel",
    "ColModernVBertProcessor",
]
