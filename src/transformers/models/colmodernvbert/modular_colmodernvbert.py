# Copyright 2026 Illuin Technology and contributors, and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..auto import CONFIG_MAPPING
from ..auto.modeling_auto import AutoModel
from ..colpali.modeling_colpali import ColPaliForRetrieval, ColPaliPreTrainedModel
from ..colqwen2.configuration_colqwen2 import ColQwen2Config
from ..idefics3.processing_idefics3 import Idefics3Processor, Idefics3ProcessorKwargs


logger = logging.get_logger(__name__)


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
    from transformers import ColModernVBertConfig, ColModernVBertForRetrieval

    config = ColModernVBertConfig()
    model = ColModernVBertForRetrieval(config)
    ```
    """

    model_type = "colmodernvbert"
    sub_configs: dict[str, Any] = {"vlm_config": PreTrainedConfig}

    def __init__(
        self,
        vlm_config=None,
        embedding_dim: int = 128,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if vlm_config is None:
            vlm_config = CONFIG_MAPPING["modernvbert"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `ModernVBertConfig` with default values."
            )
        elif isinstance(vlm_config, dict):
            vlm_config = deepcopy(vlm_config)
            if "model_type" not in vlm_config:
                raise KeyError(
                    "The `model_type` key is missing in the `vlm_config` dictionary. Please provide the model type."
                )
            vlm_config = CONFIG_MAPPING[vlm_config["model_type"]](**vlm_config)
        elif not isinstance(vlm_config, PreTrainedConfig):
            raise TypeError(
                f"Invalid type for `vlm_config`. Expected `PreTrainedConfig`, `dict`, or `None`, but got {type(vlm_config)}."
            )

        if not hasattr(vlm_config, "vocab_size"):
            vlm_config.vocab_size = vlm_config.get_text_config().vocab_size

        self.vlm_config = vlm_config
        self.embedding_dim = embedding_dim
        self.initializer_range = initializer_range
        PreTrainedConfig.__init__(**kwargs)


class ColModernVBertProcessorKwargs(Idefics3ProcessorKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
        },
        "images_kwargs": {
            "return_row_col_info": True,
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ColModernVBertProcessor(Idefics3Processor):
    r"""
    Constructs a ColModernVBert processor which wraps a ModernVBertProcessor and special methods to process images and queries, as
    well as to compute the late-interaction retrieval score.

    [`ColModernVBertProcessor`] offers all the functionalities of [`ModernVBertProcessor`]. See the [`~ModernVBertProcessor.__call__`]
    for more information.

    Args:
            image_processor ([`Idefics3ImageProcessor`]): An instance of [`Idefics3ImageProcessor`]. The image processor is a required input.
            tokenizer (`PreTrainedTokenizerFast`, *optional*): An instance of [`PreTrainedTokenizerFast`]. This should correspond with the model's text model. The tokenizer is a required input.
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
        chat_template = None  # ColModernVBert does not use chat templates

        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            image_seq_len=image_seq_len,
            **kwargs,
        )

        self.visual_prompt_prefix = visual_prompt_prefix or (
            f"<|begin_of_text|>User:{self.image_token}Describe the image.<end_of_utterance>\nAssistant:"
        )
        self.query_prefix = query_prefix or ""
        self.query_augmentation_token = self.end_of_utterance_token

    def process_images(
        self,
        images: ImageInput | None = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several image(s). Handles input validation, RGB conversion,
        and prepends the `visual_prompt_prefix` to each image. Optionally computes labels from
        `token_type_ids` when a `suffix` is provided in `text_kwargs`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
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

        # Normalize input to a flat list of images
        if is_valid_image(images):
            images = [images]
        elif isinstance(images, list) and is_valid_image(images[0]):
            pass
        elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
            raise ValueError("images must be an image, list of images or list of list of images")

        # Ensure all images are in RGB format
        images = [image.convert("RGB") for image in images]

        # Pair each image with the visual prompt prefix for the VLM backbone
        batch_doc = self.__call__(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            images_kwargs=output_kwargs["images_kwargs"],
            text_kwargs=output_kwargs["text_kwargs"],
        )

        # When suffix is provided, generate labels by masking non-suffix tokens
        if return_token_type_ids:
            labels = batch_doc["input_ids"].masked_fill(batch_doc["token_type_ids"] == 0, -100)
            batch_doc.update({"labels": labels})

        return batch_doc

    def process_queries(
        self,
        text: TextInput | list[TextInput],
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several text queries. Handles input validation, prepends the
        `query_prefix`, and appends query augmentation tokens (used to pad query embeddings for
        better late-interaction retrieval performance).

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
        """
        output_kwargs = self._merge_kwargs(
            ColModernVBertProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, list) and isinstance(text[0], str)):
            raise ValueError("Text must be a string or a list of strings")

        # Default suffix: repeat the augmentation token to pad query embeddings
        if suffix is None:
            suffix = self.query_augmentation_token * 10

        # Build final queries: prefix + original query + augmentation suffix
        texts_query: list[str] = [self.query_prefix + query + suffix for query in text]

        batch_query = self.__call__(
            text=texts_query,
            return_token_type_ids=False,
            text_kwargs=output_kwargs["text_kwargs"],
        )

        return batch_query

    def score_retrieval(
        self,
        query_embeddings: Union["torch.Tensor", list["torch.Tensor"]],
        passage_embeddings: Union["torch.Tensor", list["torch.Tensor"]],
        batch_size: int = 128,
        output_dtype: Optional["torch.dtype"] = None,
        output_device: Union["torch.device", str] = "cpu",
    ) -> "torch.Tensor":
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColQwen2, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            query_embeddings (`Union[torch.Tensor, list[torch.Tensor]`): Query embeddings.
            passage_embeddings (`Union[torch.Tensor, list[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            output_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`): The dtype of the output tensor.
                If `None`, the dtype of the input embeddings is used.
            output_device (`torch.device` or `str`, *optional*, defaults to "cpu"): The device of the output tensor.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """

        if len(query_embeddings) == 0:
            raise ValueError("No queries provided")
        if len(passage_embeddings) == 0:
            raise ValueError("No passages provided")

        if query_embeddings[0].device != passage_embeddings[0].device:
            raise ValueError("Queries and passages must be on the same device")

        if query_embeddings[0].dtype != passage_embeddings[0].dtype:
            raise ValueError("Queries and passages must have the same dtype")

        if output_dtype is None:
            output_dtype = query_embeddings[0].dtype

        scores: list[torch.Tensor] = []

        for i in range(0, len(query_embeddings), batch_size):
            batch_scores: list[torch.Tensor] = []
            batch_queries = torch.nn.utils.rnn.pad_sequence(
                query_embeddings[i : i + batch_size], batch_first=True, padding_value=0
            )
            for j in range(0, len(passage_embeddings), batch_size):
                batch_passages = torch.nn.utils.rnn.pad_sequence(
                    passage_embeddings[j : j + batch_size], batch_first=True, padding_value=0
                )
                batch_scores.append(
                    torch.einsum("bnd,csd->bcns", batch_queries, batch_passages).max(dim=3)[0].sum(dim=2)
                )
            scores.append(torch.cat(batch_scores, dim=1).to(output_dtype).to(output_device))

        return torch.cat(scores, dim=0)


@auto_docstring
class ColModernVBertPreTrainedModel(ColPaliPreTrainedModel):
    config: ColModernVBertConfig


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
    image_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True` and `pixel_values` are provided):
        Tuple of `torch.FloatTensor` (one for the output of the image modality projection + one for the output of each layer) of shape
        `(batch_size, num_channels, image_size, image_size)`.
        Hidden-states of the image encoder at the output of each layer plus the initial modality projection outputs.
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
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: ColModernVBertConfig):
        super().__init__(config)
        self.vlm = AutoModel.from_config(config.vlm_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ColModernVBertForRetrievalOutput:
        vlm_output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
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
