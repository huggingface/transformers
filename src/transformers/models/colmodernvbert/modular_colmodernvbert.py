from dataclasses import dataclass
from typing import Any, Optional, Union

from torch import nn

from ... import initialization as init
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import ModelOutput, auto_docstring, can_return_tuple, is_torch_available, logging
from ..colqwen2 import ColQwen2Config
from ..modernvbert import ModernVBertConfig, ModernVBertModel, ModernVBertProcessor


if is_torch_available():
    import torch

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
    from transformers.models.ColModernVBert import ColModernVBertConfig, ColModernVBertForRetrieval

    config = ColModernVBertConfig()
    model = ColModernVBertForRetrieval(config)
    ```
    """

    model_type = "colmodernvbert"
    sub_configs: dict[str, Any] = {"vlm_config": ModernVBertConfig}


class ColModernVBertProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ColModernVBertProcessor(ModernVBertProcessor):
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
        image_seq_len: int = 64,
        visual_prompt_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_len=image_seq_len,
            chat_template=None,  # No chat template used for retrieval
            **kwargs,
        )
        self.visual_prompt_prefix = (
            visual_prompt_prefix or "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
        )
        self.query_prefix = query_prefix or ""
        self.query_augmentation_token = self.end_of_utterance_token

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
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
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

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

            batch_doc = super().__call__(
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

            batch_query = super().__call__(
                text=texts_query,
                return_token_type_ids=False,
                text_kwargs=output_kwargs["text_kwargs"],
            )

            return batch_query

    def process_images(
        self,
        images: Optional[ImageInput] = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several image(s). This method is a wrapper around the `__call__` method of the ColModernVBertProcessor's
        [`ColModernVBertProcessor.__call__`].
        This method forwards the `images` and `kwargs` arguments to the image processor.

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
        return self.__call__(images=images, **kwargs)

    def process_queries(
        self,
        text: Union[TextInput, list[TextInput]],
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several texts. This method is a wrapper around the `__call__` method of the ColModernVBertProcessor's
        [`ColModernVBertProcessor.__call__`].
        This method forwards the `text` and `kwargs` arguments to the tokenizer.

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
        return self.__call__(text=text, **kwargs)

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
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
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
class ColModernVBertPreTrainedModel(PreTrainedModel):
    config: ColModernVBertConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        std = self.config.vlm_config.initializer_range
        cutoff_factor = self.config.vlm_config.initializer_cutoff_factor

        if isinstance(module, (nn.Linear)):
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if module.bias is not None:
                init.zeros_(module.bias)


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

    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


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
class ColModernVBertForRetrieval(ColModernVBertPreTrainedModel):
    def __init__(self, config: ColModernVBertConfig):
        super().__init__(config)
        self.config = config

        self.vlm = ModernVBertModel(config.vlm_config)
        self.embedding_proj_layer = nn.Linear(self.config.get_text_config().hidden_size, self.config.embedding_dim)

        self._tied_weights_keys = {f"model.{k}": v for k, v in (self.vlm._tied_weights_keys or {}).items()}

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        model_embeds = self.vlm.resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )

        self.config.vlm_config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vlm_config.vocab_size = model_embeds.num_embeddings
        self.vlm.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings

        return model_embeds


__all__ = [
    "ColModernVBertConfig",
    "ColModernVBertForRetrieval",
    "ColModernVBertPreTrainedModel",
    "ColModernVBertProcessor",
]
