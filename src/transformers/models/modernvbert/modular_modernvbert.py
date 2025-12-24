import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..idefics3 import Idefics3ImageProcessor, Idefics3ImageProcessorFast, Idefics3Processor
from ..modernbert import ModernBertConfig, ModernBertModel
from ..modernbert.modeling_modernbert import ModernBertPredictionHead
from ..siglip import SiglipVisionConfig, SiglipVisionModel


logger = logging.get_logger(__name__)


class ModernVBertImageProcessor(Idefics3ImageProcessor):
    pass


class ModernVBertImageProcessorFast(Idefics3ImageProcessorFast):
    pass


class ModernVBertProcessor(Idefics3Processor):
    r"""
    Constructs a ModernVBert processor which wraps a LLama tokenizer and ModernVBert image processor into a single processor.

    [`ModernVBertProcessor`] offers all the functionalities of [`Idefics3Processor`]. See
    the docstring of [`~IdeficsProcessor.__call__`] for more information.

    Args:
        image_processor (`ModernVBertImageProcessor`):
            An instance of [`ModernVBertImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """


class ModernVBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a `ModernVBert` model. It is used to
    instantiate a ModernVBert model according to the specified arguments and defines the model architecture.
    e.g. [ModernVBERT/modernvbert](https://huggingface.co/ModernVBERT/modernvbert).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    See the documentation for [`PretrainedConfig`] for more details.

    Args:
            text_config ([`ModernBertConfig`], *optional*): [`ModernBertModel`] config to build the text encoder.
            vision_config ([`SiglipVisionConfig`], *optional*): [`SiglipVisionModel`] config to build the vision encoder.
            image_token_id (`Optional`, *optional*, defaults to 50407): The token id reserved for image tokens inserted into the text stream.
            pixel_shuffle_factor (`Optional`, *optional*, defaults to 4): Scale factor used by any pixel-shuffle / upsampling operations in the vision head.
            vocab_size (`Optional`, *optional*): Vocabulary size of the text model. Defines the number of different tokens that can be represented
                by the `inputs_ids` passed when calling [`ModernVBertModel`]. If not provided, will be taken from
                the `text_config`.
            hidden_size (`Optional`, *optional*): Dimensionality of the encoder layers and the pooler layer. If not provided, will be taken from
                the `text_config`.
            num_hidden_layers (`Optional`, *optional*): Number of hidden layers in the Transformer encoder. If not provided, will be taken from
                the `text_config`.
            initializer_range (`Optional`, *optional*, defaults to 0.02): The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            initializer_cutoff_factor (`Optional`, *optional*, defaults to 2.0): The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
            classifier_pooling (`Literal`, *optional*, defaults to `"cls"`): The pooling strategy to use for classification tasks. Can be either `"cls"` or `"mean"`.
            classifier_dropout (`Optional`, *optional*, defaults to 0.0): The dropout probability for the classification head.
            classifier_bias (`Optional`, *optional*, defaults to `False`): Whether to add a bias term to the classification head.

    Example:
    ```python
    >>> from modernvbert import ModernVBertConfig

    >>> # Initializing configuration
    >>> configuration = ModernVBertConfig()

    >>> # Initializing a model from the configuration (model class is implemented in
    >>> # `modernvbert.modeling_modernvbert`)

    >>> from modernvbert import ModernVBertModel
    >>> model = ModernVBertModel(configuration)

    >>> # Accessing the model configuration
    >>> cfg = model.config
    ```"""

    model_type = "modernvbert"
    sub_configs: dict[str, Any] = {"text_config": ModernBertConfig, "vision_config": SiglipVisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: Optional[int] = 50407,
        pixel_shuffle_factor: Optional[int] = 4,
        initializer_range: Optional[float] = 0.02,
        initializer_cutoff_factor: Optional[float] = 2.0,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: Optional[float] = 0.0,
        classifier_bias: Optional[bool] = False,
        **kwargs,
    ):
        if classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {classifier_pooling}.'
            )

        if "vocab_size" in kwargs:
            logger.warning(
                "The vocab_size parameter is deprecated, please set vocab_size in the `text_config` instead."
            )

        if text_config is None:
            text_config = self.sub_configs["text_config"]()
        elif isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"](**text_config)
        self.text_config = text_config

        if vision_config is None:
            vision_config = self.sub_configs["vision_config"]()
        elif isinstance(vision_config, dict):
            vision_config = self.sub_configs["vision_config"](**vision_config)
        self.vision_config = vision_config

        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias

        super().__init__(image_token_id=image_token_id, **kwargs)


@dataclass
class ModernVBertBaseModelOutput(BaseModelOutput):
    """
    Base class for ModernVBERT model's outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class ModernVBertMaskedLMOutput(MaskedLMOutput):
    """
    Base class for ModernVBERT model's outputs with masked language modeling loss.
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class ModernVBertConnector(nn.Module):
    """
    Connector module for ModernVBERT. It performs a pixel shuffle operation followed by a linear projection to match the text model's hidden size.
    Based on https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
    """

    def __init__(self, config):
        super().__init__()
        self.pixel_shuffle_factor = config.pixel_shuffle_factor
        self.modality_projection = nn.Linear(
            config.vision_config.hidden_size * (config.pixel_shuffle_factor**2),
            config.text_config.hidden_size,
            bias=False,
        )

    def pixel_shuffle(self, x, pixel_shuffle_factor):
        batch_size, seq_length, embed_dim = x.size()
        height = width = int(seq_length**0.5)
        x = x.view(batch_size, height, width, embed_dim)
        x = x.view(batch_size, height, int(width / pixel_shuffle_factor), embed_dim * pixel_shuffle_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size,
            int(width / pixel_shuffle_factor),
            int(height / pixel_shuffle_factor),
            embed_dim * (pixel_shuffle_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        return x.reshape(
            batch_size, int(seq_length / (pixel_shuffle_factor**2)), embed_dim * (pixel_shuffle_factor**2)
        )

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.pixel_shuffle_factor)
        return self.modality_projection(image_hidden_states)


@auto_docstring
class ModernVBertPreTrainedModel(PreTrainedModel):
    config_class = ModernVBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    input_modalities = ["image", "text"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        cutoff_factor = self.config.initializer_cutoff_factor

        out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
        final_out_std = self.config.text_config.hidden_size**-0.5

        def init_weight(module: nn.Module, std: float):
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.zeros_(module.bias)

        if isinstance(module, ModernVBertConnector):
            init_weight(module.modality_projection, out_std)
        elif isinstance(module, ModernVBertForMaskedLM):
            init_weight(module.lm_head, out_std)
        elif isinstance(
            module,
            (
                ModernVBertForSequenceClassification,
                ModernVBertForMultipleChoice,
                ModernVBertForTokenClassification,
                ModernVBertForQuestionAnswering,
            ),
        ):
            init_weight(module.classifier, final_out_std)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            init_weight(module, std)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)


@auto_docstring(
    custom_intro="""
    ModernVBertModel is a model that combines a vision encoder (SigLIP) and a text encoder (ModernBert).

    ModernVBert is the base model of the visual retriver ColModernVBert, and was introduced in the following paper:
    [*ModernVBERT: Towards Smaller Visual Document Retrievers*](https://arxiv.org/abs/2510.01149).
    """
)
class ModernVBertModel(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)

        # init components
        self.connector = ModernVBertConnector(config)
        self.text_model = ModernBertModel(config.text_config)
        self.vision_model = SiglipVisionModel(config.vision_config)

        # initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_image_features(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        Derived from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        image_hidden_states = image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states)

        return image_hidden_states

    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py

        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """

        _, patch_size, _ = image_hidden_states.shape

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask[..., 0]  # slice off the hidden dim
        else:
            image_mask = input_ids == self.config.image_token_id

        # Assert that the input <image> tokens are valid (i.e. multiple of patch_size)
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("Number of <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # Images processing
        if pixel_values is not None:
            # Vision encoder pass
            image_hidden_states = self.get_image_features(
                pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask
            )

        # Merge image and text embeddings
        if image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids, inputs_embeds=inputs_embeds, image_hidden_states=image_hidden_states
            )

        # Language model pass
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        return ModernVBertBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class ModernVBertPredictionHead(ModernBertPredictionHead):
    pass


@auto_docstring
class ModernVBertForMaskedLM(ModernVBertPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embeddings.tok_embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)

        self.vocab_size = config.text_config.vocab_size

        self.model = ModernVBertModel(config)
        self.projection_head = ModernVBertPredictionHead(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=config.text_config.decoder_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, ModernVBertMaskedLMOutput]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        hidden_states = outputs[0]

        logits = self.lm_head(self.projection_head(hidden_states))

        loss = None
        if labels is not None:
            criterion = CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.vocab_size), labels.view(-1))

        return ModernVBertMaskedLMOutput(
            loss=loss,
            logits=logits.float(),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        embeds = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

        # if not tying embeddings, resize lm_head as well
        if not self.config.tie_word_embeddings:
            old_lm_head_weight = self.lm_head.weight.data
            old_lm_head_bias = self.lm_head.bias.data
            old_num_tokens, old_embedding_dim = old_lm_head_weight.shape

            if new_num_tokens is None:
                new_num_tokens = old_num_tokens

            # Create new lm_head with the new size
            new_lm_head = nn.Linear(self.lm_head.in_features, new_num_tokens, bias=self.lm_head.bias is not None)
            new_lm_head.to(old_lm_head_weight.device)

            # Initialize the new lm_head and copy over the weights from the old lm_head
            self._init_weights(new_lm_head)
            new_lm_head.weight.data[: min(old_num_tokens, new_num_tokens), :] = old_lm_head_weight[
                : min(old_num_tokens, new_num_tokens), :
            ]

            if mean_resizing and new_num_tokens > old_num_tokens:
                mean_vector = old_lm_head_weight.mean(dim=0, keepdim=True)
                new_lm_head.weight.data[old_num_tokens:, :] = mean_vector.repeat(new_num_tokens - old_num_tokens, 1)

            if old_lm_head_bias is not None:
                # If lm_head has a bias, resize it as well
                new_lm_head.bias.data[: min(old_num_tokens, new_num_tokens)] = old_lm_head_bias[
                    : min(old_num_tokens, new_num_tokens)
                ]

                if mean_resizing and new_num_tokens > old_num_tokens:
                    new_lm_head.bias.data[old_num_tokens:] = 0.0

            self.set_output_embeddings(new_lm_head)

        return embeds


@auto_docstring(
    custom_intro="""
    The ModernVBert Model with a sequence classification head on top that performs pooling.
    """
)
class ModernVBertForSequenceClassification(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernVBertModel(config)
        self.head = ModernVBertPredictionHead(config.text_config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, ModernVBertMaskedLMOutput]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        batch_size (`int`, *optional*):
            The batch size of the input. If not provided, it will be inferred from `input_ids` or `inputs_embeds`.
        seq_len (`int`, *optional*):
            The sequence length of the input. If not provided, it will be inferred from `input_ids` or `inputs_embeds`.
        """

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernVBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.
    """
)
class ModernVBertForTokenClassification(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernVBertModel(config)
        self.head = ModernVBertPredictionHead(config.text_config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, ModernVBertMaskedLMOutput]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class ModernVBertForQuestionAnswering(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernVBertModel(config)
        self.head = ModernVBertPredictionHead(config.text_config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.num_labels)

        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, ModernVBertMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernVBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.
    """
)
class ModernVBertForMultipleChoice(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)
        self.config = config

        self.model = ModernVBertModel(config)
        self.head = ModernVBertPredictionHead(config.text_config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.text_config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        pixel_values = (
            pixel_values.view(
                -1, pixel_values.size(-4), pixel_values.size(-3), pixel_values.size(-2), pixel_values.size(-1)
            )
            if pixel_values is not None
            else None
        )
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        pixel_attention_mask = (
            pixel_attention_mask.view(-1, pixel_attention_mask.size(-2), pixel_attention_mask.size(-1))
            if pixel_attention_mask is not None
            else None
        )
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        image_hidden_states = (
            image_hidden_states.view(
                -1, image_hidden_states.size(-3), image_hidden_states.size(-2), image_hidden_states.size(-1)
            )
            if image_hidden_states is not None
            else None
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        last_hidden_state = outputs[0]  # shape (num_choices, seq_len, hidden_size)

        # If classifier_pooling is "cls", isolate the <cls> token
        if self.config.classifier_pooling == "cls":
            indices_0 = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            # for left or right padding, <cls> is the first non-pad token
            if attention_mask is not None:
                # make attention_mask long for argmax
                attention_mask = attention_mask.long()
                cls_mask = attention_mask.argmax(dim=-1).to(last_hidden_state.device)
            # if no pad, <cls> is the first token
            else:
                cls_mask = torch.tensor(0, dtype=torch.long, device=last_hidden_state.device)
            # extract the <cls> token for the logits
            last_hidden_state = last_hidden_state[indices_0, cls_mask]

        # If classifier_pooling is "mean", pool the hidden states by averaging over the sequence length
        elif self.config.classifier_pooling == "mean":
            num_non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / num_non_pad_tokens

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ModernVBertImageProcessor",
    "ModernVBertImageProcessorFast",
    "ModernVBertProcessor",
    "ModernVBertConfig",
    "ModernVBertPreTrainedModel",
    "ModernVBertModel",
    "ModernVBertForMaskedLM",
    "ModernVBertForSequenceClassification",
    "ModernVBertForTokenClassification",
    "ModernVBertForQuestionAnswering",
    "ModernVBertForMultipleChoice",
]
