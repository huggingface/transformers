from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SiglipForImageClassification,
    SiglipModel,
    SiglipOutput,
    SiglipPreTrainedModel,
    SiglipTextModel,
    SiglipTextModelOutput,
    SiglipVisionModel,
    SiglipVisionModelOutput,
    SiglipVisionTransformer,
)
from transformers.models.siglip.processing_siglip import SiglipProcessor

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class Siglip2TextConfig(SiglipTextConfig):
    pass


class Siglip2VisionConfig(SiglipVisionConfig):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        del self.image_size


class Siglip2Config(SiglipConfig):
    pass


# Update: docstring SiglipTokenizer->GemmaTokenizerFast
# Update: image_processor_class and tokenizer_class
class Siglip2Processor(SiglipProcessor):
    r"""
    Constructs a Siglip2 processor which wraps a Siglip2 image processor and a Gemma tokenizer into a single processor.

    [`Siglip2Processor`] offers all the functionalities of [`Siglip2ImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~Siglip2Processor.__call__`] and [`~Siglip2Processor.decode`] for more information.

    Args:
        image_processor ([`Siglip2ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`GemmaTokenizerFast`]):
            The tokenizer is a required input.
    """

    image_processor_class = "Siglip2ImageProcessor"
    tokenizer_class = "GemmaTokenizerFast"

    # Update docstring: SiglipTokenizer->GemmaTokenizerFast, add `pixel_attention_mask` and `pixel_position_ids`
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to GemmaTokenizerFast's [`~GemmaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` argument to
        Siglip2ImageProcessor's [`~Siglip2ImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
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

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_attention_mask** -- Attention mask for the pixel values. Returned when `images` is not `None`.
            - **pixel_position_ids** -- Position ids for the pixel values. Returned when `images` is not `None`.
        """
        return super().__call__(text, images, padding, truncation, max_length, return_tensors)


class Siglip2VisionOutput(SiglipVisionModelOutput):
    pass


class Siglip2TextOutput(SiglipTextModelOutput):
    pass


class Siglip2Output(SiglipOutput):
    pass


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def _sample_positional_embeddings_unbatched(
        positional_embeddings: torch.Tensor, position_ids: torch.Tensor, shape
    ) -> torch.Tensor:
        """
        Sample the positional embeddings with provided position ids (loop over batch elements, not optimized).

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            position_ids (`torch.Tensor`):
                Pixel position ids of shape (max_num_patches, 2)

        Returns:
            `torch.Tensor`: Embeddings of shape (max_num_patches, embed_dim)
            corresponding to the input pixel position ids.
        """

        # Convert from (height, width, embed_dim) to (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        height, width = shape

        # resize positional embeddings
        # 1, dim, height, width -> 1, dim, target_height, target_width
        resized_embeddings = F.interpolate(
            positional_embeddings,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # Rearrange: 1, dim, target_height, target_width -> target_height * target_width, dim
        resized_embeddings = resized_embeddings.squeeze(0).flatten(1).transpose(1, 0)

        # Convert 2d position ids to 1d absolute position ids
        y_ids = position_ids[..., 0]
        x_ids = position_ids[..., 1]
        position_ids_1d = y_ids * width + x_ids

        # Get the positional embeddings for the absolute position ids
        sampled_positional_embeddings = resized_embeddings[position_ids_1d]

        return sampled_positional_embeddings

    def sample_positional_embeddings_vmap(
        self, positional_embeddings: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """This one does not work because of interpolation requires python scalars, but vmap does not allow to pass them
        and does not allow to"""
        target_shapes = position_ids.max(dim=1).values + 1
        return torch.vmap(self._sample_positional_embeddings_unbatched, in_dims=(None, 0, 0))(
            positional_embeddings, position_ids, target_shapes
        )

    @staticmethod
    def sample_positional_embeddings(positional_embeddings: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Sample the positional embeddings with provided position ids (loop over batch elements, not optimized).

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            position_ids (`torch.Tensor`):
                Pixel position ids of shape (batch_size, max_num_patches, 2)

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_num_patches, embed_dim)
            corresponding to the input pixel position ids.
        """
        target_shapes = position_ids.max(dim=1).values + 1

        # Convert from (height, width, embed_dim) to (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        resulted_positional_embeddings = []
        for (height, width), position_ids_i in zip(target_shapes, position_ids):
            # resize positional embeddings for i-th image
            # 1, dim, height, width -> 1, dim, target_height, target_width
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # Rearrange: 1, dim, target_height, target_width -> target_height * target_width, dim
            resized_embeddings = resized_embeddings.squeeze(0).flatten(1).transpose(1, 0)

            # Convert 2d position ids to 1d absolute position ids
            y_ids = position_ids_i[..., 0]
            x_ids = position_ids_i[..., 1]
            position_ids_1d = y_ids * width + x_ids

            # Get the positional embeddings for the absolute position ids
            gathered_positional_embeddings = resized_embeddings[position_ids_1d]
            resulted_positional_embeddings.append(gathered_positional_embeddings)

        resulted_positional_embeddings = torch.stack(resulted_positional_embeddings)
        return resulted_positional_embeddings

    @staticmethod
    def sample_positional_embeddings_with_grid_sample(
        positional_embeddings: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        THIS IS OPTIMIZED VERSION OF `sample_positional_embeddings`, BUT DUE TO `antialias=True` IN `F.interpolate`
        THE OUTPUT DOES NOT MATCH AND THAT INFLUENCES THE RESULT A LOT.

        Sample the positional embeddings with provided position ids.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            position_ids (`torch.Tensor`):
                Pixel position ids of shape (batch_size, max_num_patches, 2)

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_num_patches, embed_dim)
            corresponding to the input pixel position ids.
        """
        batch_size = position_ids.shape[0]

        # Convert from (height, width, embed_dim) to (batch_size, embed_dim, height, width) for grid_sample
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        positional_embeddings = positional_embeddings.expand(batch_size, -1, -1, -1)

        # Normalize position_ids to range [-1, 1] for grid_sample
        size = position_ids.max(dim=1, keepdim=True).values + 1
        grid = (position_ids.float() + 0.5) / size * 2 - 1

        # Flip yx -> xy for grid_sample
        grid = torch.flip(grid, dims=[-1])

        # (batch, 1, num_patches, 2) to match the expected input shape for grid_sample
        grid = grid.unsqueeze(1)

        # Sample from position embeddings
        embeddings = F.grid_sample(positional_embeddings, grid, mode="bilinear", align_corners=True)
        embeddings = embeddings.squeeze(2).transpose(1, 2)

        return embeddings

    def forward(self, pixel_values: torch.FloatTensor, position_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            position_ids (`torch.LongTensor`):
                Position ids (x, y) of shape (batch_size, max_num_patches, 2)
        """

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        embeddings = patch_embeds + self.sample_positional_embeddings(positional_embeddings, position_ids)
        return embeddings


class Siglip2VisionTransformer(SiglipVisionTransformer):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    # Update: add `position_ids` and `attention_mask`
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, position_ids)

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Siglip2PreTrainedModel(SiglipPreTrainedModel):
    pass


class Siglip2TextModel(SiglipTextModel):
    pass


class Siglip2VisionModel(SiglipVisionModel):
    # Update: add `pixel_position_ids` and `pixel_attention_mask`
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_position_ids: torch.LongTensor,
        pixel_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            position_ids=pixel_position_ids,
            attention_mask=pixel_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )


class Siglip2Model(SiglipModel):
    # Update: add `pixel_position_ids` and `pixel_attention_mask`
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        # Use Siglip2Model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            position_ids=pixel_position_ids,
            attention_mask=pixel_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        pooled_output = vision_outputs[1]

        return pooled_output

    # Update: add `pixel_position_ids` and `pixel_attention_mask`
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, Siglip2Output]:
        # Use Siglip2 model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            position_ids=pixel_position_ids,
            attention_mask=pixel_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * self.logit_scale.exp()
            + self.logit_bias
        )
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip2.py#L287
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return Siglip2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class Siglip2ForImageClassification(SiglipForImageClassification):
    # Update: add `pixel_position_ids` and `pixel_attention_mask`
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        ```python
        >>> from transformers import AutoImageProcessor, Siglip2ForImageClassification
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # note: we are loading a `Siglip2Model` from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random if seed is not set above.
        >>> image_processor = AutoImageProcessor.from_pretrained("google/siglip2")
        >>> model = Siglip2ForImageClassification.from_pretrained("google/siglip2")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the two classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: LABEL_1
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            position_ids=pixel_position_ids,
            attention_mask=pixel_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        # average pool the patch tokens
        sequence_output = torch.mean(sequence_output, dim=1)
        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Siglip2Processor",
    "Siglip2Config",
    "Siglip2TextConfig",
    "Siglip2VisionConfig",
    "Siglip2Model",
    "Siglip2PreTrainedModel",
    "Siglip2TextModel",
    "Siglip2VisionModel",
    "Siglip2ForImageClassification",
]
