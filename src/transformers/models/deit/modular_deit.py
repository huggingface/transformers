from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedImageModelingOutput,
)
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, torch_int
from ...utils.generic import can_return_tuple
from ..vit.modeling_vit import (
    ViTEmbeddings,
    ViTForImageClassification,
    ViTForMaskedImageModeling,
    ViTModel,
    ViTPreTrainedModel,
)
from .configuration_deit import DeiTConfig


class DeiTEmbeddings(ViTEmbeddings):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: DeiTConfig, use_mask_token: bool = False) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and 2 class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 2
        num_positions = self.position_embeddings.shape[1] - 2

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_and_dist_pos_embed = self.position_embeddings[:, :2]
        patch_pos_embed = self.position_embeddings[:, 2:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_and_dist_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        batch_size, seq_length, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)

        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        position_embeddings = self.position_embeddings

        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class DeiTPreTrainedModel(ViTPreTrainedModel):
    _no_split_modules = ["DeiTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DeiTEmbeddings):
            module.cls_token.data.zero_()
            module.position_embeddings.data.zero_()
            module.distillation_token.data.zero_()
            if module.mask_token is not None:
                module.mask_token.data.zero_()


class DeiTModel(ViTModel):
    pass


class DeiTForMaskedImageModeling(ViTForMaskedImageModeling):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedImageModelingOutput:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DeiTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        >>> model = DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""

        outputs: BaseModelOutputWithPooling = self.deit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        # Reshape to (batch_size, num_channels, height, width)
        # Remove the [CLS] token (index 0) and distillation token (index 1), keep only patch embeddings
        sequence_output = sequence_output[:, 2:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = int(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeiTForImageClassification(ViTForImageClassification):
    pass


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`DeiTForImageClassificationWithTeacher`].
    """
)
class DeiTForImageClassificationWithTeacherOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
        Prediction scores as the average of the cls_logits and distillation logits.
    cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
        Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
        class token).
    distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
        Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
        distillation token).
    """

    logits: Optional[torch.FloatTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    distillation_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring(
    custom_intro="""
    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of
    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

    .. warning::

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """
)
class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.deit = DeiTModel(config, add_pooling_layer=False)

        # Classifier heads
        self.cls_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DeiTForImageClassificationWithTeacherOutput:
        outputs: BaseModelOutputWithPooling = self.deit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])

        # during inference, return the average of both classifier predictions
        logits = (cls_logits + distillation_logits) / 2

        return DeiTForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "DeiTForImageClassification",
    "DeiTForImageClassificationWithTeacher",
    "DeiTForMaskedImageModeling",
    "DeiTModel",
    "DeiTPreTrainedModel",
]
