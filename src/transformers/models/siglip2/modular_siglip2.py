# Copyright 2025 The HuggingFace Inc. team.
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from tokenizers import normalizers

from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SiglipForImageClassification,
    SiglipModel,
    SiglipMultiheadAttentionPoolingHead,
    SiglipOutput,
    SiglipPreTrainedModel,
    SiglipTextModel,
    SiglipTextModelOutput,
    SiglipVisionModel,
    SiglipVisionModelOutput,
    SiglipVisionTransformer,
)

from ...masking_utils import create_bidirectional_mask
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    torch_compilable_check,
)
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs


class Siglip2Tokenizer(GemmaTokenizer):
    """
    Gemma tokenizer + SigLIP2 training default: lowercase normalization.
    """

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        super().__init__(
            vocab=vocab,
            merges=merges,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # Persist for save/load + push_to_hub dynamic tokenizer test
        if hasattr(self, "init_kwargs") and isinstance(self.init_kwargs, dict):
            self.init_kwargs.setdefault("tokenizer_class", self.__class__.__name__)

        backend = getattr(self, "_tokenizer", None)
        if backend is not None and backend.normalizer is not None:
            backend.normalizer = normalizers.Sequence([normalizers.Lowercase(), backend.normalizer])


@auto_docstring(checkpoint="google/siglip2-base-patch16-naflex")
@strict
class Siglip2TextConfig(SiglipTextConfig):
    pass


@auto_docstring(checkpoint="google/siglip2-base-patch16-naflex")
@strict
class Siglip2VisionConfig(SiglipVisionConfig):
    r"""
    num_patches (`int`, *optional*, defaults to 256):
        The number of patches in the image with the size of (`patch_size`, `patch_size`).
        The image is resized to fill maximum of this number of patches, and to preserve
        the aspect ratio. In case the resulted number of patches is lower, the image is
        padded in "patch" dimension.

    Example:

    ```python
    >>> from transformers import Siglip2VisionConfig, Siglip2VisionModel

    >>> # Initializing a Siglip2VisionConfig with google/siglip2-base-patch16-naflex style configuration
    >>> configuration = Siglip2VisionConfig()

    >>> # Initializing a Siglip2VisionModel (with random weights) from the google/siglip2-base-patch16-naflex style configuration
    >>> model = Siglip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    num_patches: int = 256
    image_size = AttributeError()


@auto_docstring(checkpoint="google/siglip2-base-patch16-naflex")
@strict
class Siglip2Config(SiglipConfig):
    pass


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
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i].tolist()  # will be itemized in F.interpolate either way
            torch_compilable_check((width > 0), "Width of resized positional embeddings must be positive.")
            torch_compilable_check((height > 0), "Height of resized positional embeddings must be positive.")
            torch_compilable_check((height * width) <= max_length, "Resized positional embeddings exceed max_length.")
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`list[tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2PreTrainedModel(SiglipPreTrainedModel):
    # nn.MultiHeadAttention mask doesn't allow for non 4d mask
    _supports_flex_attn = False
    _supports_flash_attn = False


class Siglip2VisionTransformer(SiglipVisionTransformer):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        """
        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )


class Siglip2TextModel(SiglipTextModel):
    pass


class Siglip2MultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)

        self.config = config
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=probe,
                attention_mask=attention_mask,
                encoder_hidden_states=hidden_state,
            )
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
                attention_mask = attention_mask.reshape(-1, target_len, source_len)

                # `nn.MultiheadAttention` cannot handle boolean masks (which SDPA can)
                if attention_mask.dtype == torch.bool:
                    attention_mask = torch.where(
                        attention_mask,
                        torch.tensor(0.0, device=attention_mask.device, dtype=probe.dtype),
                        torch.finfo(probe.dtype).min,
                    )

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Siglip2VisionModel(SiglipVisionModel):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, Siglip2VisionModel

        >>> model = Siglip2VisionModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            **kwargs,
        )


class Siglip2Model(SiglipModel):
    # Update: add `spatial_shapes` and `pixel_attention_mask`
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModel
        >>> from transformers.image_utils import load_image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```
        """
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            **kwargs,
        )

    # Update: add `spatial_shapes` and `pixel_attention_mask`
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_loss: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Siglip2Output:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # important: we pass `padding=max_length` since the model was trained with this
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```
        """
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            **kwargs,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip2.py#L287
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

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
    # Update: add `spatial_shapes` and `pixel_attention_mask`
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, Siglip2ForImageClassification
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> # note: we are loading a `Siglip2Model` from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random if seed is not set above.
        >>> image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
        >>> model = Siglip2ForImageClassification.from_pretrained("google/siglip2-base-patch16-224")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the two classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: LABEL_1
        ```
        """
        outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        # average pool the patch tokens
        if pixel_attention_mask is not None:
            pool_mask = pixel_attention_mask[..., None].to(sequence_output.device)
            sequence_output = torch.sum(sequence_output * pool_mask, dim=1) / torch.sum(pool_mask, dim=1)
        else:
            sequence_output = torch.mean(sequence_output, dim=1)

        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Siglip2Config",
    "Siglip2TextConfig",
    "Siglip2VisionConfig",
    "Siglip2Model",
    "Siglip2PreTrainedModel",
    "Siglip2TextModel",
    "Siglip2VisionModel",
    "Siglip2ForImageClassification",
    "Siglip2Tokenizer",
]
