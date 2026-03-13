from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as tvF

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images

from ..got_ocr2.configuration_got_ocr2 import GotOcr2Config
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2ModelOutputWithPast,
    GotOcr2Model,
    GotOcr2PreTrainedModel,
    GotOcr2ForConditionalGeneration,
    GotOcr2VisionEncoder,
)

from ...utils import TransformersKwargs, auto_docstring, logging
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessorMixin, TensorType, Unpack

from ...image_utils import SizeDict

logger = logging.get_logger(__name__)


@auto_docstring
class PPChart2TableConfig(GotOcr2Config):
    pass


@auto_docstring
class PPChart2TableImageProcessorFast(BaseImageProcessorFast):
    resample = 3
    image_mean = [0.40821073, 0.4578275, 0.48145466]
    image_std = [0.27577711, 0.26130258, 0.26862954]
    size = {"height": 1024, "width": 1024}
    patch_size = 16
    merge_size = 4
    do_resize = True
    do_rescale = True
    do_normalize = True

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # BGR to RGB conversion
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            processed_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)

        return BatchFeature(
            data={"pixel_values": pixel_values},
            tensor_type=return_tensors,
        )


@auto_docstring
class PPChart2TableProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.message_start_token = "<|im_start|>"
        self.message_end_token = "<|im_end|>"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_pad_token = "<imgpad>"
        self.image_token = "<imgpad>"  # keep the above for BC, but we need to call it `image_token`
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.system_query = "system\nYou should follow the instructions carefully and explain your answers in detail."

    def __call__(
        self,
        images,
        text=None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
        else:
            image_inputs = {}
        image_count = len(image_inputs)
        _, _, height, _ = image_inputs["pixel_values"].shape
        num_patches = height // self.image_processor.patch_size // self.image_processor.merge_size
        
        input_ids = {"input_ids": None}
        if text == None:
            query = "Chart to table"
            prompt = (
                self.message_start_token
                + self.system_query
                + self.message_end_token
                + self.message_start_token
                + "user\n"
                + self.img_start_token
                + self.img_pad_token * num_patches * num_patches
                + self.img_end_token
                + "\n"
                + query
                + self.message_end_token
                + self.message_start_token
                + "assistant\n"
            )
            input_ids = torch.tensor(self.tokenizer([prompt]).input_ids)
            input_ids = input_ids.repeat(image_count, 1)
            input_ids = {"input_ids": input_ids}
        return BatchFeature(data={**input_ids, **image_inputs})

    def postprocess(self, model_pred, **kwargs):
        return self.tokenizer.batch_decode(
            model_pred[0],
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
            clean_up_tokenization_spaces=False,
        )


class PPChart2TableVisionPreTrainedModel(GotOcr2PreTrainedModel):
    input_modalities = ("image", "text")


class PPChart2TableVisionEncoder(GotOcr2VisionEncoder, PPChart2TableVisionPreTrainedModel):
    pass


@dataclass
class PPChart2TableModelOutputWithPast(GotOcr2ModelOutputWithPast):
    pass


@auto_docstring
class PPChart2TableModel(GotOcr2Model):

    def __init__(self, config: PPChart2TableConfig):
        super().__init__(config)
        self.vision_downsample1 = nn.Conv2d(config.vision_config.output_channels, config.vision_hidden_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.vision_downsample2 = nn.Conv2d(config.vision_hidden_channels, config.output_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.multi_modal_projector = nn.Linear(config.output_channels, config.text_config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        
        image_output = self.vision_tower(pixel_values)
        last_hidden_state = image_output.last_hidden_state
        last_hidden_state = self.vision_downsample1(last_hidden_state)
        last_hidden_state = self.vision_downsample2(last_hidden_state)
        image_output.pooler_output = self.multi_modal_projector(last_hidden_state.flatten(2).transpose(2, 1))

        return image_output


@auto_docstring(
    custom_intro="""
    PP-Chart2Table model for conditional generation (table text generation from chart images),
    extending the core model with a language modeling (LM) head and generation utilities.
    """
)
class PPChart2TableForConditionalGeneration(GotOcr2ForConditionalGeneration):
    pass


__all__ = [
    "PPChart2TableForConditionalGeneration",
    "PPChart2TableModel",
    "PPChart2TablePreTrainedModel",
    "PPChart2TableConfig",
    "PPChart2TableTextPreTrainedModel",
    "PPChart2TableVisionPreTrainedModel",
    "PPChart2TableVisionModel",
    "PPChart2TableImageProcessorFast",
    "PPChart2TableProcessor",
]