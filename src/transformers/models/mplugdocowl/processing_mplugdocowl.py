# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for MPLUGDocOwl.
"""


from typing import List, Optional, Union, Tuple
#FIXME change the import from transformers to import from ...
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
#FIXME need to add image processing class name
#from transformers.models.mplugdocowl.image_processing_mplugdocowl import MPLUGDocOwlImageProcessor
import numpy as np

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    return iou, union

def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    input_image_bbox = np.array([[0, 0, input_image_size[1], input_slider_image_size[0]]])

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.copy()
    boxes3[:, 3] = input_image_size[0] / input_image_size[1] * anchors[:, 2]  # for resolution-independent iou
    
    area1 = anchors_areas
    
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = np.diag(shape_iou)  # Get diagonal for self-comparison
    index = np.argmax(shape_iou * 100 + iou)
    return index

class AnchorResize:
    def __init__(self, image_size, anchors):
        self.anchors = np.array([[0, 0, x[1] * image_size[1], x[0] * image_size[0]] for x in anchors])
        self.anchor_areas = box_area(self.anchors)
        self.image_size = image_size

    def forward(self, img, skip_resize=False):
        selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.shape[1], img.shape[0]))
        target_size = self.anchors[selected_anchor][2:]  # w, h
        if skip_resize:
            return selected_anchor
        return np.resize(img, (int(target_size[1]), int(target_strong_size[0]))), selected_anchor

    def __repr__(self):
        detail = f"(size={self.image_size}, anchors={self.anchors})"
        return f"{self.__class__.__name__}{detail}"

class MPLUGDocOwlProcessor(ProcessorMixin):
    r"""
    Constructs a MPLUGDocOwl processor which wraps a MPLUGDocOwl image processor and a MPLUGDocOwl tokenizer into a single processor.

    [`MPLUGDocOwlProcessor`] offers all the functionalities of [`MPLUGDocOwlImageProcessor`] and [`MPLUGDocOwlTokenizerFast`]. See the
    [`~MPLUGDocOwlProcessor.__call__`] and [`~MPLUGDocOwlProcessor.decode`] for more information.

    Args:
        image_processor ([`MPLUGDocOwlImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`MPLUGDocOwlTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MPLUGDocOwlImageProcessor"
    tokenizer_class = ("AutoTokenizer")#, "AutoTokenizerFast")
    
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        add_textual_crop_indicator: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to MPLUGDocOwlTokenizerFast's [`~MPLUGDocOwlTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MPLUGDocOwlImageProcessor's [`~MPLUGDocOwlImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
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
        """
        #FIXME need to add image processing class name properly
        
        if images is not None:
            pixel_values = self.image_processor(images, do_rescale=False, do_convert_rgb=True, do_shape_adaptive_cropping=True, do_resize=True, do_normalize=True, return_tensors=return_tensors,image_mean=(0.48145466, 0.4578275, 0.40821073), image_std=(0.26862954, 0.26130258, 0.27577711),resample=None,size=224)
        else:
            pixel_values = None
        #text prpeocessing
        breakpoint()
        media_token = '<|image|>'
        assert media_token in text
        patch_positions = pixel_values['patch_positions']
        num_patches = pixel_values['num_patches']
        anchor_max = pixel_values['anchor_max']
        text_list = text.split(media_token)
        text = text_list[0]
        image_token_ptr = 0
        for next_text in text_list[1:]:
            if add_textual_crop_indicator:
                # generate image placeholders with interleaved texutual crop indicator
                # e.g. <global_img><|image|><crop_img_row0_col0><|image|><crop_img_row0_col1><|image|>...
                for patch_pos in patch_positions.tolist():
                    # global non-crop image
                    if patch_pos[0] == anchor_max and patch_pos[1] == anchor_max:
                        text += '<global_img><|image|>'
                    else:
                        row_col = 'row'+str(patch_pos[0])+'_col'+str(patch_pos[1])
                        text += '<crop_img_'+row_col+'><|image|>'
            else: 
                # generate successive image placeholders for a image, 1 crop img == 1 <|image|>
                breakpoint()
                text += '<|image|>'*num_patches
            text += next_text
            image_token_ptr += 1

        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values['pixel_values']})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to MPLUGDocOwlTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to MPLUGDocOwlTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

#test the code
'''
from PIL import Image
from transformers.models.mplugdocowl.image_processing_mplugdocowl import MPLUGDocOwlImageProcessor
from transformers import AutoTokenizer, AddedToken
image_processor = MPLUGDocOwlImageProcessor()
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
tokenizer.add_special_tokens({"pad_token": "<pad>"})

#add tokens for shape-adaptive cropping module related textual crop indicators
new_tokens = [f'<crop_img_row{i}_col{j}>' for i in range(10) for j in range(10)]
tokenizer.add_tokens(new_tokens, special_tokens=True)
processor = MPLUGDocOwlProcessor(image_processor, tokenizer)
image = Image.open("/home/dana_aubakirova/test_image.tif")
query = "<|image|>How are you?"
output = processor(images=image, text=query)
breakpoint()
'''