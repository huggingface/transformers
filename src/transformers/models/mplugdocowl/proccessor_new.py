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


from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

import numpy as np
from PIL import Image

grid_dict = {
    'grid_1':[
        (1,1)],
    'grid_4':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1)],
    'grid_9':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)],
    'grid_3x3':[
        (3,3)],
    'grid_20':[
        (1, 1), 
        (1, 2), (2, 1), 
        (1, 3), (3, 1), (1, 4), (2, 2), (4, 1), 
        (1, 5), (5, 1), 
        (1, 6), (2, 3), (3, 2), (6, 1), 
        (1, 7), (7, 1), 
        (1, 8), (2, 4), (4, 2), (8, 1), 
        (1, 9), (3, 3), (9, 1), 
        (1, 10), (2, 5), (5, 2), (10, 1), 
        (1, 11), (11, 1), 
        (2, 6), (3, 4), (4, 3), (6, 2), 
        (2, 7), (7, 2), 
        (3, 5), (5, 3), 
        (2, 8), (4, 4), (8, 2), 
        (2, 9), (3, 6), (6, 3), (9, 2), 
        (2, 10), (4, 5), (5, 4), (10, 2)]
}


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
    def __init__(self, image_size, anchors, interpolation=Image.BILINEAR, antialias=False):
        # xyxy
        self.anchors = np.array(
            [[0, 0, anchor[1] * image_size[1], anchor[0] * image_size[0]]
             for anchor in anchors]
        )
        self.anchor_areas = box_area(self.anchors)
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img, skip_resize=False):
        # Resize image based on selected anchor
        input_image_size = (img.height, img.width)
        selected_anchor = anchor_rank(self.anchors, self.anchor_areas, input_image_size)
        target_size = self.anchors[selected_anchor][2:].astype(int)  # target width, height
        if skip_resize:
            return selected_anchor  # For debug purposes
        resized_img = img.resize((target_size[0], target_size[1]), resample=self.interpolation)
        return resized_img, selected_anchor

    def __repr__(self):
        detail = f"AnchorResize(image_size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation}, antialias={self.antialias})"
        return detail

class ShapeAdaptiveImageProcessor(ProcessorMixin):
    def __init__(self, image_size=224, anchors='grid_9', grid_dict=grid_dict, add_global_img = True,add_textual_crop_indicator=False):
        if grid_dict is None:
            grid_dict = {'grid_9': [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0)]}  # Define your grid_dict appropriately
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.anchors = [tuple(_) for _ in grid_dict[anchors]]
        self.anchor_max = max(max(_) for _ in self.anchors)
        self.anchors_areas = [box_area(np.array([[0, 0, w*self.image_size[1], h*self.image_size[0]]])) for w, h in self.anchors]
        self.add_global_img = add_global_img

    def _process_image(self, images):
            new_images = []
            new_patch_position = []
            num_image_mult = []

            for image in images:
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype('uint8'), 'RGB')

                # Resize the image according to the selected anchor
                image_np = np.array(image)
                selected_anchor = self.anchor_rank(np.array(self.anchors), np.array(self.anchors_areas), image_np.shape[:2])
                anchor_size = self.anchors[selected_anchor]
                new_size = (int(anchor_size[1] * self.image_size[1]), int(anchor_size[0] * self.image_size[0]))
                resized_image = np.array(image.resize(new_size, Image.BICUBIC))

                # Normalize the image (example normalization values)
                #resized_image = resized_image / 255.0
                #resized_image = (resized_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                # Reshape the image
                num_h, num_w = anchor_size
                image_input = resized_image.reshape((num_h, self.image_size[0], num_w, self.image_size[1], 3))
                image_input = image_input.transpose(0, 2, 4, 1, 3).reshape(-1, self.image_size[0], self.image_size[1], 3)

                if self.add_global_img:
                    global_image = np.array(image)
                    #global_image = (global_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    global_image = global_image[np.newaxis, ...]
                    image_input = np.concatenate([global_image, image_input], axis=0)

                anchor = self.anchors[selected_anchor]  # w,h
                patch_position = np.concatenate([
                    np.repeat(np.arange(anchor[0])[:, np.newaxis], anchor[1], axis=1)[:, :, np.newaxis],
                    np.repeat(np.arange(anchor[1])[np.newaxis, :], anchor[0], axis=0)[:, :, np.newaxis]
                ], axis=2)
                patch_position = patch_position.reshape(-1, 2)  # num_patch, (ph, pw)

                if self.add_global_img:
                    patch_position = np.concatenate([np.ones((1, 2)) * self.anchor_max, patch_position], axis=0)

                new_images.append(image_input)
                new_patch_position.append(patch_position)
                num_image_mult.append(patch_position.shape[0])

            new_images = np.concatenate(new_images, axis=0)
            new_patch_position = np.concatenate(new_patch_position, axis=0)
            return new_images, new_patch_position, num_image_mult
    
    def __call__(self, images, return_tensors=None):
        
        processed_images, patch_positions, num_image_mult = self._process_image(images)
        
        #if return_tensors == "pt":
            #processed_images = torch.tensor(processed_images).permute(0, 3, 1, 2)
       # if return_tensors == "np":
        processed_images = np.array(processed_images).transpose(0, 3, 1, 2)
        
        return {"pixel_values": processed_images, "patch_positions": patch_positions, "num_image_mult": num_image_mult}


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
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("AutoTokenizer")#, "AutoTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        do_resize: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
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
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        else:
            pixel_values = None
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

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






