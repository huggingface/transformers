# coding=utf-8
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
"""
Processor class for SAMHQ.
"""

from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

from ...image_utils import ImageInput, VideoInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import AudioInput, BatchEncoding, PreTokenizedInput, TextInput
from ...utils import is_tf_available, is_torch_available



if is_torch_available():
    import torch



class SamHQImagesKwargs(ImagesKwargs):
    segmentation_maps: Optional[ImageInput]
    input_points: Optional[List[List[float]]]
    input_labels: Optional[List[List[int]]]
    input_boxes: Optional[List[List[List[float]]]]
    point_pad_value: Optional[int]


class SamHQProcessorKwargs(ProcessingKwargs,total=False):
    images_kwargs: SamHQImagesKwargs
    _defaults = {
        "images_kwargs": {
            "point_pad_value": -10,
            
        }
    }


class SamHQProcessor(ProcessorMixin):
    r"""
    Constructs a SAMHQ processor which wraps a SAMHQ image processor and an 2D points & Bounding boxes processor into a
    single processor.
    
    """
    attributes = ["image_processor"]
    image_processor_class = "SamImageProcessor"

    optional_call_args = [
        "segmentation_maps",
        "input_points",
        "input_labels",
        "input_boxes",
    ]

    
    def __init__(self,image_processor):
        super().__init__(image_processor)
        self.target_size = self.image_processor.size["longest_edge"]

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        # The following is to capture `segmentation_maps`, `input_points`, `input_labels` and `input_boxes`
        # arguments that may be passed as a positional argument.
        # See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details,
        # or this conversation for more context:
        # https://github.com/huggingface/transformers/pull/32544#discussion_r1720208116
        # This behavior is only needed for backward compatibility and will be removed in future versions.
        *args,  # to be deprecated
        text: Optional[Union[TextInput,PreTokenizedInput,List[TextInput],List[PreTokenizedInput]]] = None,
        audio: Optional[AudioInput] = None,
        video: Optional[VideoInput] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.
        """
        output_kwargs = self._merge_kwargs(
            SamHQProcessorKwargs,
            tokenizer_init_kwargs={},
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )
        
        input_points = output_kwargs["images_kwargs"].pop("input_points",None)
        input_labels = output_kwargs["images_kwargs"].pop("input_labels",None)
        input_boxes = output_kwargs["images_kwargs"].pop("input_boxes",None)

        encoding_image_processor = self.image_processor(
            images,
            **output_kwargs["images_kwargs"],
        )

        
        original_sizes = encoding_image_processor["original_sizes"]

        if hasattr(original_sizes,"numpy"):
            original_sizes = original_sizes.numpy()

        input_points,input_labels,input_boxes = self._check_and_preprocess_points(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )

        encoding_image_processor = self._normalize_and_convert(
            encoding_image_processor,
            original_sizes,
            input_points=input_points,                                                                                              
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors=output_kwargs["common_kwargs"].get("return_tensors"),
            point_pad_value=output_kwargs["images_kwargs"].get("point_pad_value"),
        )

        return encoding_image_processor

    def _normalize_and_convert(
        self,
        encoding_image_processor,
        original_sizes,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors="pt",
        point_pad_value=-10,
    ):
        """
        Normalize and convert the image processor output to the expected format.
        """
        if input_points is not None:
            if len(original_sizes) != len(input_points):
                input_points = [
                    self._normalize_coordinates(self.target_size,point,original_sizes[0])
                    for point in input_points
                ]
            else:
                input_points = [
                    self._normalize_coordinates(self.target_size,point,original_size)
                    for point,original_size in zip(input_points,original_sizes)
                ]

        if not all(point.shape == input_points[0].shape for point in input_points):
            if input_labels is not None:
                input_points,input_labels = self._pad_points_and_labels(
                    input_points,input_labels,point_pad_value
                )

        input_points = np.array(input_points)

        if input_labels is not None:
            input_labels = np.array(input_labels)   

        if input_boxes is not None:
            if len(original_sizes) != len(input_boxes):
                input_boxes = [
                    self._normalize_coordinates(self.target_size,box,original_sizes[0],is_bounding_box=True)
                    for box in input_boxes
                ]
            else:
                input_boxes = [
                    self._normalize_coordinates(self.target_size,box,original_size,is_bounding_box=True)
                    for box,original_size in zip(input_boxes,original_sizes)
                ]
            input_boxes = np.array(input_boxes)

        if input_boxes is not None:
            if return_tensors == "pt":
                input_boxes = torch.from_numpy(input_boxes)
                input_boxes = input_boxes.unsqueeze(1) if len(input_boxes.shape) != 3 else input_boxes
            encoding_image_processor.update({"input_boxes":input_boxes})
        if input_points is not None:
            if return_tensors == "pt":
                input_points = torch.from_numpy(input_points)
                input_points = input_points.unsqueeze(1) if len(input_points.shape) != 4 else input_points
            encoding_image_processor.update({"input_points":input_points})
        if input_labels is not None:
            if return_tensors == "pt":
                input_labels = torch.from_numpy(input_labels)
                input_labels = input_labels.unsqueeze(1) if len(input_labels.shape) != 3 else input_labels
            encoding_image_processor.update({"input_labels":input_labels})

        return encoding_image_processor
    
    def _pad_points_and_labels(self,input_points,input_labels,point_pad_value):
        r"""
        The method pads the 2D points and labels to the maximum number of points in the batch.
        """
        expected_nb_points = max([point.shape[0] for point in input_points])
        processed_input_points = []
        for i,point in enumerate(input_points):
            if point.shape[0] != expected_nb_points:
                point = np.concatenate(
                    [point,np.zeros((expected_nb_points-point.shape[0],2))+point_pad_value],axis=0
                )
                input_labels[i] = np.append(input_labels[i],[point_pad_value])
            processed_input_points.append(point)
        input_points = processed_input_points
        return input_points,input_labels
    
    def _normalize_coordinates(self,target_size:int,coords:np.ndarray,original_size,is_bounding_box=False)->np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H,W) format.
        """
        old_h,old_w = original_size
        new_h,new_w = self.image_processor._get_preprocess_shape(original_size,longest_edge=target_size)
        coords = deepcopy(coords).astype(float)

        if is_bounding_box:
            coords = coords.reshape(-1,2,2)

        coords[...,0] = coords[...,0]*(new_w/old_w)
        coords[...,1] = coords[...,1]*(new_h/old_h)

        if is_bounding_box:
            coords = coords.reshape(-1,4)

        return coords
    

    def _check_and_preprocess_points(
        self,
        input_points=None,
        input_labels=None,
        input_boxes=None,
    ):
        r"""
        Check and preprocesses the 2D points, labels and bounding boxes. It checks if the input is valid and if they
        are, it converts the coordinates of the points and bounding boxes. If a user passes directly a `torch.Tensor`,
        it is converted to a `numpy.ndarray` and then to a `list`.
        """
        if input_points is not None:
            if hasattr(input_points,"numpy"):
                input_points = input_points.numpy().tolist()

            if not isinstance(input_points,list) or not isinstance(input_points[0],list):
                raise ValueError("Input points must be a list of list of floating points.")
            input_points = [np.array(input_point) for input_point in input_points]
        else:
            input_points = None

        if input_labels is not None:
            if hasattr(input_labels,"numpy"):
                input_labels = input_labels.numpy().tolist()

            if not isinstance(input_labels,list) or not isinstance(input_labels[0],list):
                raise ValueError("Input labels must be a list of list integers.")
            input_labels = [np.array(label) for label in input_labels]
        else:
            input_labels = None

        if input_boxes is not None:
            if hasattr(input_boxes,"numpy"):
                input_boxes = input_boxes.numpy().tolist()

            if (
                not isinstance(input_boxes,list)
                or not isinstance(input_boxes[0],list)
                or not isinstance(input_boxes[0][0],list)
            ):
                raise ValueError("Input boxes must be a list of list of list of floating points.")
            input_boxes = [np.array(box).astype(np.float32) for box in input_boxes]
        else:
            input_boxes = None  

        return input_points,input_labels,input_boxes
    
    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))
    
    def post_process_masks(self,*args,**kwargs):
        return self.image_processor.post_process_masks(*args,**kwargs)
    

__all__ = ["SamHQProcessor"]


        
            
                
                
        
                
            
        
        
        
        