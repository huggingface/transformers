# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Image/Text processor class for OWL-ViT
"""


from ...processing_utils import ProcessorMixin


class Owlv2Processor(ProcessorMixin):
    r"""
    Constructs an Owlv2 processor which wraps [`Owlv2ImageProcessor`] and [`CLIPTokenizer`]/[`CLIPTokenizerFast`] into
    a single processor that interits both the image processor and tokenizer functionalities. See the
    [`~OwlViTProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more information.

    Args:
        image_processor ([`Owlv2ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Owlv2ImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_object_detection with OWLViT->OWLv2
    def post_process_object_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process_object_detection`]. Please refer
        to the docstring of this method for more information.
        """
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_image_guided_detection with OWLViT->OWLv2
    def post_process_image_guided_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process_one_shot_object_detection`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_image_guided_detection(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.batch_decode
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
