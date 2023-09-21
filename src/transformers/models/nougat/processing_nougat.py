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
Processor class for Nougat.
"""

from ...processing_utils import ProcessorMixin


class NougatProcessor(ProcessorMixin):
    r"""
    Constructs a Nougat processor which wraps a Nougat image processor and a Nougat tokenizer into a single processor.

    [`NougatProcessor`] offers all the functionalities of [`NougatImageProcessor`] and [`NougatTokenizerFast`]. See the
    [`~NougatProcessor.__call__`] and [`~NougatProcessor.decode`] for more information.

    Args:
        image_processor ([`NougatImageProcessor`]):
            An instance of [`NougatImageProcessor`]. The image processor is a required input.
        tokenizer ([`NougatTokenizerFast`]):
            An instance of [`NougatTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~NougatProcessor.as_target_processor`] this method forwards all its arguments to NougatTokenizer's
        [`~NougatTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_generation(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.post_process_generation`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.post_process_generation(*args, **kwargs)
