# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from typing import Mapping, Sequence

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import logging
from ..gemma3.processing_gemma3 import Gemma3Processor, Gemma3ProcessorKwargs

logger = logging.get_logger(__name__)

SHIELDGEMMA2_POLICIES: Mapping[str, str] = {
    "dangerous": (
        "No Dangerous Content: The image shall not contain content that"
        " facilitates or encourages activities that could cause real-world harm"
        " (e.g., building firearms and explosive devices, promotion of"
        " terrorism, instructions for suicide)."
    ),
    "sexual": (
        "No Sexually Explicit content: The image shall not contain content that"
        " depicts explicit or graphic sexual acts (e.g., pornography, erotic"
        " nudity, depictions of rape or sexual assault)."
    ),
    "violence": (
        "No Violence/Gore content: The image shall not contain content that"
        " depicts shocking, sensational, or gratuitous violence (e.g.,"
        " excessive blood and gore, gratuitous violence against animals,"
        " extreme injury or moment of death)."
    )
}


class ShieldGemma2Processor(Gemma3Processor):

    def _disable_pan_and_scan(self, kwargs: dict[str, dict[str, bool]]):
        if kwargs.get("images_kwargs", {}).get("do_pan_and_scan") is True:
            logger.warning_once("ShieldGemma2 does not support pan and scan.")
            kwargs["images_kwargs"]["do_pan_and_scan"] = False

    def __call__(
        self,
        images: ImageInput = None,
        text=None,
        videos=None,
        audio=None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        del text, videos, audio

        if not images:
            raise ValueError("ShieldGemma 2 needs images to classify")
        elif not isinstance(images, Sequence):
            images = [images]

        if not self.chat_template:
            raise ValueError("ShieldGemma 2 requires the use of a specific chat template")

        self._disable_pan_and_scan(kwargs)
        kwargs["padding"] = True
        kwargs["padding_side"] = "left"

        policies = kwargs.get("policies", list(SHIELDGEMMA2_POLICIES.keys()))
        policy_definitions = {
            **SHIELDGEMMA2_POLICIES,
            **kwargs.get("custom_policies", {}),
        }

        messages = []
        combinatoric_images = []
        for img in images:
            for policy in policies:
                messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": policy_definitions[policy]}
                        ]
                    }
                ])
                combinatoric_images.append([img])

        text = self.apply_chat_template(messages, tokenize=False)
        return super().__call__(images=combinatoric_images, text=text, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["ShieldGemma2Processor"]
