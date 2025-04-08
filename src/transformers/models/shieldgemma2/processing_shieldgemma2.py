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
from collections.abc import Mapping, Sequence
from typing import Optional

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import logging
from ..gemma3.processing_gemma3 import Gemma3Processor, Gemma3ProcessorKwargs


logger = logging.get_logger(__name__)

DEFAULT_SHIELDGEMMA2_POLICIES: Mapping[str, str] = {
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
    ),
}


class ShieldGemma2ProcessorKwargs(Gemma3ProcessorKwargs, total=False):
    policies: Optional[Sequence[str]]
    custom_policies: Optional[Mapping[str, str]]
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "images_kwargs": {
            "do_pan_and_scan": False,
        },
    }


class ShieldGemma2Processor(Gemma3Processor):
    def __init__(
        self, image_processor, tokenizer, chat_template=None, image_seq_length=256, policy_definitions=None, **kwargs
    ):
        """A processor for the ShieldGemma 2 model.

        Args:
            image_processor: The image processor to use, typically a `Gemma3ImageProcessorFast` instance.
            tokenizer: The tokenizer to use, typically a `GemmaTokenizerFast` instance.
            chat_template: The chat template to use with this processor. Typically, this is unset as the processor
                configuration on Hugging Face Hub includes this value already.
            image_seq_length: The number of soft tokens per image. Typically, this is unset as the processor
                configuration on Hugging Face Hub includes this value already.
            policy_definitions: A mapping from policy name to its description in text used as the default policies to
                classify images against. The policy descriptions are included in the text of the prompts generated by
                this processor. Typically, this is unset as the processor configuration on Hugging Face Hub includes
                the base policies ShieldGemma was trained on.
        """
        super().__init__(image_processor, tokenizer, chat_template, image_seq_length, **kwargs)
        if policy_definitions is None:
            self.policy_definitions = DEFAULT_SHIELDGEMMA2_POLICIES
        else:
            self.policy_definitions = policy_definitions

    def __call__(
        self,
        images: ImageInput = None,
        text=None,
        videos=None,
        audio=None,
        **kwargs: Unpack[ShieldGemma2ProcessorKwargs],
    ) -> BatchFeature:
        """Generates a batch of inputs from the provided images.

        ShieldGemma was trained to classify image content for policy compliance using a specific prompt construction.
        This processor generates a batch of such prompts from the provided images by:

        1.  Creating a list of conversations, one for each `<image, policy>` pair;
        2.  Converting these conversations to text using `self.apply_chat_template()`; and
        3.  Encoding the conversations and images using the same techniques as `Gemma3Processor`.

        Args:
            images: A single image or a list of images to include in the batch.
            text: Not supported.
            videos: Not supported.
            audio: Not supported.
            kwargs: An optional dictionary of keyword arguments to configre the
                processor. Possible values include:

                *   `custom_policies`: Additional policy definitions that augment the `self.policy_definitions` passed
                    into the constructor. Note that `custom_policies` that share a key with `self.policy_definitions`
                    will override the policy description
                *   `policies`: (Optional) a list of keys in the joint `self.policy_definitions | custom_policies`
                    dictionary of specific interest for the provided images. If empty or None, prompts will be
                    generated for every key in the joint dictionary.

        Returns:
            A `BatchFeature` continaing `input_ids`, `pixel_values`, etc. where each Tensor is of shape
            `(len(images) * len(policies), )`, and the order within the batch will be
            img1_policy1, ... img1_policyN, ... imgM_policyN.
        """
        del text, videos, audio

        if not images:
            raise ValueError("ShieldGemma 2 needs images to classify")
        elif not isinstance(images, Sequence):
            images = [images]

        if not self.chat_template:
            raise ValueError("ShieldGemma 2 requires the use of a specific chat template")

        # Disable pan and scan
        images_kwargs = kwargs.setdefault("images_kwargs", {})
        if images_kwargs.get("do_pan_and_scan") is True:
            logger.warning_once("ShieldGemma2 does not support pan and scan.")
            images_kwargs["do_pan_and_scan"] = False

        # Enable padding on the batch during tokenization
        text_kwargs = kwargs.setdefault("text_kwargs", {})
        if "padding" not in text_kwargs:
            text_kwargs["padding"] = kwargs.pop("padding", True)
            text_kwargs["padding_side"] = kwargs.pop("padding_side", "left")

        policy_definitions: Mapping[str, str] = {
            **self.policy_definitions,
            **kwargs.get("custom_policies", {}),
        }

        if (policies := kwargs.get("policies")) is None:
            policies = list(policy_definitions.keys())

        # TODO(ryanmullins): Support images from PIL or URLs.
        messages = []
        expanded_images = []
        for img in images:
            for policy in policies:
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": policy_definitions[policy]},
                            ],
                        }
                    ]
                )
                expanded_images.append([img])

        text = self.apply_chat_template(messages, tokenize=False)
        return super().__call__(images=expanded_images, text=text, **kwargs)

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
