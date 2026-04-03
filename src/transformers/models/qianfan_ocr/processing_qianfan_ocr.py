# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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


from ..internvl.processing_internvl import InternVLProcessor
from ...image_utils import ImageInput
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...image_processing_utils import BatchFeature


class QianfanOCRProcessor(InternVLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_seq_length: int = 256,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=None,
            image_seq_length=image_seq_length,
            chat_template=chat_template,
            **kwargs,
        )
        # QianfanOCR has no video support, but InternVLProcessor code accesses
        # self.video_processor. ProcessorMixin.__init__ skips setting it because
        # video_processor is absent from QianfanOCRProcessor.__init__ signature,
        # so we set it explicitly here.
        self.video_processor = None

    def apply_chat_template(self, conversations, **kwargs):
        # Normalize str content to list format so that processing_utils can
        # iterate over content items when tokenize=True, e.g.:
        #   {"role": "user", "content": "hello"}
        #   -> {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        # The Jinja template handles both str and list content natively.
        # Works for both single conversations (list of dicts) and batches
        # (list of list of dicts).
        def _normalize_single(convs):
            normalized = []
            for message in convs:
                if isinstance(message.get("content"), str):
                    message = {**message, "content": [{"type": "text", "text": message["content"]}]}
                normalized.append(message)
            return normalized

        if isinstance(conversations, list) and conversations:
            if isinstance(conversations[0], dict):
                conversations = _normalize_single(conversations)
            elif isinstance(conversations[0], list):
                conversations = [_normalize_single(conv) for conv in conversations]
        return super().apply_chat_template(conversations, **kwargs)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs,
    ) -> BatchFeature:
        # QianfanOCR has no video or audio support. Drop those keys if they arrive
        # from apply_chat_template's internal self(...) call, so that InternVLProcessor
        # never sees them (video=non-None would crash; duplicate videos= would error).
        kwargs.pop("videos", None)
        kwargs.pop("audio", None)
        return super().__call__(images=images, text=text, videos=None, **kwargs)


__all__ = ["QianfanOCRProcessor"]
