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

from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class ColPaliProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
            "return_mm_token_type_ids": False,
            "return_text_replacement_offsets": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ColPaliProcessor(PaliGemmaProcessor):
    valid_processor_kwargs = ColPaliProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        visual_prompt_prefix: str = "Describe the image.",
        query_prefix: str = "Question: ",
    ):
        r"""
        visual_prompt_prefix (`str`, *optional*, defaults to `"Describe the image."`):
            A string that gets tokenized and prepended to the image tokens.
        query_prefix (`str`, *optional*, defaults to `"Question: "`):
            A prefix to be used for the query.
        """
        self.visual_prompt_prefix = visual_prompt_prefix
        self.query_prefix = query_prefix
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.

        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ColPaliProcessorKwargs],
    ) -> BatchFeature:
        kwargs["return_token_type_ids"] = True
        output_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        if text is not None:
            # Query mode: augment text before base class tokenizes it
            if suffix is None:
                suffix = self.query_augmentation_token * 10

            text = [f"{self.tokenizer.bos_token}{self.query_prefix}{sample}{suffix}\n" for sample in text]
            output_kwargs["text_kwargs"].setdefault("max_length", 50)

        model_inputs = super().__call__(images=images, text=text, **output_kwargs)
        if images is not None:
            model_inputs["labels"] = model_inputs["input_ids"].masked_fill(model_inputs["token_type_ids"] == 0, -100)
        return model_inputs

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        ProcessorMixin.validate_inputs(images=images, text=text)
        if text is None and images is None:
            raise ValueError("Either text or images must be provided")
        if text is not None and images is not None:
            raise ValueError("Only one of text or images can be processed at a time")

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        images, text, *_ = ProcessorMixin.prepare_inputs_layout(images=images, text=text, **kwargs)
        if images is not None:
            images = make_flat_list_of_images(images)
            text = [f"{self.image_token}{self.tokenizer.bos_token}{self.visual_prompt_prefix}\n" for _ in len(images)]
        return images, text, videos, audio

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        return self.image_token * self.image_seq_length

    def process_images(
        self,
        images: ImageInput | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        This method forwards the `images` and `kwargs` arguments to ColPaliProcessor's [`~ColPaliProcessor.__call__`].
        """
        return self.__call__(images=images, **kwargs)

    def process_queries(
        self,
        text: TextInput | list[TextInput],
        **kwargs,
    ) -> BatchFeature:
        """
        This method forwards the `text` and `kwargs` arguments to ColPaliProcessor's [`~ColPaliProcessor.__call__`].
        """
        return self.__call__(text=text, **kwargs)

    def score_retrieval(
        self,
        query_embeddings,
        passage_embeddings,
        batch_size: int = 128,
        output_dtype=None,
        output_device="cpu",
    ):
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings and passage embeddings.
        """
        if len(query_embeddings) == 0:
            raise ValueError("No queries provided")
        if len(passage_embeddings) == 0:
            raise ValueError("No passages provided")
        if query_embeddings[0].device != passage_embeddings[0].device:
            raise ValueError("Queries and passages must be on the same device")
        if query_embeddings[0].dtype != passage_embeddings[0].dtype:
            raise ValueError("Queries and passages must have the same dtype")
        if output_dtype is None:
            output_dtype = query_embeddings[0].dtype
        scores = []
        for i in range(0, len(query_embeddings), batch_size):
            batch_scores = []
            batch_queries = torch.nn.utils.rnn.pad_sequence(
                query_embeddings[i : i + batch_size], batch_first=True, padding_value=0
            )
            for j in range(0, len(passage_embeddings), batch_size):
                batch_passages = torch.nn.utils.rnn.pad_sequence(
                    passage_embeddings[j : j + batch_size], batch_first=True, padding_value=0
                )
                batch_scores.append(
                    torch.einsum("bnd,csd->bcns", batch_queries, batch_passages).max(dim=3)[0].sum(dim=2)
                )
            scores.append(torch.cat(batch_scores, dim=1).to(output_dtype).to(output_device))
        return torch.cat(scores, dim=0)


__all__ = [
    "ColPaliProcessor",
]
