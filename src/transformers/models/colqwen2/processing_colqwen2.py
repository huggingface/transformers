#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/colqwen2/modular_colqwen2.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_colqwen2.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
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


from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available


if is_torch_available():
    import torch


class ColQwen2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ColQwen2Processor(ProcessorMixin):
    r"""
    Constructs a ColQwen2 processor which wraps a Qwen2VLProcessor and special methods to process images and queries, as
    well as to compute the late-interaction retrieval score.

    [`ColQwen2Processor`] offers all the functionalities of [`Qwen2VLProcessor`]. See the [`~Qwen2VLProcessor.__call__`]
    for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        visual_prompt_prefix (`str`, *optional*): A string that gets tokenized and prepended to the image tokens.
        query_prefix (`str`, *optional*): A prefix to be used for the query.
    """

    attributes = ["image_processor", "tokenizer"]

    valid_kwargs = ["chat_template", "visual_prompt_prefix", "query_prefix", "num_image_tokens"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        visual_prompt_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token

        self.visual_prompt_prefix = (
            visual_prompt_prefix
            or "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
        )
        self.query_prefix = query_prefix or "Query: "

        self.tokenizer.padding_side = "left"

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model either (1) one or several texts, either (2) one or several image(s). This method is a custom
        wrapper around the Qwen2VLProcessor's [`~Qwen2VLProcessor.__call__`] method adapted for the ColQwen2 model. It cannot process
        both text and images at the same time.

        When preparing the the text(s), this method forwards the `text` and `kwargs` arguments to Qwen2TokenizerFast's
        [`~Qwen2TokenizerFast.__call__`].
        When preparing the the image(s), this method forwards the `images` and `kwargs` arguments to Qwen2VLImageProcessor's
        [`~Qwen2VLImageProcessor.__call__`].
        Please refer to the doctsring of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            ColQwen2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if text is None and images is None:
            raise ValueError("Either text or images must be provided")
        if text is not None and images is not None:
            raise ValueError("Only one of text or images can be processed at a time")

        if images is not None:
            if is_valid_image(images):
                images = [images]
            elif isinstance(images, list) and is_valid_image(images[0]):
                pass
            elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                raise ValueError("images must be an image, list of images or list of list of images")

            texts_doc = [self.visual_prompt_prefix] * len(images)

            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

            if image_grid_thw is not None:
                merge_length = self.image_processor.merge_size**2
                index = 0
                for i in range(len(texts_doc)):
                    while self.image_token in texts_doc[i]:
                        texts_doc[i] = texts_doc[i].replace(
                            self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                        )
                        index += 1
                    texts_doc[i] = texts_doc[i].replace("<|placeholder|>", self.image_token)

            text_inputs = self.tokenizer(
                texts_doc,
                return_token_type_ids=False,
                **output_kwargs["text_kwargs"],
            )

            return_data = BatchFeature(data={**text_inputs, **image_inputs})

            # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
            offsets = return_data["image_grid_thw"][:, 1] * return_data["image_grid_thw"][:, 2]  # (batch_size,)

            # Split the pixel_values tensor into a list of tensors, one per image
            pixel_values = list(
                torch.split(return_data["pixel_values"], offsets.tolist())
            )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

            # Pad the list of pixel_value tensors to the same length along the sequence dimension
            return_data["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
                pixel_values, batch_first=True
            )  # (batch_size, max_num_patches, pixel_values)

            if return_token_type_ids:
                labels = return_data["input_ids"].masked_fill(return_data["token_type_ids"] == 0, -100)
                return_data.update({"labels": labels})

            return return_data

        elif text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, list) and isinstance(text[0], str)):
                raise ValueError("Text must be a string or a list of strings")

            if suffix is None:
                suffix = self.query_augmentation_token * 10

            texts_query: List[str] = []

            for query in text:
                augmented_query = self.query_prefix + query + suffix
                texts_query.append(augmented_query)

            batch_query = self.tokenizer(
                texts_query,
                return_token_type_ids=False,
                **output_kwargs["text_kwargs"],
            )

            return batch_query

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
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.

        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    def process_images(
        self,
        images: ImageInput = None,
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several image(s). This method is a wrapper around the `__call__` method of the ColQwen2Processor's
        [`ColQwen2Processor.__call__`].

        This method forwards the `images` and `kwargs` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`].

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        return self.__call__(images=images, **kwargs)

    def process_queries(
        self,
        text: Union[TextInput, List[TextInput]],
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several texts. This method is a wrapper around the `__call__` method of the ColQwen2Processor's
        [`ColQwen2Processor.__call__`].

        This method forwards the `text` and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`].

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
        """
        return self.__call__(text=text, **kwargs)

    def score_retrieval(
        self,
        query_embeddings: Union["torch.Tensor", List["torch.Tensor"]],
        passage_embeddings: Union["torch.Tensor", List["torch.Tensor"]],
        batch_size: int = 128,
        output_dtype: Optional["torch.dtype"] = None,
        output_device: Union["torch.device", str] = "cpu",
    ) -> "torch.Tensor":
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColQwen2, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            query_embeddings (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            passage_embeddings (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            output_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`): The dtype of the output tensor.
                If `None`, the dtype of the input embeddings is used.
            output_device (`torch.device` or `str`, *optional*, defaults to "cpu"): The device of the output tensor.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
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

        scores: List[torch.Tensor] = []

        for i in range(0, len(query_embeddings), batch_size):
            batch_scores: List[torch.Tensor] = []
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


__all__ = ["ColQwen2Processor"]
