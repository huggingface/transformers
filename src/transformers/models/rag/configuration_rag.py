# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""RAG model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="")
@strict
class RagConfig(PreTrainedConfig):
    r"""
    prefix (`str`, *optional*):
        A string prefix prepended to every input before passing to the generator model.
    title_sep (`str`, *optional*, defaults to  `" / "`):
        Separator inserted between the title and the text of the retrieved document when calling [`RagRetriever`].
    doc_sep (`str`, *optional*, defaults to  `" // "`):
        Separator inserted between the text of the retrieved document and the original input when calling
        [`RagRetriever`].
    n_docs (`int`, *optional*, defaults to 5):
        Number of documents to retrieve.
    max_combined_length (`int`, *optional*, defaults to 300):
        Max length of contextualized input returned by [`~RagRetriever.__call__`].
    retrieval_vector_size (`int`, *optional*, defaults to 768):
        Dimensionality of the document embeddings indexed by [`RagRetriever`].
    retrieval_batch_size (`int`, *optional*, defaults to 8):
        Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
        [`RagRetriever`].
    dataset (`str`, *optional*, defaults to `"wiki_dpr"`):
        A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
        using `datasets.list_datasets()`).
    dataset_split (`str`, *optional*, defaults to `"train"`):
        Which split of the `dataset` to load.
    index_name (`str`, *optional*, defaults to `"compressed"`):
        The index name of the index associated with the `dataset`. One can choose between `"legacy"`, `"exact"` and
        `"compressed"`.
    index_path (`str`, *optional*):
        The path to the serialized faiss index on disk.
    passages_path (`str`, *optional*):
        A path to text passages compatible with the faiss index. Required if using
        [`~models.rag.retrieval_rag.LegacyIndex`]
    use_dummy_dataset (`bool`, *optional*, defaults to `False`):
        Whether to load a "dummy" variant of the dataset specified by `dataset`.
    reduce_loss (`bool`, *optional*, defaults to `False`):
        Whether or not to reduce the NLL loss using the `torch.Tensor.sum` operation.
    label_smoothing (`float`, *optional*, defaults to 0.0):
        Only relevant if `return_loss` is set to `True`. Controls the `epsilon` parameter value for label smoothing
        in the loss calculation. If set to 0, no label smoothing is performed.
    do_deduplication (`bool`, *optional*, defaults to `True`):
        Whether or not to deduplicate the generations from different context documents for a given input. Has to be
        set to `False` if used while training with distributed backend.
    exclude_bos_score (`bool`, *optional*, defaults to `False`):
        Whether or not to disregard the BOS token when computing the loss.
    do_marginalize (`bool`, *optional*, defaults to `False`):
        If `True`, the logits are marginalized over all documents by making use of
        `torch.nn.functional.log_softmax`.
    output_retrieved (`bool`, *optional*, defaults to `False`):
        If set to `True`, `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
        `context_attention_mask` are returned. See returned tensors for more detail.
    dataset_revision (`str`, *optional*,):
        The revision (commit hash, tag, or branch) of the Hugging Face dataset used for retrieval.
    """

    model_type = "rag"
    has_no_defaults_at_init = True

    vocab_size: int | None = None
    is_encoder_decoder: bool = True
    prefix: str | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    decoder_start_token_id: int | None = None
    title_sep: str = " / "
    doc_sep: str = " // "
    n_docs: int = 5
    max_combined_length: int = 300
    retrieval_vector_size: int = 768
    retrieval_batch_size: int = 8
    dataset: str = "wiki_dpr"
    dataset_split: str = "train"
    index_name: str = "compressed"
    index_path: str | None = None
    passages_path: str | None = None
    use_dummy_dataset: bool = False
    reduce_loss: bool = False
    label_smoothing: float = 0.0
    do_deduplication: bool = True
    exclude_bos_score: bool = False
    do_marginalize: bool = False
    output_retrieved: bool = False
    use_cache: bool = True
    dataset_revision: str | None = None

    def __post_init__(self, **kwargs):
        if "question_encoder" not in kwargs or "generator" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because not both `question_encoder` and"
                f" `generator` sub-configurations are passed, but only {kwargs}"
            )

        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        super().__post_init__(**kwargs)

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PreTrainedConfig, generator_config: PreTrainedConfig, **kwargs
    ) -> PreTrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)


__all__ = ["RagConfig"]
