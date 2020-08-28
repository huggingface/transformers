# coding=utf-8
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
""" RAG model configuration """


from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings_to_callable


RAG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/rag-sequence-nq": "TBA",
    "facebook/rag-token-nq": "TBA",
}

RAG_CONFIG_DOC = r"""
    :class:`~transformers.RagConfig` is the configuration class to store the configuration of a `RagModel`.

    Args:
        title_sep (:obj:`str`, optional, defaults to  ``" / "``):
            Separator inserted between the title and the text of the retrieved document when running
            `:func:`~transformers.RagModel.contextualize``.
        doc_sep (:obj:`str`, optional, defaults to  ``" // "``):
            Separator inserted between the the text of the retrieved document and the original input when running
            `:func:`~transformers.RagModel.contextualize``.
        n_docs (:obj:`int`, optional, defaults to ``5``):
            Number of retrieved docs.
        max_combined_length (:int:`bool`, optional, defaults to ``300``):
            Max length of contextualized input returned by `:func:`~transformers.RagModel.contextualize``.
        retrieval_vector_size (:obj:`int`, optional, defaults to ``768``):
            Dimensionality of the document embeddings indexed by the ``retriever``.
        retrieval_batch_size (:obj:`int`, optional, defaults to ``8``):
            Retrieval batch size - the number of queries issues concurrently to the faiss index excapsulated
            by the ``retriever``.
        retriever_type (:obj:`str`, optional, defaults to ``hf_retriever``):
            A type of index encapsulated by the ``retriever``. Possible options include:

                - ``hf_retriever`` - and index build for an instance of :class:`~nlp.Datasets`
                - ``legacy_retriever`` - an index build with the native DPR implementation (see https://github.com/facebookresearch/DPR for details).
        dataset (:obj:`str`, optional, defaults to ``wiki_dpr``):
            A datatset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids with ``nlp.list_datasets()``).
        dataset_split (:obj:`str`, optional, defaults to ``train``)
            Which split of the ``dataset`` to load.
        index_name (:obj:`str`, optional, defaults to ``train``)
            The index_name of the index associated with the ``dataset``.
        index_path (:obj:`str`, optional, defaults to ``None``)
            The path to the serialized faiss index on disk.
        passages_path: (:obj:`str`, optional, defaults to ``None``):
            A path to text passages compatible with the faiss index. Required if using :class:`~transformers.retrieval_rag.LegacyIndex`
        pretrained_question_encoder_tokenizer_name_or_path: (:obj:`str`, optional, defaults to ``facebook/dpr-question_encoder-single-nq-base``):
            A string specifying the ``question_encoder`` tokenizer to be loaded.
        pretrained_question_encoder_name_or_path: (:obj:`str`, optional, defaults to ``facebook/dpr-question_encoder-single-nq-base``):
            A string specifying the ``question_encoder`` model to be loaded.
        pretrained_generator_tokenizer_name_or_path: (:obj:`str`, optional, defaults to ``facebook/bart-large``):
            A string specifying the ``generator`` tokenizer to be loaded.
        pretrained_generator_name_or_path: (:obj:`str`, optional, defaults to ``facebook/bart-large``):
            A string specifying the ``generator`` model to be loaded.
"""


@add_start_docstrings_to_callable(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = "rag"

    def __init__(
        self,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        retriever_type="hf_retriever",
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="embeddings",
        index_path=None,
        passages_path=None,
        pretrained_question_encoder_tokenizer_name_or_path="facebook/dpr-question_encoder-single-nq-base",
        pretrained_question_encoder_name_or_path="facebook/dpr-question_encoder-single-nq-base",
        pretrained_generator_tokenizer_name_or_path="facebook/bart-large",
        pretrained_generator_name_or_path="facebook/bart-large",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.retriever_type = retriever_type

        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path

        self.pretrained_question_encoder_tokenizer_name_or_path = pretrained_question_encoder_tokenizer_name_or_path
        self.pretrained_question_encoder_name_or_path = pretrained_question_encoder_name_or_path
        self.pretrained_generator_tokenizer_name_or_path = pretrained_generator_tokenizer_name_or_path
        self.pretrained_generator_name_or_path = pretrained_generator_name_or_path
