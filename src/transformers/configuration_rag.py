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

import copy

from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings_to_callable


RAG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/rag-sequence-nq": "TBA",
    "facebook/rag-token-nq": "TBA",
}

RAG_CONFIG_DOC = r"""
    :class:`~transformers.RagConfig` is the configuration class to store the configuration of a `RagModel`.

    Args:
        vocab_size (:obj:`int`, optional, defaults to ``None``):
            Vocabulary size of the underlying generator model.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as an encoder/decoder or not.
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
        dummy (:obj:`bool`, optional, defaults to ``False``)
            Whether to load a ``dummy`` variant of the dataset specified by ``dataset`` argument.
        pretrained_question_encoder_tokenizer_name_or_path: (:obj:`str`, optional, defaults to ``facebook/dpr-question_encoder-single-nq-base``):
            A string specifying the ``question_encoder`` tokenizer to be loaded.
        pretrained_question_encoder_name_or_path: (:obj:`str`, optional, defaults to ``facebook/dpr-question_encoder-single-nq-base``):
            A string specifying the ``question_encoder`` model to be loaded.
        pretrained_generator_tokenizer_name_or_path: (:obj:`str`, optional, defaults to ``facebook/bart-large``):
            A string specifying the ``generator`` tokenizer to be loaded.
        pretrained_generator_name_or_path: (:obj:`str`, optional, defaults to ``facebook/bart-large``):
            A string specifying the ``generator`` model to be loaded.

    Args linked to the tokenizer - they have to be compatible with equivalent parameters of the ``generator``:
        prefix (:obj:`str`, `optional`):
            A specific prompt that should be added at the beginning of each text before calling the model.
        bos_token_id (:obj:`int`, `optional`):
            The id of the `beginning-of-stream` token.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`)"
            The id of the `end-of-stream` token.
        decoder_start_token_id** (:obj:`int`, `optional`):
            If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
        marginalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, `logits`, returned as part of :class:`~transformers.file_utils.Seq2SeqLMOutputWithDocs` are marginalized, yielding
            the shape of :obj:`(batch_size, sequence_length, hidden_size)`. Otherwise we return raw, non-marginalized logits of shape
            :obj:`(batch_size * n_docs, sequence_length, hidden_size)`. ``marginalize`` is set to :obj:`True` during generation. The parameter is
            ignored if ``return_loss`` is set to :obj:`True`.

"""


@add_start_docstrings_to_callable(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = "rag"

    def __init__(
        self,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="compressed",
        index_path=None,
        passages_path=None,
        use_dummy_dataset=False,
        reduce_loss=False,
        label_smoothing=0.0,
        deduplicate=True,  # defaults to True
        num_doc_return_sequences=1,  # defaults to 1
        num_doc_beams=1,  # defaults to 1
        exclude_bos_score=False,
        do_marginalize=False,
        output_retrieved=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert (
            "question_encoder" in kwargs and "generator" in kwargs
        ), "Config has to be initialized with question_encoder and generator config"
        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        from .configuration_auto import AutoConfig

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize
        self.vocab_size = vocab_size
        self.is_encoder_decoder = is_encoder_decoder
        self.prefix = prefix
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset

        self.output_retrieved = output_retrieved

        self.deduplicate = deduplicate
        self.num_doc_return_sequences = num_doc_return_sequences
        self.num_doc_beams = num_doc_beams

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["question_encoder"] = self.question_encoder.to_dict()
        output["generator"] = self.generator.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
