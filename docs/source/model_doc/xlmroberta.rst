XLM-RoBERTa
------------------------------------------

The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__
by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán,
Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's RoBERTa model released in 2019.
It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

The abstract from the paper is the following:

*This paper shows that pretraining multilingual language models at scale leads to significant performance gains for
a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred
languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly
outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy
on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on
low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model.
We also present a detailed empirical evaluation of the key factors that are required to achieve these gains,
including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and
low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling
without sacrificing per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE
and XNLI benchmarks. We will make XLM-R code, data, and models publicly available.*

Tips:

- XLM-R is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
  not require `lang` tensors to understand which language is used, and should be able to determine the correct
  language from the input ids.
- This implementation is the same as RoBERTa. Refer to the `documentation of RoBERTa <./roberta.html>`__ for usage
  examples as well as the information relative to the inputs and outputs.

XLMRobertaConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaConfig
    :members:


XLMRobertaTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


XLMRobertaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaModel
    :members:


XLMRobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForMaskedLM
    :members:


XLMRobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForSequenceClassification
    :members:


XLMRobertaForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForMultipleChoice
    :members:


XLMRobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForTokenClassification
    :members:


TFXLMRobertaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMRobertaModel
    :members:


TFXLMRobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMRobertaForMaskedLM
    :members:


TFXLMRobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMRobertaForSequenceClassification
    :members:


TFXLMRobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMRobertaForTokenClassification
    :members:
