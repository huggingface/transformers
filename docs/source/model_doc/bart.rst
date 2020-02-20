Bart
----------------------------------------------------
**DISCLAIMER:** This model is still a work in progress, if you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@sshleifer

The Bart model was `proposed <https://arxiv.org/abs/1910.13461>`_ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer on 29 Oct, 2019.
It is a sequence to sequence model where both encoder and decoder are transformers. The paper also introduces a novel pretraining objective, and demonstrates excellent summarization results.
The authors released their code `here <https://github.com/pytorch/fairseq/tree/master/examples/bart>`_

**Abstract:**

*We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and many other more recent pretraining schemes. We evaluate a number of noising approaches, finding the best performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine translation, with only target language pretraining. We also report ablation experiments that replicate other pretraining schemes within the BART framework, to better measure which factors most influence end-task performance.*
`BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension`


Notes:
- Bart doesn't use :obj:`token_type_ids`, for sequence classification just use BartTokenizer.encode to get the proper splitting.
- Inputs to the decoder are created by BartModel.forward if they are not passed. This is different than some other model APIs.
- Model predictions are intended to be identical to the original implementation. This only works, however, if the string you pass to fairseq.encode starts with a space.

BartModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartModel
    :members: forward


BartForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForMaskedLM
    :members: forward


BartForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForSequenceClassification
    :members: forward

BartConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartConfig
    :members:

Automatic Creation of Decoder Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is enabled by default

.. autofunction:: transformers.modeling_bart._prepare_bart_decoder_inputs
