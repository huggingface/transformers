Pretrained models
================================================

Here is the full list of the currently provided pretrained models together with a short presentation of each model.

+===============+============================================================+===========================+ 
| Architecture  | Shortcut name                                              | Details of the model      |
+===============+============================================================+===========================+ 
|               | ``bert-base-uncased``                                      | 12-layer, 768-hidden, 12-heads, 110M parameters
|               |                                                            | Trained on lower-cased English text                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-large-uncased``                                     | 24-layer, 1024-hidden, 16-heads, 340M parameters
|               |                                                            | Trained on lower-cased English text                  |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-base-cased``                                        | 12-layer, 768-hidden, 12-heads, 110M parameters
|               |                                                            | Trained on cased English text                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-large-cased``                                       | 24-layer, 1024-hidden, 16-heads, 340M parameters                  |
|               |                                                            | Trained on cased English text                  |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-base-multilingual-uncased``                         | (Original, not recommended) 12-layer, 768-hidden, 12-heads, 110M parameters
|               |                                                            | Trained on lower-cased text in the top 102 languages with the largest Wikipedias
|               |                                                            | (see `details <https://github.com/google-research/bert/blob/master/multilingual.md>`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-base-multilingual-cased``                           | (New, **recommended**) 12-layer, 768-hidden, 12-heads, 110M parameters                  |
|               |                                                            | Trained on cased text in the top 104 languages with the largest Wikipedias
|               |                                                            | (see `details <https://github.com/google-research/bert/blob/master/multilingual.md>`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|    BERT       | ``bert-base-chinese``                                      | 12-layer, 768-hidden, 12-heads, 110M parameters                  |
|               |                                                            | Trained on cased Chinese Simplified and Traditional text |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-base-german-cased``                                 | 12-layer, 768-hidden, 12-heads, 110M parameters                  |
|               |                                                            | Trained on cased German text by Deepset.ai |
|               |                                                            | (see `details on deepset.ai website <https://deepset.ai/german-bert>`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-large-uncased-whole-word-masking``                  | 24-layer, 1024-hidden, 16-heads, 340M parameters                  |
|               |                                                            | Trained on lower-cased English text using Whole-Word-Masking                  |
|               |                                                            | (see `details <https://github.com/google-research/bert/#bert>`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-large-cased-whole-word-masking``                    | 24-layer, 1024-hidden, 16-heads, 340M parameters                  |
|               |                                                            | Trained on cased English text using Whole-Word-Masking                  |
|               |                                                            | (see `details <https://github.com/google-research/bert/#bert>`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-large-uncased-whole-word-masking-finetuned-squad``  | 24-layer, 1024-hidden, 16-heads, 340M parameters                  |
|               |                                                            | The ``bert-large-uncased-whole-word-masking`` model fine-tuned on SQuAD                  |
|               |                                                            | (see details of fine-tuning in the `example section`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-large-cased-whole-word-masking-finetuned-squad``    | 24-layer, 1024-hidden, 16-heads, 340M parameters                  |
|               |                                                            | The ``bert-large-cased-whole-word-masking`` model fine-tuned on SQuAD                  |
|               |                                                            | (see `details of fine-tuning in the example section <https://huggingface.co/pytorch-transformers/examples.html>`_)                 |
|               +------------------------------------------------------------+---------------------------+ 
|               | ``bert-base-cased-finetuned-mrpc``                         | 12-layer, 768-hidden, 12-heads, 110M parameters                  |
|               |                                                            | The ``bert-base-cased`` model fine-tuned on MRPC                  |
|               |                                                            | (see `details of fine-tuning in the example section <https://huggingface.co/pytorch-transformers/examples.html>`_)                 |
+---------------+------------------------------------------------------------+---------------------------+ 
|    GPT        | Cells may span columns.                                                                |
+---------------+----------------------------------------------------------------------------------------+ 

.. <https://huggingface.co/pytorch-transformers/examples.html>`_