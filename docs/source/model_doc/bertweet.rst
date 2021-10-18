.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

BERTweet
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BERTweet model was proposed in `BERTweet: A pre-trained language model for English Tweets
<https://www.aclweb.org/anthology/2020.emnlp-demos.2.pdf>`__ by Dat Quoc Nguyen, Thanh Vu, Anh Tuan Nguyen.

The abstract from the paper is the following:

*We present BERTweet, the first public large-scale pre-trained language model for English Tweets. Our BERTweet, having
the same architecture as BERT-base (Devlin et al., 2019), is trained using the RoBERTa pre-training procedure (Liu et
al., 2019). Experiments show that BERTweet outperforms strong baselines RoBERTa-base and XLM-R-base (Conneau et al.,
2020), producing better performance results than the previous state-of-the-art models on three Tweet NLP tasks:
Part-of-speech tagging, Named-entity recognition and text classification.*

Example of use:

.. code-block::

    >>> import torch
    >>> from transformers import AutoModel, AutoTokenizer 

    >>> bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

    >>> # For transformers v4.x+: 
    >>> tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

    >>> # For transformers v3.x: 
    >>> # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    >>> # INPUT TWEET IS ALREADY NORMALIZED!
    >>> line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

    >>> input_ids = torch.tensor([tokenizer.encode(line)])

    >>> with torch.no_grad():
    ...     features = bertweet(input_ids)  # Models outputs are now tuples

    >>> # With TensorFlow 2.0+:
    >>> # from transformers import TFAutoModel
    >>> # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")

This model was contributed by `dqnguyen <https://huggingface.co/dqnguyen>`__. The original code can be found `here
<https://github.com/VinAIResearch/BERTweet>`__.

BertweetTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetTokenizer
    :members: 
