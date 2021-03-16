Transformers
=======================================================================================================================

State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.

🤗 Transformers (formerly known as `pytorch-transformers` and `pytorch-pretrained-bert`) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.

This is the documentation of our repository `transformers <https://github.com/huggingface/transformers>`_.

Features
-----------------------------------------------------------------------------------------------------------------------

- High performance on NLU and NLG tasks
- Low barrier to entry for educators and practitioners

State-of-the-art NLP for everyone:

- Deep learning researchers
- Hands-on practitioners
- AI/ML/NLP teachers and educators

.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Lower compute costs, smaller carbon footprint:

- Researchers can share trained models instead of always retraining
- Practitioners can reduce compute time and production costs
- 8 architectures with over 30 pretrained models, some in more than 100 languages

Choose the right framework for every part of a model's lifetime:

- Train state-of-the-art models in 3 lines of code
- Deep interoperability between TensorFlow 2.0 and PyTorch models
- Move a single model between TF2.0/PyTorch frameworks at will
- Seamlessly pick the right framework for training, evaluation, production

Experimental support for Flax with a few models right now, expected to grow in the coming months.

`All the model checkpoints <https://huggingface.co/models>`__ are seamlessly integrated from the huggingface.co `model
hub <https://huggingface.co>`__ where they are uploaded directly by `users <https://huggingface.co/users>`__ and
`organizations <https://huggingface.co/organizations>`__.

Current number of checkpoints: |checkpoints|

.. |checkpoints| image:: https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen

Contents
-----------------------------------------------------------------------------------------------------------------------

The documentation is organized in five parts:

- **GET STARTED** contains a quick tour, the installation instructions and some useful information about our philosophy
  and a glossary.
- **USING 🤗 TRANSFORMERS** contains general tutorials on how to use the library.
- **ADVANCED GUIDES** contains more advanced guides that are more specific to a given script or part of the library.
- **RESEARCH** focuses on tutorials that have less to do with how to use the library but more about general research in
  transformers model
- The three last section contain the documentation of each public class and function, grouped in:

    - **MAIN CLASSES** for the main classes exposing the important APIs of the library.
    - **MODELS** for the classes and functions related to each model implemented in the library.
    - **INTERNAL HELPERS** for the classes and functions we use internally.

The library currently contains PyTorch, Tensorflow and Flax implementations, pretrained model weights, usage scripts
and conversion utilities for the following models:

..
    This list is updated automatically from the README with `make fix-copies`. Do not update manually!

1. :doc:`ALBERT <model_doc/albert>` (from Google Research and the Toyota Technological Institute at Chicago) released
   with the paper `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
   <https://arxiv.org/abs/1909.11942>`__, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush
   Sharma, Radu Soricut.
2. :doc:`BART <model_doc/bart>` (from Facebook) released with the paper `BART: Denoising Sequence-to-Sequence
   Pre-training for Natural Language Generation, Translation, and Comprehension
   <https://arxiv.org/pdf/1910.13461.pdf>`__ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman
   Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
3. :doc:`BARThez <model_doc/barthez>` (from École polytechnique) released with the paper `BARThez: a Skilled Pretrained
   French Sequence-to-Sequence Model <https://arxiv.org/abs/2010.12321>`__ by Moussa Kamal Eddine, Antoine J.-P.
   Tixier, Michalis Vazirgiannis.
4. :doc:`BERT <model_doc/bert>` (from Google) released with the paper `BERT: Pre-training of Deep Bidirectional
   Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ by Jacob Devlin, Ming-Wei Chang,
   Kenton Lee and Kristina Toutanova.
5. :doc:`BERT For Sequence Generation <model_doc/bertgeneration>` (from Google) released with the paper `Leveraging
   Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi
   Narayan, Aliaksei Severyn.
6. :doc:`Blenderbot <model_doc/blenderbot>` (from Facebook) released with the paper `Recipes for building an
   open-domain chatbot <https://arxiv.org/abs/2004.13637>`__ by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary
   Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
7. :doc:`BlenderbotSmall <model_doc/blenderbot_small>` (from Facebook) released with the paper `Recipes for building an
   open-domain chatbot <https://arxiv.org/abs/2004.13637>`__ by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary
   Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
8. :doc:`BORT <model_doc/bort>` (from Alexa) released with the paper `Optimal Subarchitecture Extraction For BERT
   <https://arxiv.org/abs/2010.10499>`__ by Adrian de Wynter and Daniel J. Perry.
9. :doc:`CamemBERT <model_doc/camembert>` (from Inria/Facebook/Sorbonne) released with the paper `CamemBERT: a Tasty
   French Language Model <https://arxiv.org/abs/1911.03894>`__ by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz
   Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
10. :doc:`ConvBERT <model_doc/convbert>` (from YituTech) released with the paper `ConvBERT: Improving BERT with
    Span-based Dynamic Convolution <https://arxiv.org/abs/2008.02496>`__ by Zihang Jiang, Weihao Yu, Daquan Zhou,
    Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
11. :doc:`CTRL <model_doc/ctrl>` (from Salesforce) released with the paper `CTRL: A Conditional Transformer Language
    Model for Controllable Generation <https://arxiv.org/abs/1909.05858>`__ by Nitish Shirish Keskar*, Bryan McCann*,
    Lav R. Varshney, Caiming Xiong and Richard Socher.
12. :doc:`DeBERTa <model_doc/deberta>` (from Microsoft) released with the paper `DeBERTa: Decoding-enhanced BERT with
    Disentangled Attention <https://arxiv.org/abs/2006.03654>`__ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu
    Chen.
13. :doc:`DeBERTa-v2 <model_doc/deberta_v2>` (from Microsoft) released with the paper `DeBERTa: Decoding-enhanced BERT
    with Disentangled Attention <https://arxiv.org/abs/2006.03654>`__ by Pengcheng He, Xiaodong Liu, Jianfeng Gao,
    Weizhu Chen.
14. :doc:`DialoGPT <model_doc/dialogpt>` (from Microsoft Research) released with the paper `DialoGPT: Large-Scale
    Generative Pre-training for Conversational Response Generation <https://arxiv.org/abs/1911.00536>`__ by Yizhe
    Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
15. :doc:`DistilBERT <model_doc/distilbert>` (from HuggingFace), released together with the paper `DistilBERT, a
    distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/abs/1910.01108>`__ by Victor
    Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into `DistilGPT2
    <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__, RoBERTa into `DistilRoBERTa
    <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__, Multilingual BERT into
    `DistilmBERT <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__ and a German
    version of DistilBERT.
16. :doc:`DPR <model_doc/dpr>` (from Facebook) released with the paper `Dense Passage Retrieval for Open-Domain
    Question Answering <https://arxiv.org/abs/2004.04906>`__ by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick
    Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
17. :doc:`ELECTRA <model_doc/electra>` (from Google Research/Stanford University) released with the paper `ELECTRA:
    Pre-training text encoders as discriminators rather than generators <https://arxiv.org/abs/2003.10555>`__ by Kevin
    Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
18. :doc:`FlauBERT <model_doc/flaubert>` (from CNRS) released with the paper `FlauBERT: Unsupervised Language Model
    Pre-training for French <https://arxiv.org/abs/1912.05372>`__ by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne,
    Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
19. :doc:`Funnel Transformer <model_doc/funnel>` (from CMU/Google Brain) released with the paper `Funnel-Transformer:
    Filtering out Sequential Redundancy for Efficient Language Processing <https://arxiv.org/abs/2006.03236>`__ by
    Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
20. :doc:`GPT <model_doc/gpt>` (from OpenAI) released with the paper `Improving Language Understanding by Generative
    Pre-Training <https://blog.openai.com/language-unsupervised/>`__ by Alec Radford, Karthik Narasimhan, Tim Salimans
    and Ilya Sutskever.
21. :doc:`GPT-2 <model_doc/gpt2>` (from OpenAI) released with the paper `Language Models are Unsupervised Multitask
    Learners <https://blog.openai.com/better-language-models/>`__ by Alec Radford*, Jeffrey Wu*, Rewon Child, David
    Luan, Dario Amodei** and Ilya Sutskever**.
22. :doc:`I-BERT <model_doc/ibert>` (from Berkeley) released with the paper `I-BERT: Integer-only BERT Quantization
    <https://arxiv.org/abs/2101.01321>`__ by Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer
23. :doc:`LayoutLM <model_doc/layoutlm>` (from Microsoft Research Asia) released with the paper `LayoutLM: Pre-training
    of Text and Layout for Document Image Understanding <https://arxiv.org/abs/1912.13318>`__ by Yiheng Xu, Minghao Li,
    Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
24. :doc:`LED <model_doc/led>` (from AllenAI) released with the paper `Longformer: The Long-Document Transformer
    <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy, Matthew E. Peters, Arman Cohan.
25. :doc:`Longformer <model_doc/longformer>` (from AllenAI) released with the paper `Longformer: The Long-Document
    Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy, Matthew E. Peters, Arman Cohan.
26. :doc:`LXMERT <model_doc/lxmert>` (from UNC Chapel Hill) released with the paper `LXMERT: Learning Cross-Modality
    Encoder Representations from Transformers for Open-Domain Question Answering <https://arxiv.org/abs/1908.07490>`__
    by Hao Tan and Mohit Bansal.
27. :doc:`M2M100 <model_doc/m2m_100>` (from Facebook) released with the paper `Beyond English-Centric Multilingual
    Machine Translation <https://arxiv.org/abs/2010.11125>`__ by by Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi
    Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman
    Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin.
28. :doc:`MarianMT <model_doc/marian>` Machine translation models trained using `OPUS <http://opus.nlpl.eu/>`__ data by
    Jörg Tiedemann. The `Marian Framework <https://marian-nmt.github.io/>`__ is being developed by the Microsoft
    Translator Team.
29. :doc:`MBart <model_doc/mbart>` (from Facebook) released with the paper `Multilingual Denoising Pre-training for
    Neural Machine Translation <https://arxiv.org/abs/2001.08210>`__ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li,
    Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
30. :doc:`MBart-50 <model_doc/mbart>` (from Facebook) released with the paper `Multilingual Translation with Extensible
    Multilingual Pretraining and Finetuning <https://arxiv.org/abs/2008.00401>`__ by Yuqing Tang, Chau Tran, Xian Li,
    Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan.
31. :doc:`MPNet <model_doc/mpnet>` (from Microsoft Research) released with the paper `MPNet: Masked and Permuted
    Pre-training for Language Understanding <https://arxiv.org/abs/2004.09297>`__ by Kaitao Song, Xu Tan, Tao Qin,
    Jianfeng Lu, Tie-Yan Liu.
32. :doc:`MT5 <model_doc/mt5>` (from Google AI) released with the paper `mT5: A massively multilingual pre-trained
    text-to-text transformer <https://arxiv.org/abs/2010.11934>`__ by Linting Xue, Noah Constant, Adam Roberts, Mihir
    Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
33. :doc:`Pegasus <model_doc/pegasus>` (from Google) released with the paper `PEGASUS: Pre-training with Extracted
    Gap-sentences for Abstractive Summarization <https://arxiv.org/abs/1912.08777>`__> by Jingqing Zhang, Yao Zhao,
    Mohammad Saleh and Peter J. Liu.
34. :doc:`ProphetNet <model_doc/prophetnet>` (from Microsoft Research) released with the paper `ProphetNet: Predicting
    Future N-gram for Sequence-to-Sequence Pre-training <https://arxiv.org/abs/2001.04063>`__ by Yu Yan, Weizhen Qi,
    Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
35. :doc:`Reformer <model_doc/reformer>` (from Google Research) released with the paper `Reformer: The Efficient
    Transformer <https://arxiv.org/abs/2001.04451>`__ by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
36. :doc:`RoBERTa <model_doc/roberta>` (from Facebook), released together with the paper a `Robustly Optimized BERT
    Pretraining Approach <https://arxiv.org/abs/1907.11692>`__ by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
    Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
37. :doc:`SpeechToTextTransformer <model_doc/speech_to_text>` (from Facebook), released together with the paper
    `fairseq S2T: Fast Speech-to-Text Modeling with fairseq <https://arxiv.org/abs/2010.05171>`__ by Changhan Wang, Yun
    Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino.
38. :doc:`SqueezeBert <model_doc/squeezebert>` released with the paper `SqueezeBERT: What can computer vision teach NLP
    about efficient neural networks? <https://arxiv.org/abs/2006.11316>`__ by Forrest N. Iandola, Albert E. Shaw, Ravi
    Krishna, and Kurt W. Keutzer.
39. :doc:`T5 <model_doc/t5>` (from Google AI) released with the paper `Exploring the Limits of Transfer Learning with a
    Unified Text-to-Text Transformer <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel and Noam Shazeer and Adam
    Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
40. :doc:`TAPAS <model_doc/tapas>` (from Google AI) released with the paper `TAPAS: Weakly Supervised Table Parsing via
    Pre-training <https://arxiv.org/abs/2004.02349>`__ by Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller,
    Francesco Piccinno and Julian Martin Eisenschlos.
41. :doc:`Transformer-XL <model_doc/transformerxl>` (from Google/CMU) released with the paper `Transformer-XL:
    Attentive Language Models Beyond a Fixed-Length Context <https://arxiv.org/abs/1901.02860>`__ by Zihang Dai*,
    Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
42. :doc:`Wav2Vec2 <model_doc/wav2vec2>` (from Facebook AI) released with the paper `wav2vec 2.0: A Framework for
    Self-Supervised Learning of Speech Representations <https://arxiv.org/abs/2006.11477>`__ by Alexei Baevski, Henry
    Zhou, Abdelrahman Mohamed, Michael Auli.
43. :doc:`XLM <model_doc/xlm>` (from Facebook) released together with the paper `Cross-lingual Language Model
    Pretraining <https://arxiv.org/abs/1901.07291>`__ by Guillaume Lample and Alexis Conneau.
44. :doc:`XLM-ProphetNet <model_doc/xlmprophetnet>` (from Microsoft Research) released with the paper `ProphetNet:
    Predicting Future N-gram for Sequence-to-Sequence Pre-training <https://arxiv.org/abs/2001.04063>`__ by Yu Yan,
    Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
45. :doc:`XLM-RoBERTa <model_doc/xlmroberta>` (from Facebook AI), released together with the paper `Unsupervised
    Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__ by Alexis Conneau*, Kartikay
    Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke
    Zettlemoyer and Veselin Stoyanov.
46. :doc:`XLNet <model_doc/xlnet>` (from Google/CMU) released with the paper `​XLNet: Generalized Autoregressive
    Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`__ by Zhilin Yang*, Zihang Dai*, Yiming
    Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
47. :doc:`XLSR-Wav2Vec2 <model_doc/xlsr_wav2vec2>` (from Facebook AI) released with the paper `Unsupervised
    Cross-Lingual Representation Learning For Speech Recognition <https://arxiv.org/abs/2006.13979>`__ by Alexis
    Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli.


.. _bigtable:

The table below represents the current support in the library for each of those models, whether they have a Python
tokenizer (called "slow"). A "fast" tokenizer backed by the 🤗 Tokenizers library, whether they have support in PyTorch,
TensorFlow and/or Flax.

..
    This table is updated automatically from the auto modules with `make fix-copies`. Do not update manually!

.. rst-class:: center-aligned-table

+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            Model            | Tokenizer slow | Tokenizer fast | PyTorch support | TensorFlow support | Flax Support |
+=============================+================+================+=================+====================+==============+
|           ALBERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            BART             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            BERT             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|       Bert Generation       |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         Blenderbot          |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|       BlenderbotSmall       |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            CTRL             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          CamemBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          ConvBERT           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             DPR             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           DeBERTa           |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         DeBERTa-v2          |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         DistilBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           ELECTRA           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|       Encoder decoder       |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
| FairSeq Machine-Translation |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          FlauBERT           |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|     Funnel Transformer      |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           I-BERT            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             LED             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           LXMERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          LayoutLM           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         Longformer          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           M2M100            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            MPNet            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           Marian            |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         MobileBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         OpenAI GPT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|        OpenAI GPT-2         |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           Pegasus           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         ProphetNet          |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             RAG             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          Reformer           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          RetriBERT          |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           RoBERTa           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         Speech2Text         |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         SqueezeBERT         |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             T5              |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            TAPAS            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|       Transformer-XL        |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          Wav2Vec2           |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             XLM             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         XLM-RoBERTa         |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|        XLMProphetNet        |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            XLNet            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|            mBART            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             mT5             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+

.. toctree::
    :maxdepth: 2
    :caption: Get started

    quicktour
    installation
    philosophy
    glossary

.. toctree::
    :maxdepth: 2
    :caption: Using 🤗 Transformers

    task_summary
    model_summary
    preprocessing
    training
    model_sharing
    tokenizer_summary
    multilingual

.. toctree::
    :maxdepth: 2
    :caption: Advanced guides

    pretrained_models
    examples
    custom_datasets
    notebooks
    community
    converting_tensorflow_models
    migration
    contributing
    add_new_model
    testing
    serialization

.. toctree::
    :maxdepth: 2
    :caption: Research

    bertology
    perplexity
    benchmarks

.. toctree::
    :maxdepth: 2
    :caption: Main Classes

    main_classes/callback
    main_classes/configuration
    main_classes/logging
    main_classes/model
    main_classes/optimizer_schedules
    main_classes/output
    main_classes/pipelines
    main_classes/processors
    main_classes/tokenizer
    main_classes/trainer
    main_classes/feature_extractor

.. toctree::
    :maxdepth: 2
    :caption: Models

    model_doc/albert
    model_doc/auto
    model_doc/bart
    model_doc/barthez
    model_doc/bert
    model_doc/bertweet
    model_doc/bertgeneration
    model_doc/blenderbot
    model_doc/blenderbot_small
    model_doc/bort
    model_doc/camembert
    model_doc/convbert
    model_doc/ctrl
    model_doc/deberta
    model_doc/deberta_v2
    model_doc/dialogpt
    model_doc/distilbert
    model_doc/dpr
    model_doc/electra
    model_doc/encoderdecoder
    model_doc/flaubert
    model_doc/fsmt
    model_doc/funnel
    model_doc/herbert
    model_doc/ibert
    model_doc/layoutlm
    model_doc/led
    model_doc/longformer
    model_doc/lxmert
    model_doc/marian
    model_doc/m2m_100
    model_doc/mbart
    model_doc/mobilebert
    model_doc/mpnet
    model_doc/mt5
    model_doc/gpt
    model_doc/gpt2
    model_doc/pegasus
    model_doc/phobert
    model_doc/prophetnet
    model_doc/rag
    model_doc/reformer
    model_doc/retribert
    model_doc/roberta
    model_doc/speech_to_text
    model_doc/squeezebert
    model_doc/t5
    model_doc/tapas
    model_doc/transformerxl
    model_doc/wav2vec2
    model_doc/xlm
    model_doc/xlmprophetnet
    model_doc/xlmroberta
    model_doc/xlnet
    model_doc/xlsr_wav2vec2

.. toctree::
    :maxdepth: 2
    :caption: Internal Helpers

    internal/modeling_utils
    internal/pipelines_utils
    internal/tokenization_utils
    internal/trainer_utils
    internal/generation_utils
    internal/file_utils
