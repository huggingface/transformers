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
3. :doc:`BERT <model_doc/bert>` (from Google) released with the paper `BERT: Pre-training of Deep Bidirectional
   Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ by Jacob Devlin, Ming-Wei Chang,
   Kenton Lee and Kristina Toutanova.
4. :doc:`BERT For Sequence Generation <model_doc/bertgeneration>` (from Google) released with the paper `Leveraging
   Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi
   Narayan, Aliaksei Severyn.
5. :doc:`Blenderbot <model_doc/blenderbot>` (from Facebook) released with the paper `Recipes for building an
   open-domain chatbot <https://arxiv.org/abs/2004.13637>`__ by Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary
   Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
6. :doc:`CamemBERT <model_doc/camembert>` (from Inria/Facebook/Sorbonne) released with the paper `CamemBERT: a Tasty
   French Language Model <https://arxiv.org/abs/1911.03894>`__ by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz
   Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
7. :doc:`CTRL <model_doc/ctrl>` (from Salesforce) released with the paper `CTRL: A Conditional Transformer Language
   Model for Controllable Generation <https://arxiv.org/abs/1909.05858>`__ by Nitish Shirish Keskar*, Bryan McCann*,
   Lav R. Varshney, Caiming Xiong and Richard Socher.
8. :doc:`DeBERTa <model_doc/deberta>` (from Microsoft Research) released with the paper `DeBERTa: Decoding-enhanced
   BERT with Disentangled Attention <https://arxiv.org/abs/2006.03654>`__ by Pengcheng He, Xiaodong Liu, Jianfeng Gao,
   Weizhu Chen.
9. :doc:`DialoGPT <model_doc/dialogpt>` (from Microsoft Research) released with the paper `DialoGPT: Large-Scale
   Generative Pre-training for Conversational Response Generation <https://arxiv.org/abs/1911.00536>`__ by Yizhe Zhang,
   Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
10. :doc:`DistilBERT <model_doc/distilbert>` (from HuggingFace), released together with the paper `DistilBERT, a
    distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/abs/1910.01108>`__ by Victor
    Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into `DistilGPT2
    <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__, RoBERTa into `DistilRoBERTa
    <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__, Multilingual BERT into
    `DistilmBERT <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__ and a German
    version of DistilBERT.
11. :doc:`DPR <model_doc/dpr>` (from Facebook) released with the paper `Dense Passage Retrieval for Open-Domain
    Question Answering <https://arxiv.org/abs/2004.04906>`__ by Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick
    Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
12. :doc:`ELECTRA <model_doc/electra>` (from Google Research/Stanford University) released with the paper `ELECTRA:
    Pre-training text encoders as discriminators rather than generators <https://arxiv.org/abs/2003.10555>`__ by Kevin
    Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
13. :doc:`FlauBERT <model_doc/flaubert>` (from CNRS) released with the paper `FlauBERT: Unsupervised Language Model
    Pre-training for French <https://arxiv.org/abs/1912.05372>`__ by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne,
    Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
14. :doc:`Funnel Transformer <model_doc/funnel>` (from CMU/Google Brain) released with the paper `Funnel-Transformer:
    Filtering out Sequential Redundancy for Efficient Language Processing <https://arxiv.org/abs/2006.03236>`__ by
    Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
15. :doc:`GPT <model_doc/gpt>` (from OpenAI) released with the paper `Improving Language Understanding by Generative
    Pre-Training <https://blog.openai.com/language-unsupervised/>`__ by Alec Radford, Karthik Narasimhan, Tim Salimans
    and Ilya Sutskever.
16. :doc:`GPT-2 <model_doc/gpt2>` (from OpenAI) released with the paper `Language Models are Unsupervised Multitask
    Learners <https://blog.openai.com/better-language-models/>`__ by Alec Radford*, Jeffrey Wu*, Rewon Child, David
    Luan, Dario Amodei** and Ilya Sutskever**.
17. :doc:`LayoutLM <model_doc/layoutlm>` (from Microsoft Research Asia) released with the paper `LayoutLM: Pre-training
    of Text and Layout for Document Image Understanding <https://arxiv.org/abs/1912.13318>`__ by Yiheng Xu, Minghao Li,
    Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
18. :doc:`Longformer <model_doc/longformer>` (from AllenAI) released with the paper `Longformer: The Long-Document
    Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy, Matthew E. Peters, Arman Cohan.
19. :doc:`LXMERT <model_doc/lxmert>` (from UNC Chapel Hill) released with the paper `LXMERT: Learning Cross-Modality
    Encoder Representations from Transformers for Open-Domain Question Answering <https://arxiv.org/abs/1908.07490>`__
    by Hao Tan and Mohit Bansal.
20. :doc:`MarianMT <model_doc/marian>` Machine translation models trained using `OPUS <http://opus.nlpl.eu/>`__ data by
    Jörg Tiedemann. The `Marian Framework <https://marian-nmt.github.io/>`__ is being developed by the Microsoft
    Translator Team.
21. :doc:`MBart <model_doc/mbart>` (from Facebook) released with the paper `Multilingual Denoising Pre-training for
    Neural Machine Translation <https://arxiv.org/abs/2001.08210>`__ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li,
    Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
22. :doc:`MT5 <model_doc/mt5>` (from Google AI) released with the paper `mT5: A massively multilingual pre-trained
    text-to-text transformer <https://arxiv.org/abs/2010.11934>`__ by Linting Xue, Noah Constant, Adam Roberts, Mihir
    Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
23. :doc:`Pegasus <model_doc/pegasus>` (from Google) released with the paper `PEGASUS: Pre-training with Extracted
    Gap-sentences for Abstractive Summarization <https://arxiv.org/abs/1912.08777>`__> by Jingqing Zhang, Yao Zhao,
    Mohammad Saleh and Peter J. Liu.
24. :doc:`ProphetNet <model_doc/prophetnet>` (from Microsoft Research) released with the paper `ProphetNet: Predicting
    Future N-gram for Sequence-to-Sequence Pre-training <https://arxiv.org/abs/2001.04063>`__ by Yu Yan, Weizhen Qi,
    Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
25. :doc:`Reformer <model_doc/reformer>` (from Google Research) released with the paper `Reformer: The Efficient
    Transformer <https://arxiv.org/abs/2001.04451>`__ by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
26. :doc:`RoBERTa <model_doc/roberta>` (from Facebook), released together with the paper a `Robustly Optimized BERT
    Pretraining Approach <https://arxiv.org/abs/1907.11692>`__ by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
    Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. ultilingual BERT into `DistilmBERT
    <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__ and a German version of
    DistilBERT.
27. :doc:`SqueezeBert <model_doc/squeezebert>` released with the paper `SqueezeBERT: What can computer vision teach NLP
    about efficient neural networks? <https://arxiv.org/abs/2006.11316>`__ by Forrest N. Iandola, Albert E. Shaw, Ravi
    Krishna, and Kurt W. Keutzer.
28. :doc:`T5 <model_doc/t5>` (from Google AI) released with the paper `Exploring the Limits of Transfer Learning with a
    Unified Text-to-Text Transformer <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel and Noam Shazeer and Adam
    Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
29. :doc:`Transformer-XL <model_doc/transformerxl>` (from Google/CMU) released with the paper `Transformer-XL:
    Attentive Language Models Beyond a Fixed-Length Context <https://arxiv.org/abs/1901.02860>`__ by Zihang Dai*,
    Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
30. :doc:`XLM <model_doc/xlm>` (from Facebook) released together with the paper `Cross-lingual Language Model
    Pretraining <https://arxiv.org/abs/1901.07291>`__ by Guillaume Lample and Alexis Conneau.
31. :doc:`XLM-ProphetNet <model_doc/xlmprophetnet>` (from Microsoft Research) released with the paper `ProphetNet:
    Predicting Future N-gram for Sequence-to-Sequence Pre-training <https://arxiv.org/abs/2001.04063>`__ by Yu Yan,
    Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou.
32. :doc:`XLM-RoBERTa <model_doc/xlmroberta>` (from Facebook AI), released together with the paper `Unsupervised
    Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__ by Alexis Conneau*, Kartikay
    Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke
    Zettlemoyer and Veselin Stoyanov.
33. :doc:`XLNet <model_doc/xlnet>` (from Google/CMU) released with the paper `​XLNet: Generalized Autoregressive
    Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`__ by Zhilin Yang*, Zihang Dai*, Yiming
    Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
34. `Other community models <https://huggingface.co/models>`__, contributed by the `community
    <https://huggingface.co/users>`__.


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
|            CTRL             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          CamemBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             DPR             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           DeBERTa           |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
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
|           LXMERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          LayoutLM           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         Longformer          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
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
|             RAG             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          Reformer           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|          RetriBERT          |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|           RoBERTa           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|         SqueezeBERT         |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|             T5              |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
+-----------------------------+----------------+----------------+-----------------+--------------------+--------------+
|       Transformer-XL        |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
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
    converting_tensorflow_models
    migration
    contributing
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

.. toctree::
    :maxdepth: 2
    :caption: Models

    model_doc/albert
    model_doc/auto
    model_doc/bart
    model_doc/bert
    model_doc/bertgeneration
    model_doc/blenderbot
    model_doc/camembert
    model_doc/ctrl
    model_doc/deberta
    model_doc/dialogpt
    model_doc/distilbert
    model_doc/dpr
    model_doc/electra
    model_doc/encoderdecoder
    model_doc/flaubert
    model_doc/fsmt
    model_doc/funnel
    model_doc/layoutlm
    model_doc/longformer
    model_doc/lxmert
    model_doc/marian
    model_doc/mbart
    model_doc/mobilebert
    model_doc/mt5
    model_doc/gpt
    model_doc/gpt2
    model_doc/pegasus
    model_doc/prophetnet
    model_doc/rag
    model_doc/reformer
    model_doc/retribert
    model_doc/roberta
    model_doc/squeezebert
    model_doc/t5
    model_doc/transformerxl
    model_doc/xlm
    model_doc/xlmprophetnet
    model_doc/xlmroberta
    model_doc/xlnet

.. toctree::
    :maxdepth: 2
    :caption: Internal Helpers

    internal/modeling_utils
    internal/pipelines_utils
    internal/tokenization_utils
    internal/trainer_utils
    internal/generation_utils
