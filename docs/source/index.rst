Transformers
================================================================================================================================================

🤗 Transformers (formerly known as `pytorch-transformers` and `pytorch-pretrained-bert`) provides general-purpose architectures
(BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural Language Understanding (NLU) and Natural Language Generation
(NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch.

This is the documentation of our repository `transformers <https://github.com/huggingface/transformers>`__.

Features
---------------------------------------------------

- As easy to use as pytorch-transformers
- As powerful and concise as Keras
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

Contents
---------------------------------

The library currently contains PyTorch and Tensorflow implementations, pre-trained model weights, usage scripts and conversion utilities for the following models:

1. `BERT <https://github.com/google-research/bert>`_ (from Google) released with the paper `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. `GPT <https://github.com/openai/finetune-transformer-lm>`_ (from OpenAI) released with the paper `Improving Language Understanding by Generative Pre-Training <https://blog.openai.com/language-unsupervised>`_ by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. `GPT-2 <https://blog.openai.com/better-language-models>`_ (from OpenAI) released with the paper `Language Models are Unsupervised Multitask Learners <https://blog.openai.com/better-language-models>`_ by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. `Transformer-XL <https://github.com/kimiyoung/transformer-xl>`_ (from Google/CMU) released with the paper `Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context <https://arxiv.org/abs/1901.02860>`_ by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. `XLNet <https://github.com/zihangdai/xlnet>`_ (from Google/CMU) released with the paper `​XLNet: Generalized Autoregressive Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`_ by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. `XLM <https://github.com/facebookresearch/XLM>`_ (from Facebook) released together with the paper `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`_ by Guillaume Lample and Alexis Conneau.
7. `RoBERTa <https://github.com/pytorch/fairseq/tree/master/examples/roberta>`_ (from Facebook), released together with the paper a `Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_ by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. `DistilBERT <https://huggingface.co/transformers/model_doc/distilbert.html>`_ (from HuggingFace) released together with the paper `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/abs/1910.01108>`_ by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into `DistilGPT2 <https://github.com/huggingface/transformers/tree/master/examples/distillation>`_.
9. `CTRL <https://github.com/pytorch/fairseq/tree/master/examples/ctrl>`_ (from Salesforce), released together with the paper `CTRL: A Conditional Transformer Language Model for Controllable Generation <https://www.github.com/salesforce/ctrl>`_ by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
10. `CamemBERT <https://huggingface.co/transformers/model_doc/camembert.html>`_ (from FAIR, Inria, Sorbonne Université) released together with the paper `CamemBERT: a Tasty French Language Model <https://arxiv.org/abs/1911.03894>`_ by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suarez, Yoann Dupont, Laurent Romary, Eric Villemonte de la Clergerie, Djame Seddah, and Benoît Sagot.
11. `ALBERT <https://github.com/google-research/ALBERT>`_ (from Google Research), released together with the paper a `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations <https://arxiv.org/abs/1909.11942>`_ by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
12. `XLM-RoBERTa <https://github.com/pytorch/fairseq/tree/master/examples/xlmr>`_ (from Facebook AI), released together with the paper `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`_ by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
13. `FlauBERT <https://github.com/getalp/Flaubert>`_ (from CNRS) released with the paper `FlauBERT: Unsupervised Language Model Pre-training for French <https://arxiv.org/abs/1912.05372>`_ by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.

.. toctree::
    :maxdepth: 2
    :caption: Notes

    installation
    quickstart
    glossary
    pretrained_models
    usage
    model_sharing
    examples
    notebooks
    serialization
    converting_tensorflow_models
    migration
    bertology
    torchscript
    multilingual
    benchmarks

.. toctree::
    :maxdepth: 2
    :caption: Main classes

    main_classes/configuration
    main_classes/model
    main_classes/tokenizer
    main_classes/pipelines
    main_classes/optimizer_schedules
    main_classes/processors

.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    model_doc/auto
    model_doc/encoderdecoder
    model_doc/bert
    model_doc/gpt
    model_doc/transformerxl
    model_doc/gpt2
    model_doc/xlm
    model_doc/xlnet
    model_doc/roberta
    model_doc/distilbert
    model_doc/ctrl
    model_doc/camembert
    model_doc/albert
    model_doc/xlmroberta
    model_doc/flaubert
    model_doc/bart
    model_doc/t5
    model_doc/electra
    model_doc/dialogpt
    model_doc/reformer
