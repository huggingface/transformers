Pytorch-Transformers
================================================================================================================================================

PyTorch-Transformers is a library of state-of-the-art pre-trained models for Natural Language Processing (NLP).

The library currently contains PyTorch implementations, pre-trained model weights, usage scripts and conversion utilities for the following models:

1. `BERT <https://github.com/google-research/bert>`_ (from Google) released with the paper `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. `GPT <https://github.com/openai/finetune-transformer-lm>`_ (from OpenAI) released with the paper `Improving Language Understanding by Generative Pre-Training <https://blog.openai.com/language-unsupervised>`_ by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. `GPT-2 <https://blog.openai.com/better-language-models>`_ (from OpenAI) released with the paper `Language Models are Unsupervised Multitask Learners <https://blog.openai.com/better-language-models>`_ by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. `Transformer-XL <https://github.com/kimiyoung/transformer-xl>`_ (from Google/CMU) released with the paper `Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context <https://arxiv.org/abs/1901.02860>`_ by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. `XLNet <https://github.com/zihangdai/xlnet>`_ (from Google/CMU) released with the paper `​XLNet: Generalized Autoregressive Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`_ by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. `XLM <https://github.com/facebookresearch/XLM>`_ (from Facebook) released together with the paper `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`_ by Guillaume Lample and Alexis Conneau.
7. `RoBERTa <https://github.com/pytorch/fairseq/tree/master/examples/roberta>`_ (from Facebook), released together with the paper a `Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_ by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. `DistilBERT <https://huggingface.co/pytorch-transformers/model_doc/distilbert.html>`_ (from HuggingFace) released together with the blog post `Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT <https://medium.com/huggingface/distilbert-8cf3380435b5>`_ by Victor Sanh, Lysandre Debut and Thomas Wolf.

.. toctree::
    :maxdepth: 2
    :caption: Notes

    installation
    quickstart
    pretrained_models
    examples
    notebooks
    serialization
    converting_tensorflow_models
    migration
    bertology
    torchscript

.. toctree::
    :maxdepth: 2
    :caption: Main classes

    main_classes/configuration
    main_classes/model
    main_classes/tokenizer
    main_classes/optimizer_schedules

.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    model_doc/auto
    model_doc/bert
    model_doc/gpt
    model_doc/transformerxl
    model_doc/gpt2
    model_doc/xlm
    model_doc/xlnet
    model_doc/roberta
    model_doc/distilbert
