MMBT
----------------------------------------------------

Overview
~~~~~
The MMBT model was presented in `Supervised Multimodal Bitransformers for Classifying Images and Text <https://arxiv.org/pdf/1909.02950.pdf>`_ Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine
Here the abstract: 

*Self-supervised bidirectional transformer models such as BERT have led to dramatic improvements in a wide variety of textual classification tasks. The modern digital world is increasingly multimodal, however, and textual information is often accompanied by other modalities such as images. We introduce a supervised multimodal bitransformer model that fuses information from text and image encoders, and obtain state-of-the-art performance on various multimodal classification benchmark tasks, outperforming strong baselines, including on hard test sets specifically designed to measure multimodal performance.*

The Authors' code can be found `here <https://github.com/facebookresearch/mmbt/>`_ .

Tips
~~~~~~~~~~~~~~~~~~~~
- TODO (PVP): polish docstring and add tips.


MMBTConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MMBTConfig
    :members:


MMBTModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MMBTModel
    :members:


MMBTForClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MMBTForClassification
    :members:
