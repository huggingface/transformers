Installation
================================================

This repo was tested on Python 2.7 and 3.5+ (examples are tested only on python 3.5+) and PyTorch 0.4.1/1.0.0

With pip
^^^^^^^^

PyTorch pretrained bert can be installed by pip as follows:

.. code-block:: bash

   pip install pytorch-pretrained-bert

If you want to reproduce the original tokenization process of the ``OpenAI GPT`` paper, you will need to install ``ftfy`` (limit to version 4.4.3 if you are using Python 2) and ``SpaCy`` :

.. code-block:: bash

   pip install spacy ftfy==4.4.3
   python -m spacy download en

If you don't install ``ftfy`` and ``SpaCy``\ , the ``OpenAI GPT`` tokenizer will default to tokenize using BERT's ``BasicTokenizer`` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).

From source
^^^^^^^^^^^

Clone the repository and run:

.. code-block:: bash

   pip install [--editable] .

Here also, if you want to reproduce the original tokenization process of the ``OpenAI GPT`` model, you will need to install ``ftfy`` (limit to version 4.4.3 if you are using Python 2) and ``SpaCy`` :

.. code-block:: bash

   pip install spacy ftfy==4.4.3
   python -m spacy download en

Again, if you don't install ``ftfy`` and ``SpaCy``\ , the ``OpenAI GPT`` tokenizer will default to tokenize using BERT's ``BasicTokenizer`` followed by Byte-Pair Encoding (which should be fine for most usage).

A series of tests is included in the `tests folder <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/tests>`_ and can be run using ``pytest`` (install pytest if needed: ``pip install pytest``\ ).

You can run the tests with the command:

.. code-block:: bash

   python -m pytest -sv tests/
