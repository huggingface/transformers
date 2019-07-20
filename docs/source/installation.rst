Installation
================================================

This repo was tested on Python 2.7 and 3.5+ (examples are tested only on python 3.5+) and PyTorch 0.4.1/1.0.0

With pip
^^^^^^^^

PyTorch pretrained bert can be installed with pip as follows:

.. code-block:: bash

   pip install pytorch-transformers

From source
^^^^^^^^^^^

Clone the repository and instal locally:

.. code-block:: bash

    git clone https://github.com/huggingface/pytorch-transformers.git
    cd pytorch-transformers
    pip install [--editable] .


Tests
^^^^^

An extensive test suite is included for the library and the example scripts. Library tests can be found in the `tests folder <https://github.com/huggingface/pytorch-transformers/tree/master/pytorch_transformers/tests>`_ and examples tests in the `examples folder <https://github.com/huggingface/pytorch-transformers/tree/master/examples>`_.

These tests can be run using `pytest` (install pytest if needed with `pip install pytest`).

You can run the tests from the root of the cloned repository with the commands:

.. code-block:: bash

    python -m pytest -sv ./pytorch_transformers/tests/
    python -m pytest -sv ./examples/


OpenAI GPT original tokenization workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to reproduce the original tokenization process of the ``OpenAI GPT`` paper, you will need to install ``ftfy`` (limit to version 4.4.3 if you are using Python 2) and ``SpaCy`` :

.. code-block:: bash

   pip install spacy ftfy==4.4.3
   python -m spacy download en

If you don't install ``ftfy`` and ``SpaCy``\ , the ``OpenAI GPT`` tokenizer will default to tokenize using BERT's ``BasicTokenizer`` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).
