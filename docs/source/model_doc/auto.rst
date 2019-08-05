AutoModels
-----------

In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you are supplying to the ``from_pretrained`` method.

AutoClasses are here to do this job for you so that you automatically retreive the relevant model given the name/path to the pretrained weights/config/vocabulary.

There are two types of AutoClasses:

- ``AutoModel``, ``AutoConfig`` and ``AutoTokenizer``: instantiating these ones will directly create a class of the relevant architecture (ex: ``model = AutoModel.from_pretrained('bert-base-cased')`` will create a instance of ``BertModel``)
- All the others (``AutoModelWithLMHead``, ``AutoModelForSequenceClassification``...)  are standardized Auto classes for finetuning. Instantiating these will create instance of the same class (``AutoModelWithLMHead``, ``AutoModelForSequenceClassification``...) comprising (i) the relevant base model class (as mentioned just above) and (ii) a standard fine-tuning head on top, convenient for the task.


``AutoConfig``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoConfig
    :members:


``AutoModel``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoModel
    :members:


``AutoModelWithLMHead``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoModelWithLMHead
    :members:


``AutoModelForSequenceClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoModelForSequenceClassification
    :members:


``AutoTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoTokenizer
    :members:
