Using tokenizers from 🤗 Tokenizers
=======================================================================================================================

The :class:`~transformers.PreTrainedTokenizerFast` depends on the `tokenizers
<https://huggingface.co/docs/tokenizers>`__ library. The tokenizers obtained from the 🤗 Tokenizers library can be
loaded very simply into 🤗 Transformers.

Before getting in the specifics, let's first start by creating a dummy tokenizer in a few lines:

.. code-block::

    >>> from tokenizers import Tokenizer
    >>> from tokenizers.models import BPE
    >>> from tokenizers.trainers import BpeTrainer
    >>> from tokenizers.pre_tokenizers import Whitespace

    >>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    >>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    >>> tokenizer.pre_tokenizer = Whitespace()
    >>> files = [...]
    >>> tokenizer.train(files, trainer)

We now have a tokenizer trained on the files we defined. We can either continue using it in that runtime, or save it to
a JSON file for future re-use.

Loading directly from the tokenizer object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's see how to leverage this tokenizer object in the 🤗 Transformers library. The
:class:`~transformers.PreTrainedTokenizerFast` class allows for easy instantiation, by accepting the instantiated
`tokenizer` object as an argument:

.. code-block::

    >>> from transformers import PreTrainedTokenizerFast

    >>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

This object can now be used with all the methods shared by the 🤗 Transformers tokenizers! Head to :doc:`the tokenizer
page <main_classes/tokenizer>` for more information.

Loading from a JSON file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to load a tokenizer from a JSON file, let's first start by saving our tokenizer:

.. code-block::

    >>> tokenizer.save("tokenizer.json")

The path to which we saved this file can be passed to the :class:`~transformers.PreTrainedTokenizerFast` initialization
method using the :obj:`tokenizer_file` parameter:

.. code-block::

    >>> from transformers import PreTrainedTokenizerFast

    >>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

This object can now be used with all the methods shared by the 🤗 Transformers tokenizers! Head to :doc:`the tokenizer
page <main_classes/tokenizer>` for more information.
