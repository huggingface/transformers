Bart
----------------------------------------------------

FIXME

Tips:

- xxx
- Need leading spaces for fairseq equality
- Bart doesn't have `token_type_ids`, for sequence classification just use BartTokenizer.encode to get the proper splitting.


BartConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartConfig
    :members:

Inputs
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.modeling_bart.prepare_bart_inputs_dict


BartTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartTokenizer
    :members: forward


BartModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartModel
    :members: forward


BartForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForMaskedLM
    :members: forward


BartForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForSequenceClassification
    :members: forward
