<!--- Copyright 2020 The HuggingFace Team. All rights reserved.

     Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
     with the License. You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
     the specific language governing permissions and limitations under the License.

-->

DataCollator
-----------------------------------------------------------------------------------------------------------------------

DataCollators are objects that will form a batch by using a list of elements as input. These lists of elements are of
the same type as the elements of :obj:`train_dataset` or :obj:`eval_dataset`.

A data collator will default to :func:`transformers.data.data_collator.default_data_collator` if no `tokenizer` has
been provided. This is a function that takes a list of samples from a Dataset as input and collates them into a batch
of a dict-like object. The default collator performs special handling of potential keys:

    - ``label``: handles a single value (int or float) per object
    - ``label_ids``: handles a list of values per object

This function does not perform any preprocessing. An example of use can be found in glue and ner.


Default data collator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.data.data_collator.default_data_collator


DataCollatorWithPadding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorWithPadding
    :special-members: __call__
    :members:

DataCollatorForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForTokenClassification
    :special-members: __call__
    :members:

DataCollatorForSeq2Seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForSeq2Seq
    :special-members: __call__
    :members:

DataCollatorForLanguageModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForLanguageModeling
    :special-members: __call__
    :members: mask_tokens

DataCollatorForWholeWordMask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForWholeWordMask
    :special-members: __call__
    :members: mask_tokens

DataCollatorForSOP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForSOP
    :special-members: __call__
    :members: mask_tokens

DataCollatorForPermutationLanguageModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForPermutationLanguageModeling
    :special-members: __call__
    :members: mask_tokens
