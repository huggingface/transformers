.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Data Collator
-----------------------------------------------------------------------------------------------------------------------

Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of
the same type as the elements of :obj:`train_dataset` or :obj:`eval_dataset`.

To be able to build batches, data collators may apply some processing (like padding). Some of them (like
:class:`~transformers.DataCollatorForLanguageModeling`) also apply some random data augmentation (like random masking)
on the formed batch.

Examples of use can be found in the :doc:`example scripts <../examples>` or :doc:`example notebooks <../notebooks>`.


Default data collator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.data.data_collator.default_data_collator


DefaultDataCollator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DefaultDataCollator
    :members:


DataCollatorWithPadding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorWithPadding
    :members:


DataCollatorForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForTokenClassification
    :members:


DataCollatorForSeq2Seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForSeq2Seq
    :members:


DataCollatorForLanguageModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForLanguageModeling
    :members: numpy_mask_tokens, tf_mask_tokens, torch_mask_tokens


DataCollatorForWholeWordMask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForWholeWordMask
    :members: numpy_mask_tokens, tf_mask_tokens, torch_mask_tokens


DataCollatorForPermutationLanguageModeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.data.data_collator.DataCollatorForPermutationLanguageModeling
    :members: numpy_mask_tokens, tf_mask_tokens, torch_mask_tokens
