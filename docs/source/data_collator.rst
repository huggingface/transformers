<!---
     Copyright 2020 The HuggingFace Team. All rights reserved.

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.

-->

DataCollator
-----------------------------------------------------------------------------------------------------------------------

DataCollators are objects that will form a batch by using a list of elements as input. These lists of elements are of
type :obj:`train_dataset` or :obj:`eval_dataset`.

The base class :class:`~transformers.data.data_collator` can use the default
:func:`transformers.data.data_collator.default_data_collator`, which is a function that takes a list of samples from a
Dataset as input and collates them into a batch of a dict-like object. The default collator performs special handling
of potential keys:

    - ``label``: handles a single value (int or float) per object
    - ``label_ids``: handles a list of values per object

This function does not perform any preprocessing. An example of use can be found in glue and ner.

A list of DataCollators can be found here:

    - :class:`~transformers.data.data_collator.DataCollatorWithPadding`
    - :class:`~transformers.data.data_collator.DataCollatorForTokenClassification`
    - :class:`~transformers.data.data_collator.DataCollatorForSeq2Seq`
    - :class:`~transformers.data.data_collator.DataCollatorForLanguageModeling`
    - :class:`~transformers.data.data_collator.DataCollatorForWholeWordMask`
    - :class:`~transformers.data.data_collator.DataCollatorForSOP`
    - :class:`~transformers.data.data_collator.DataCollatorForPermutationLanguageModeling`






