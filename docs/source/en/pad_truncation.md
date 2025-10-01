<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Padding and truncation

Batched inputs are often different lengths, so they can't be converted to fixed-size tensors. Padding and truncation are strategies for dealing with this problem, to create rectangular tensors from batches of varying lengths. Padding adds a special **padding token** to ensure shorter sequences will have the same length as either the longest sequence in a batch or the maximum length accepted by the model. Truncation works in the other direction by truncating long sequences.

In most cases, padding your batch to the length of the longest sequence and truncating to the maximum length a model can accept works pretty well. However, the API supports more strategies if you need them. The three arguments you need to know are: `padding`, `truncation` and `max_length`.

The `padding` argument controls padding. It can be a boolean or a string:

- `True` or `'longest'`: pad to the longest sequence in the batch (no padding is applied if you only provide
    a single sequence).
- `'max_length'`: pad to a length specified by the `max_length` argument or the maximum length accepted
    by the model if no `max_length` is provided (`max_length=None`). Padding will still be applied if you only provide a single sequence.
- `False` or `'do_not_pad'`: no padding is applied. This is the default behavior.

The `truncation` argument controls truncation. It can be a boolean or a string:

- `True` or `'longest_first'`: truncate to a maximum length specified by the `max_length` argument or
    the maximum length accepted by the model if no `max_length` is provided (`max_length=None`). This will
    truncate token by token, removing a token from the longest sequence in the pair until the proper length is
    reached.
- `'only_second'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`). This will only truncate
    the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `'only_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`). This will only truncate
    the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `False` or `'do_not_truncate'`: no truncation is applied. This is the default behavior.

The `max_length` argument controls the length of the padding and truncation. It can be an integer or `None`, in which case it will default to the maximum length the model can accept. If the model has no specific maximum input length, truncation or padding to `max_length` is deactivated.

The following table summarizes the recommended way to setup padding and truncation. If you use pairs of input sequences in any of the following examples, you can replace `truncation=True` by a `STRATEGY` selected in
`['only_first', 'only_second', 'longest_first']`, i.e. `truncation='only_second'` or `truncation='longest_first'` to control how both sequences in the pair are truncated as detailed before.

| Truncation                           | Padding                           | Instruction                                                                                 |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------|
| no truncation                        | no padding                        | `tokenizer(batch_sentences)`                                                           |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True)` or                                          |
|                                      |                                   | `tokenizer(batch_sentences, padding='longest')`                                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length')`                                     |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
|                                      | padding to a multiple of a value  | `tokenizer(batch_sentences, padding=True, pad_to_multiple_of=8)`                        |
| truncation to max model input length | no padding                        | `tokenizer(batch_sentences, truncation=True)` or                                       |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY)`                                      |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True, truncation=True)` or                         |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length', truncation=True)` or                 |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`                |
|                                      | padding to specific length        | Not possible                                                                                |
| truncation to specific length        | no padding                        | `tokenizer(batch_sentences, truncation=True, max_length=42)` or                        |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`                       |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` or          |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`         |
|                                      | padding to max model input length | Not possible                                                                                |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` or  |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)` |
