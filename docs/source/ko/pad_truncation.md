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

# 패딩과 잘라내기[[padding-and-truncation]]

배치 입력은 길이가 다른 경우가 많아서 고정 크기 텐서로 변환할 수 없습니다. 패딩과 잘라내기는 다양한 길이의 배치에서 직사각형 텐서를 생성할 수 있도록 이 문제를 해결하는 전략입니다. 패딩은 특수한 **패딩 토큰**을 추가하여 짧은 시퀀스가 배치에서 가장 긴 시퀀스 또는 모델에서 허용하는 최대 길이와 동일한 길이를 갖도록 합니다. 잘라내기는 긴 시퀀스를 잘라내어 패딩과 다른 방식으로 시퀀스의 길이를 동일하게 합니다.

대부분의 경우 배치에 가장 긴 시퀀스의 길이로 패딩하고 모델이 허용할 수 있는 최대 길이로 잘라내는 것이 잘 작동합니다. 그러나 필요하다면 API가 지원하는 더 많은 전략을 사용할 수 있습니다. 필요한 인수는 `padding`, `truncation`, `max_length` 세 가지입니다.

`padding` 인수는 패딩을 제어합니다. 불리언 또는 문자열일 수 있습니다:

  - `True` 또는 `'longest'`: 배치에서 가장 긴 시퀀스로 패딩합니다(단일 시퀀스만 제공하는 경우 패딩이 적용되지 않습니다).
  - `'max_length'`: `max_length` 인수가 지정한 길이로 패딩하거나, `max_length`가 제공되지 않은 경우(`max_length=None`) 모델에서 허용되는 최대 길이로 패딩합니다. 단일 시퀀스만 제공하는 경우에도 패딩이 적용됩니다.
  - `False` 또는 `'do_not_pad'`: 패딩이 적용되지 않습니다. 이것이 기본 동작입니다.

`truncation` 인수는 잘라낼 방법을 정합니다. 불리언 또는 문자열일 수 있습니다:

  - `True` 또는 `longest_first`: `max_length` 인수가 지정한 최대 길이로 잘라내거나, 
    `max_length`가 제공되지 않은 경우(`max_length=None`) 모델에서 허용되는 최대 길이로 잘라냅니다. 
    시퀀스 쌍에서 가장 긴 시퀀스의 토큰을 적절한 길이에 도달할 때까지 하나씩 제거합니다.
  - `'only_second'`: `max_length` 인수가 지정한 최대 길이로 잘라내거나, 
    `max_length`가 제공되지 않은 경우(`max_length=None`) 모델에서 허용되는 최대 길이로 잘라냅니다.
    시퀀스 쌍(또는 시퀀스 쌍의 배치)가 제공된 경우 쌍의 두 번째 문장만 잘라냅니다.
  - `'only_first'`: `max_length` 인수가 지정한 최대 길이로 잘라내거나, 
    `max_length`가 제공되지 않은 경우(`max_length=None`) 모델에서 허용되는 최대 길이로 잘라냅니다. 
    시퀀스 쌍(또는 시퀀스 쌍의 배치)가 제공된 경우 쌍의 첫 번째 문장만 잘라냅니다.
  - `False` 또는 `'do_not_truncate'`: 잘라내기를 적용하지 않습니다. 이것이 기본 동작입니다.

`max_length` 인수는 패딩 및 잘라내기를 적용할 길이를 제어합니다. 이 인수는 정수 또는 `None`일 수 있으며, `None`일 경우 모델이 허용할 수 있는 최대 길이로 기본값이 설정됩니다. 모델에 특정한 최대 입력 길이가 없는 경우 `max_length`에 대한 잘라내기 또는 패딩이 비활성화됩니다.

다음 표에는 패딩 및 잘라내기를 설정하는 권장 방법이 요약되어 있습니다. 
입력으로 시퀀스 쌍을 사용하는 경우, 다음 예제에서 `truncation=True`를 `['only_first', 'only_second', 'longest_first']`에서 선택한 `STRATEGY`, 즉 `truncation='only_second'` 또는 `truncation='longest_first'`로 바꾸면 앞서 설명한 대로 쌍의 두 시퀀스가 잘리는 방식을 제어할 수 있습니다.

| 잘라내기                             | 패딩                              | 사용 방법                                                                                 |
|--------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------|
| 잘라내기 없음                        | 패딩 없음                          | `tokenizer(batch_sentences)`                                                             |
|                                      | 배치 내 최대 길이로 패딩           | `tokenizer(batch_sentences, padding=True)` 또는                                          |
|                                      |                                   | `tokenizer(batch_sentences, padding='longest')`                                          |
|                                      | 모델의 최대 입력 길이로 패딩      | `tokenizer(batch_sentences, padding='max_length')`                                        |
|                                      | 특정 길이로 패딩                  | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                         |
|                                      | 다양한 길이로 패딩                | `tokenizer(batch_sentences, padding=True, pad_to_multiple_of=8)`                          |
| 모델의 최대 입력 길이로 잘라내기      | 패딩 없음                         | `tokenizer(batch_sentences, truncation=True)` 또는                                        |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY)`                                         |
|                                      | 배치 내 최대 길이로 패딩          | `tokenizer(batch_sentences, padding=True, truncation=True)` 또는                          |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`                           |
|                                      | 모델의 최대 입력 길이로 패딩      | `tokenizer(batch_sentences, padding='max_length', truncation=True)` 또는                  |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`                   |
|                                      | 특정 길이로 패딩                  | 사용 불가                                                                              |
| 특정 길이로 잘라내기                 | 패딩 없음                         | `tokenizer(batch_sentences, truncation=True, max_length=42)` 또는                         |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`                         |
|                                      | 배치 내 최대 길이로 패딩          | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` 또는           |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`           |
|                                      | 모델의 최대 입력 길이로 패딩       | 사용 불가                                                                             |
|                                      | 특정 길이로 패딩                   | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` 또는  |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)`   |
