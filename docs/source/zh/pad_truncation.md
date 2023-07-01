<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）进行许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按许可证分发的软件是按“按原样”分发的基础上，不提供任何明示或暗示的担保或条件。请参阅许可证特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们 doc-builder（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确渲染。
-->

# Padding and truncation 填充和截断

批量输入通常具有不同的长度，因此无法转换为固定大小的张量。填充和截断是处理此问题的策略，用于从长度不同的批次创建矩形张量。填充会添加一个特殊的 **填充标记**，以确保较短的序列与批次中的最长序列或模型接受的最大长度具有相同的长度。截断则相反，通过截断长序列来处理。

在大多数情况下，将批次填充到最长序列的长度并将其截断为模型可以接受的最大长度通常效果很好。但是，如果需要，API 还支持更多的策略。您需要的三个参数是：`padding`，`truncation` 和 `max_length`。

`padding` 参数控制填充。它可以是布尔值或字符串：
  - `True` 或 `'longest'`：填充到批次中最长的序列（如果只提供一个序列，则不进行填充）。  - `'max_length'`：填充到由 `max_length` 参数指定的长度或模型接受的最大长度（如果未提供 `max_length=None`）。如果只提供一个序列，则仍将应用填充。  - `False` 或 `'do_not_pad'`：不进行填充。这是默认行为。    by the model if no `max_length` is provided (`max_length=None`). Padding will still be applied if you only provide a single sequence.
`truncation` 参数控制截断。它可以是布尔值或字符串：
  - `True` 或 `'longest_first'`：截断到由 `max_length` 参数指定的最大长度或模型接受的最大长度（如果未提供 `max_length=None`）。这将逐个标记进行截断，从最长序列中删除一个标记，直到达到适当的长度。
  - `'only_second'`：截断到由 `max_length` 参数指定的最大长度或模型接受的最大长度（如果未提供 `max_length=None`）。如果提供了一对序列（或一批序列对），则仅截断第二个句子。    the maximum length accepted by the model if no `max_length` is provided (`max_length=None`). This will
  - `'only_first'`：截断到由 `max_length` 参数指定的最大长度或模型接受的最大长度（如果未提供 `max_length=None`）。如果提供了一对序列（或一批序列对），则仅截断第一个句子。    reached.
  - `False` 或 `'do_not_truncate'`：不进行截断。这是默认行为。   

`max_length` 参数控制填充和截断的长度。它可以是整数或 `None`，在这种情况下，它将默认为模型可以接受的最大长度。如果模型没有特定的最大输入长度，则截断或填充到 `max_length` 将被禁用。

下表总结了设置填充和截断的推荐方式。如果在以下示例中使用输入序列对，可以将 `truncation=True` 替换为在 `['only_first', 'only_second', 'longest_first']` 中选择的 `STRATEGY`，即 `truncation='only_second'` 或 `truncation='longest_first'`，以控制序列对中的两个序列如前所述的截断方式

| Truncation                           | Padding                           | Instruction                                                                                 |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------|
| no truncation                        | no padding                        | `tokenizer(batch_sentences)`                                                           |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True)` or                                          |
|                                      |                                   | `tokenizer(batch_sentences, padding='longest')`                                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length')`                                     |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
|                                      | padding to a multiple of a value  | `tokenizer(batch_sentences, padding = True, pad_to_multiple_of = 8)                        |
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
