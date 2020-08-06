# Model name

## Model description

This model is a sequence-to-sequence question generator which takes an answer and context as an input, and generates a question as an output. It is based on a pretrained `t5-base` model.

## Intended uses & limitations

The model is trained to generate reading comprehension-style questions with answers extracted from a text. The model performs best with full sentence answers, but can also be used with single word or short phrase answers.

#### How to use

The model takes concatenated answers and context as an input sequence, and will generate a full question sentence as an output sequence. The max sequence length is 512 tokens. Inputs should be organised into the following format:
```
answer_token <answer-phrase> context_token <context-from-text>
```
The input sequence can then be encoded and passed as the `input_ids` argument in the model's `generate()` method.

For best results, a large number of questions can be generated, and then filtered using [iarfmoose/bert-base-cased-qa-evaluator](https://huggingface.co/iarfmoose/bert-base-cased-qa-evaluator).

For examples, please see https://github.com/iarfmoose/question_generator.

#### Limitations and bias

The model is limited to generating questions in the same style as those found in [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [CoQA](https://stanfordnlp.github.io/coqa/), and [MSMARCO](https://microsoft.github.io/msmarco/). The generated questions can potentially be leading or reflect biases that are present in the context. If the context is too short or completely absent, or if the context and answer do not match, the generated question is likely to be incoherent.

## Training data

The model was fine-tuned on a dataset made up of several well-known QA datasets ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [CoQA](https://stanfordnlp.github.io/coqa/), and [MSMARCO](https://microsoft.github.io/msmarco/)). The datasets were restructured by concatenating the answer and context fields into the previously-mentioned format. The question field from the datasets was used as the target during training. The full training set was roughly 200,000 examples.

## Training procedure

The model was trained for 20 epochs over the training set with a learning rate of 1e-3. The batch size was only 4 due to GPU memory limitations when training on Google Colab.
