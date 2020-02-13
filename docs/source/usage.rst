Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This page shows the most frequent use-cases when using the library. The models available allow for many different
configurations and a great versatility in use-cases. The most simple ones are presented here, showcasing usage
for tasks such as question answering, sequence classification, named entity recognition and others.

These examples leverage auto-models, but feel free to modify the code to be more specific and adapt it to your specific
use-case.

In order for a model to perform well on a task, it must be loaded from a checkpoint corresponding to that task. These
checkpoints are usually pre-trained on a large corpus of data and fine-tuned on a specific task. This means the
following:

- Not all models were fine-tuned on all tasks. If you want to fine-tune a model on a specific task, you can leverage
  one of the `run_$TASK.py` script in the `examples` directory.
- Fine-tuned models were fine-tuned on a specific dataset. This dataset may or may not overlap with your use-case
  and domain. As mentioned previously, you may leverage the `examples` scripts to fine-tune your model, or you
  may create your own training script.

In order to do an inference on a task, several mechanisms are made available by the library:

- Pipelines
- Using a model directly with a tokenizer

Both approaches are showcased here.

.. note::

    All tasks presented here leverage pre-trained checkpoints that were fine-tuned on specific tasks. Loading a
    checkpoint that was not fine-tuned on a specific task would load only the base transformer layers and not the
    additional head that is used for the task, initializing the weights of that head randomly.

    This produces random output.

Sequence Classification
--------------------------

Sequence classification is the task of classifying sequences according to a given number of classes. An example
of sequence classification is the GLUE dataset, which is entirely based on that task. If you would like to fine-tune
a model on a GLUE sequence classification task, you may leverage the `run_glue.py` or `run_tf_glue.py` scripts.

Here is an example using the pipelines do to sentiment analysis: identifying if a sequence is positive or negative.
It leverages a fine-tuned model on sst2, which is a GLUE task.

::

    from transformers import pipeline

    nlp = pipeline("sentiment-analysis")

    print(nlp("I hate you"))
    print(nlp("I love you"))

This should return a label ("POSITIVE" or "NEGATIVE") alongside a score, as follows:

::

    [{'label': 'NEGATIVE', 'score': 0.9991129}]
    [{'label': 'POSITIVE', 'score': 0.99986565}]


Here is an example of doing a sequence classification using a model to determine if two sequences are paraphrases
of each other. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it
  with the weights stored in the checkpoint.
- Build a sequence from the two sentences, with the correct model-specific separators token type ids
  and attention masks
- Pass this sequence through the model so that it is classified in one of the two available classes: 0
  (not a paraphrase) and 1 (is a paraphrase)
- Compute the softmax of the result to get probabilities over the classes
- Print the results

::

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

    classes = ["not paraphrase", "is paraphrase"]

    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
    not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")

    paraphrase_classification_logits = model(**paraphrase)[0]
    not_paraphrase_classification_logits = model(**not_paraphrase)[0]

    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

    print("Should be paraphrase")
    for i in range(len(classes)):
        print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")

    print("\nShould not be paraphrase")
    for i in range(len(classes)):
        print(f"{classes[i]}: {round(not_paraphrase_results[i] * 100)}%")

This should output the following results:

::

    Should be paraphrase
    not paraphrase: 10%
    is paraphrase: 90%

    Should not be paraphrase
    not paraphrase: 94%
    is paraphrase: 6%

Extractive Question Answering
----------------------------------------------------

Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the `run_squad.py`.

Here is an example using the pipelines do to sentiment analysis: identifying if a sequence is positive or negative.
It leverages a fine-tuned model on sst2, which is a GLUE task.

::

    from transformers import pipeline

    nlp = pipeline("question-answering")

    context = r"""
    Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
    question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
    a model on a SQuAD task, you may leverage the `run_squad.py`.
    """

    print(nlp(question="What is extractive question answering?", context=context))
    print(nlp(question="What is a good example of a question answering dataset?", context=context))

This should return an answer extracted from the text, a confidence score, alongside "start" and "end" values which
are the positions of the extracted answer in the text.

::

    {'score': 0.622232091629833, 'start': 34, 'end': 96, 'answer': 'the task of extracting an answer from a text given a question.'}
    {'score': 0.5115299158662765, 'start': 147, 'end': 161, 'answer': 'SQuAD dataset,'}


Here is an example of question answering using a model and a tokenizer. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it
  with the weights stored in the checkpoint.
- Define a text and a few questions.
- Iterate over the questions and build a sequence from the text and the current question, with the correct
  model-specific separators token type ids and attention masks
- Pass this sequence through the model. This outputs a range of scores across the entire sequence tokens (question and
  text), for both the start and end positions.
- Compute the softmax of the result to get probabilities over the tokens
- Fetch the tokens from the identified start and stop values, convert those tokens to a string.
- Print the results

::

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """

    questions = [
        "How many pretrained models are available in Transformers?",
        "What does Transformers provide?",
        "Transformers provides interoperability between which frameworks?",
    ]

    for i in range(len(questions)):
        question = questions[i]
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = model(**inputs)

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

This should output the questions followed by the answers:

::

    Question: How many pretrained models are available in Transformers?
    Answer: over 32 +

    Question: What does Transformers provide?
    Answer: general - purpose architectures

    Question: Transformers provides interoperability between which frameworks?
    Answer: tensorflow 2 . 0 and pytorch



Language Modeling
----------------------------------------------------

Language modeling is the task of fitting a model to a corpus, which can be domain specific. All popular transformer
based models are trained using a variant of language modeling, e.g. BERT with masked language modeling, GPT-2 with
causal language modeling.

Language modeling can be useful outside of pre-training as well, for example to shift the model distribution to be
domain-specific: using a language model trained over a very large corpus, and then fine-tuning it to a news dataset
or on scientific paper e.g. `LysandreJik/arxiv-nlp <https://huggingface.co/lysandre/arxiv-nlp>`__.

Masked Language Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Masked language modeling is the task of masking tokens in a sequence with a masking token, and prompting the model to
fill that mask with an appropriate token. This allows the model to attend to both the right context (tokens on the
right of the mask) and the left context (tokens on the left of the mask). Such a training creates a strong basis
for downstream tasks requiring bi-directional context such as SQuAD
(see `Lewis, Lui, Goyal et al. <https://arxiv.org/abs/1910.13461>`__, part 4.2).

Here is an example of using pipelines to replace a mask from a sequence:

::

    from transformers import pipeline

    nlp = pipeline("fill-mask")
    print(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

This should output the sequences with the mask filled, the confidence score as well as the token id in the tokenizer
vocabulary:

::

    [
        {'sequence': '<s> HuggingFace is creating a tool that the community uses to solve NLP tasks.</s>', 'score': 0.15627853572368622, 'token': 3944},
        {'sequence': '<s> HuggingFace is creating a framework that the community uses to solve NLP tasks.</s>', 'score': 0.11690319329500198, 'token': 7208},
        {'sequence': '<s> HuggingFace is creating a library that the community uses to solve NLP tasks.</s>', 'score': 0.058063216507434845, 'token': 5560},
        {'sequence': '<s> HuggingFace is creating a database that the community uses to solve NLP tasks.</s>', 'score': 0.04211743175983429, 'token': 8503},
        {'sequence': '<s> HuggingFace is creating a prototype that the community uses to solve NLP tasks.</s>', 'score': 0.024718601256608963, 'token': 17715}
    ]

Here is an example doing masked language modeling using a model and a tokenizer. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a DistilBERT model and
  loads it with the weights stored in the checkpoint.
- Define a sequence with a masked token, placing the :obj:`tokenizer.mask_token` instead of a word.
- Encode that sequence into IDs and find the position of the masked token in that list of IDs.
- Retrieve the predictions at the index of the mask token: this tensor has the same size as the vocabulary, and the
  values are the scores attributed to each token. The model gives higher score to tokens he deems probable in that
  context.
- Retrieve the top 5 tokens using the PyTorch :obj:`topk` method.
- Replace the mask token by the tokens and print the results

::

    from transformers import AutoModelWithLMHead, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

    sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

    input = tokenizer.encode(sequence, return_tensors="pt")
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]

    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))


This should print five sequences, with the top 5 tokens predicted by the model:

::

    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.

It is totally possible to put more than one mask token in a single sequence, but doing so would reduce the available
context (one less token in the context) and therefore decrease the prediction accuracy.


Causal Language Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Causal language modeling is the task of predicting the token following a sequence of tokens. In this situation, the
model only attends to the left context (tokens on the left of the mask). Such a training is particularly interesting
for generation tasks.

There is currently no pipeline to do causal language modeling/generation. This page will be completed once there is one.

Here is an example using the tokenizer and model. leveraging the :func:`generate` method to generate the tokens
following the initial sequence.

::

    from transformers import AutoModelWithLMHead, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")

    sequence = f"Hugging Face is based in DUMBO, New York City, and is"

    input = tokenizer.encode(sequence, return_tensors="pt")
    generated = model.generate(input, max_length=50)

    resulting_string = tokenizer.decode(generated.tolist()[0])
    print(resulting_string)

This outputs a (hopefully) coherent string from the original sequence:

::

    Hugging Face is based in DUMBO, New York City, and is a live-action TV series based on the novel by John
    Carpenter, and its producers, David Kustlin and Steve Pichar. The film is directed by!


Named Entity Recognition
----------------------------------------------------

Multiple Choice
----------------------------------------------------

Summarization
----------------------------------------------------
