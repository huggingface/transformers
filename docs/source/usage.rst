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

Here is an example of doing a sequence classification using a model to determine if two sequences are paraphrases
of each other. The process is the following:

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
--------------------------

Extractive Question Answering is the task of extracting an answer from a text according to a given question classifying sequences according to a given number of classes. An example
of sequence classification is the GLUE dataset, which is entirely based on that task. If you would like to fine-tune
a model on a GLUE sequence classification task, you may leverage the `run_glue.py` or `run_tf_glue.py` scripts.

Here is an example of doing a sequence classification using a model lo
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