Summary of the tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This page shows the most frequent use-cases when using the library. The models available allow for many different
configurations and a great versatility in use-cases. The most simple ones are presented here, showcasing usage
for tasks such as question answering, sequence classification, named entity recognition and others.

These examples leverage auto-models, which are classes that will instantiate a model according to a given checkpoint,
automatically selecting the correct model architecture. Please check the :class:`~transformers.AutoModel` documentation
for more information.
Feel free to modify the code to be more specific and adapt it to your specific use-case.

In order for a model to perform well on a task, it must be loaded from a checkpoint corresponding to that task. These
checkpoints are usually pre-trained on a large corpus of data and fine-tuned on a specific task. This means the
following:

- Not all models were fine-tuned on all tasks. If you want to fine-tune a model on a specific task, you can leverage
  one of the `run_$TASK.py` script in the
  `examples <https://github.com/huggingface/transformers/tree/master/examples>`_ directory.
- Fine-tuned models were fine-tuned on a specific dataset. This dataset may or may not overlap with your use-case
  and domain. As mentioned previously, you may leverage the
  `examples <https://github.com/huggingface/transformers/tree/master/examples>`_ scripts to fine-tune your model, or you
  may create your own training script.

In order to do an inference on a task, several mechanisms are made available by the library:

- Pipelines: very easy-to-use abstractions, which require as little as two lines of code.
- Using a model directly with a tokenizer (PyTorch/TensorFlow): the full inference using the model. Less abstraction,
  but much more powerful.

Both approaches are showcased here.

.. note::

    All tasks presented here leverage pre-trained checkpoints that were fine-tuned on specific tasks. Loading a
    checkpoint that was not fine-tuned on a specific task would load only the base transformer layers and not the
    additional head that is used for the task, initializing the weights of that head randomly.

    This would produce random output.

Sequence Classification
--------------------------

Sequence classification is the task of classifying sequences according to a given number of classes. An example
of sequence classification is the GLUE dataset, which is entirely based on that task. If you would like to fine-tune
a model on a GLUE sequence classification task, you may leverage the
`run_glue.py <https://github.com/huggingface/transformers/tree/master/examples/text-classification/run_glue.py>`_ or
`run_tf_glue.py <https://github.com/huggingface/transformers/tree/master/examples/text-classification/run_tf_glue.py>`_ scripts.

Here is an example using the pipelines do to sentiment analysis: identifying if a sequence is positive or negative.
It leverages a fine-tuned model on sst2, which is a GLUE task.

This returns a label ("POSITIVE" or "NEGATIVE") alongside a score, as follows:

.. code-block::

    >>> from transformers import pipeline

    >>> nlp = pipeline("sentiment-analysis")

    >>> result = nlp("I hate you")[0]
    >>> print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    label: NEGATIVE, with score: 0.9991

    >>> result = nlp("I love you")[0]
    >>> print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    label: POSITIVE, with score: 0.9999


Here is an example of doing a sequence classification using a model to determine if two sequences are paraphrases
of each other. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it
  with the weights stored in the checkpoint.
- Build a sequence from the two sentences, with the correct model-specific separators token type ids
  and attention masks (:func:`~transformers.PreTrainedTokenizer.encode` and
  :func:`~transformers.PreTrainedTokenizer.__call__` take care of this)
- Pass this sequence through the model so that it is classified in one of the two available classes: 0
  (not a paraphrase) and 1 (is a paraphrase)
- Compute the softmax of the result to get probabilities over the classes
- Print the results

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

    >>> classes = ["not paraphrase", "is paraphrase"]

    >>> sequence_0 = "The company HuggingFace is based in New York City"
    >>> sequence_1 = "Apples are especially bad for your health"
    >>> sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    >>> paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
    >>> not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

    >>> paraphrase_classification_logits = model(**paraphrase)[0]
    >>> not_paraphrase_classification_logits = model(**not_paraphrase)[0]

    >>> paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    >>> not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

    >>> # Should be paraphrase
    >>> for i in range(len(classes)):
    ...     print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
    not paraphrase: 10%
    is paraphrase: 90%

    >>> # Should not be paraphrase
    >>> for i in range(len(classes)):
    ...     print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
    not paraphrase: 94%
    is paraphrase: 6%
    >>> ## TENSORFLOW CODE
    >>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    >>> model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

    >>> classes = ["not paraphrase", "is paraphrase"]

    >>> sequence_0 = "The company HuggingFace is based in New York City"
    >>> sequence_1 = "Apples are especially bad for your health"
    >>> sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    >>> paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="tf")
    >>> not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="tf")

    >>> paraphrase_classification_logits = model(paraphrase)[0]
    >>> not_paraphrase_classification_logits = model(not_paraphrase)[0]

    >>> paraphrase_results = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
    >>> not_paraphrase_results = tf.nn.softmax(not_paraphrase_classification_logits, axis=1).numpy()[0]

    >>> # Should be paraphrase
    >>> for i in range(len(classes)):
    ...     print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
    not paraphrase: 10%
    is paraphrase: 90%

    >>> # Should not be paraphrase
    >>> for i in range(len(classes)):
    ...     print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
    not paraphrase: 94%
    is paraphrase: 6%

Extractive Question Answering
----------------------------------------------------

Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the `run_squad.py`.

Here is an example using the pipelines do to question answering: extracting an answer from a text given a question.
It leverages a fine-tuned model on SQuAD.

.. code-block::

    >>> from transformers import pipeline

    >>> nlp = pipeline("question-answering")

    >>> context = r"""
    ... Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
    ... question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
    ... a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
    ... """

This returns an answer extracted from the text, a confidence score, alongside "start" and "end" values which
are the positions of the extracted answer in the text.

.. code-block::

    >>> result = nlp(question="What is extractive question answering?", context=context)
    >>> print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
    Answer: 'the task of extracting an answer from a text given a question.', score: 0.6226, start: 34, end: 96

    >>> result = nlp(question="What is a good example of a question answering dataset?", context=context)
    >>> print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
    Answer: 'SQuAD dataset,', score: 0.5053, start: 147, end: 161


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

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    >>> model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    >>> text = r"""
    ... 🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    ... architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    ... Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    ... TensorFlow 2.0 and PyTorch.
    ... """

    >>> questions = [
    ...     "How many pretrained models are available in 🤗 Transformers?",
    ...     "What does 🤗 Transformers provide?",
    ...     "🤗 Transformers provides interoperability between which frameworks?",
    ... ]

    >>> for question in questions:
    ...     inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    ...     input_ids = inputs["input_ids"].tolist()[0]
    ...
    ...     text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    ...     answer_start_scores, answer_end_scores = model(**inputs)
    ...
    ...     answer_start = torch.argmax(
    ...         answer_start_scores
    ...     )  # Get the most likely beginning of answer with the argmax of the score
    ...     answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    ...
    ...     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    ...
    ...     print(f"Question: {question}")
    ...     print(f"Answer: {answer}")
    Question: How many pretrained models are available in 🤗 Transformers?
    Answer: over 32 +
    Question: What does 🤗 Transformers provide?
    Answer: general - purpose architectures
    Question: 🤗 Transformers provides interoperability between which frameworks?
    Answer: tensorflow 2 . 0 and pytorch
    >>> ## TENSORFLOW CODE
    >>> from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    >>> model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    >>> text = r"""
    ... 🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    ... architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    ... Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    ... TensorFlow 2.0 and PyTorch.
    ... """

    >>> questions = [
    ...     "How many pretrained models are available in 🤗 Transformers?",
    ...     "What does 🤗 Transformers provide?",
    ...     "🤗 Transformers provides interoperability between which frameworks?",
    ... ]

    >>> for question in questions:
    ...     inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="tf")
    ...     input_ids = inputs["input_ids"].numpy()[0]
    ...
    ...     text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    ...     answer_start_scores, answer_end_scores = model(inputs)
    ...
    ...     answer_start = tf.argmax(
    ...         answer_start_scores, axis=1
    ...     ).numpy()[0]  # Get the most likely beginning of answer with the argmax of the score
    ...     answer_end = (
    ...         tf.argmax(answer_end_scores, axis=1) + 1
    ...     ).numpy()[0]  # Get the most likely end of answer with the argmax of the score
    ...     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    ...
    ...     print(f"Question: {question}")
    ...     print(f"Answer: {answer}")
    Question: How many pretrained models are available in 🤗 Transformers?
    Answer: over 32 +
    Question: What does 🤗 Transformers provide?
    Answer: general - purpose architectures
    Question: 🤗 Transformers provides interoperability between which frameworks?
    Answer: tensorflow 2 . 0 and pytorch



Language Modeling
----------------------------------------------------

Language modeling is the task of fitting a model to a corpus, which can be domain specific. All popular transformer
based models are trained using a variant of language modeling, e.g. BERT with masked language modeling, GPT-2 with
causal language modeling.

Language modeling can be useful outside of pre-training as well, for example to shift the model distribution to be
domain-specific: using a language model trained over a very large corpus, and then fine-tuning it to a news dataset
or on scientific papers e.g. `LysandreJik/arxiv-nlp <https://huggingface.co/lysandre/arxiv-nlp>`__.

Masked Language Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Masked language modeling is the task of masking tokens in a sequence with a masking token, and prompting the model to
fill that mask with an appropriate token. This allows the model to attend to both the right context (tokens on the
right of the mask) and the left context (tokens on the left of the mask). Such a training creates a strong basis
for downstream tasks requiring bi-directional context such as SQuAD (question answering,
see `Lewis, Lui, Goyal et al. <https://arxiv.org/abs/1910.13461>`__, part 4.2).

Here is an example of using pipelines to replace a mask from a sequence:

.. code-block::

    >>> from transformers import pipeline

    >>> nlp = pipeline("fill-mask")

This outputs the sequences with the mask filled, the confidence score as well as the token id in the tokenizer
vocabulary:

.. code-block::

    >>> from pprint import pprint
    >>> pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
    [{'score': 0.1792745739221573,
      'sequence': '<s>HuggingFace is creating a tool that the community uses to '
                  'solve NLP tasks.</s>',
      'token': 3944,
      'token_str': 'Ġtool'},
     {'score': 0.11349421739578247,
      'sequence': '<s>HuggingFace is creating a framework that the community uses '
                  'to solve NLP tasks.</s>',
      'token': 7208,
      'token_str': 'Ġframework'},
     {'score': 0.05243554711341858,
      'sequence': '<s>HuggingFace is creating a library that the community uses to '
                  'solve NLP tasks.</s>',
      'token': 5560,
      'token_str': 'Ġlibrary'},
     {'score': 0.03493533283472061,
      'sequence': '<s>HuggingFace is creating a database that the community uses '
                  'to solve NLP tasks.</s>',
      'token': 8503,
      'token_str': 'Ġdatabase'},
     {'score': 0.02860250137746334,
      'sequence': '<s>HuggingFace is creating a prototype that the community uses '
                  'to solve NLP tasks.</s>',
      'token': 17715,
      'token_str': 'Ġprototype'}]

Here is an example doing masked language modeling using a model and a tokenizer. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a DistilBERT model and
  loads it with the weights stored in the checkpoint.
- Define a sequence with a masked token, placing the :obj:`tokenizer.mask_token` instead of a word.
- Encode that sequence into IDs and find the position of the masked token in that list of IDs.
- Retrieve the predictions at the index of the mask token: this tensor has the same size as the vocabulary, and the
  values are the scores attributed to each token. The model gives higher score to tokens he deems probable in that
  context.
- Retrieve the top 5 tokens using the PyTorch :obj:`topk` or TensorFlow :obj:`top_k` methods.
- Replace the mask token by the tokens and print the results

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoModelWithLMHead, AutoTokenizer
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    >>> model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

    >>> sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

    >>> input = tokenizer.encode(sequence, return_tensors="pt")
    >>> mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

    >>> token_logits = model(input)[0]
    >>> mask_token_logits = token_logits[0, mask_token_index, :]

    >>> top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    >>> ## TENSORFLOW CODE
    >>> from transformers import TFAutoModelWithLMHead, AutoTokenizer
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    >>> model = TFAutoModelWithLMHead.from_pretrained("distilbert-base-cased")

    >>> sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

    >>> input = tokenizer.encode(sequence, return_tensors="tf")
    >>> mask_token_index = tf.where(input == tokenizer.mask_token_id)[0, 1]

    >>> token_logits = model(input)[0]
    >>> mask_token_logits = token_logits[0, mask_token_index, :]

    >>> top_5_tokens = tf.math.top_k(mask_token_logits, 5).indices.numpy()


This prints five sequences, with the top 5 tokens predicted by the model:

.. code-block::

    >>> for token in top_5_tokens:
    ...     print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
    Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.


Causal Language Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Causal language modeling is the task of predicting the token following a sequence of tokens. In this situation, the
model only attends to the left context (tokens on the left of the mask). Such a training is particularly interesting
for generation tasks.

Usually, the next token is predicted by sampling from the logits of the last hidden state the model produces from the input sequence.

Here is an example using the tokenizer and model and leveraging the :func:`~transformers.PreTrainedModel.top_k_top_p_filtering` method to sample the next token following an input sequence of tokens.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
    >>> import torch
    >>> from torch.nn import functional as F

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelWithLMHead.from_pretrained("gpt2")

    >>> sequence = f"Hugging Face is based in DUMBO, New York City, and "

    >>> input_ids = tokenizer.encode(sequence, return_tensors="pt")

    >>> # get logits of last hidden state
    >>> next_token_logits = model(input_ids)[0][:, -1, :]

    >>> # filter
    >>> filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

    >>> # sample
    >>> probs = F.softmax(filtered_next_token_logits, dim=-1)
    >>> next_token = torch.multinomial(probs, num_samples=1)

    >>> generated = torch.cat([input_ids, next_token], dim=-1)

    >>> resulting_string = tokenizer.decode(generated.tolist()[0])
    >>> ## TENSORFLOW CODE
    >>> from transformers import TFAutoModelWithLMHead, AutoTokenizer, tf_top_k_top_p_filtering
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = TFAutoModelWithLMHead.from_pretrained("gpt2")

    >>> sequence = f"Hugging Face is based in DUMBO, New York City, and "

    >>> input_ids = tokenizer.encode(sequence, return_tensors="tf")

    >>> # get logits of last hidden state
    >>> next_token_logits = model(input_ids)[0][:, -1, :]

    >>> # filter
    >>> filtered_next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

    >>> # sample
    >>> next_token = tf.random.categorical(filtered_next_token_logits, dtype=tf.int32, num_samples=1)

    >>> generated = tf.concat([input_ids, next_token], axis=1)

    >>> resulting_string = tokenizer.decode(generated.numpy().tolist()[0])


This outputs a (hopefully) coherent next token following the original sequence, which is in our case is the word *has*:

.. code-block::

    print(resulting_string)
    Hugging Face is based in DUMBO, New York City, and has

In the next section, we show how this functionality is leveraged in :func:`~transformers.PreTrainedModel.generate` to generate multiple tokens up to a user-defined length.

Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In text generation (*a.k.a* *open-ended text generation*) the goal is to create a coherent portion of text that is a continuation from the given context. As an example, is it shown how *GPT-2* can be used in pipelines to generate text. As a default all models apply *Top-K* sampling when used in pipelines as configured in their respective configurations (see `gpt-2 config <https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json>`_ for example).

.. code-block::

    >>> from transformers import pipeline

    >>> text_generator = pipeline("text-generation")
    >>> print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))
    [{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]



Here the model generates a random text with a total maximal length of *50* tokens from context *"As far as I am concerned, I will"*.
The default arguments of ``PreTrainedModel.generate()`` can directly be overriden in the pipeline as is shown above for the argument ``max_length``.

Here is an example for text generation using XLNet and its tokenzier. 

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoModelWithLMHead, AutoTokenizer

    >>> model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased")
    >>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

    >>> # Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
    >>> PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    ... (except for Alexei and Maria) are discovered.
    ... The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    ... remainder of the story. 1883 Western Siberia,
    ... a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    ... Rasputin has a vision and denounces one of the men as a horse thief. Although his
    ... father initially slaps him for making such an accusation, Rasputin watches as the
    ... man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    ... the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    ... with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    >>> prompt = "Today the weather is really nice and I am planning on "
    >>> inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")

    >>> prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    >>> outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
    >>> generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]

    >>> ## TENSORFLOW CODE
    >>> from transformers import TFAutoModelWithLMHead, AutoTokenizer

    >>> model = TFAutoModelWithLMHead.from_pretrained("xlnet-base-cased")
    >>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

    >>> # Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
    >>> PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    ... (except for Alexei and Maria) are discovered.
    ... The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    ... remainder of the story. 1883 Western Siberia,
    ... a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    ... Rasputin has a vision and denounces one of the men as a horse thief. Although his
    ... father initially slaps him for making such an accusation, Rasputin watches as the
    ... man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    ... the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    ... with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    >>> prompt = "Today the weather is really nice and I am planning on "
    >>> inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="tf")

    >>> prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    >>> outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
    >>> generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]

.. code-block::

    print(generated)

Text generation is currently possible with *GPT-2*, *OpenAi-GPT*, *CTRL*, *XLNet*, *Transfo-XL* and *Reformer* in PyTorch and for most models in Tensorflow as well. As can be seen in the example above *XLNet* and *Transfo-xl* often need to be padded to work well.
GPT-2 is usually a good choice for *open-ended text generation* because it was trained on millions on webpages with a causal language modeling objective.

For more information on how to apply different decoding strategies for text generation, please also refer to our generation blog post `here <https://huggingface.co/blog/how-to-generate>`_.


Named Entity Recognition
----------------------------------------------------

Named Entity Recognition (NER) is the task of classifying tokens according to a class, for example identifying a
token as a person, an organisation or a location.
An example of a named entity recognition dataset is the CoNLL-2003 dataset, which is entirely based on that task.
If you would like to fine-tune a model on an NER task, you may leverage the `ner/run_ner.py` (PyTorch),
`ner/run_pl_ner.py` (leveraging pytorch-lightning) or the `ner/run_tf_ner.py` (TensorFlow) scripts.

Here is an example using the pipelines do to named entity recognition, trying to identify tokens as belonging to one
of 9 classes:

- O, Outside of a named entity
- B-MIS, Beginning of a miscellaneous entity right after another miscellaneous entity
- I-MIS, Miscellaneous entity
- B-PER, Beginning of a person's name right after another person's name
- I-PER, Person's name
- B-ORG, Beginning of an organisation right after another organisation
- I-ORG, Organisation
- B-LOC, Beginning of a location right after another location
- I-LOC, Location

It leverages a fine-tuned model on CoNLL-2003, fine-tuned by `@stefan-it <https://github.com/stefan-it>`__ from
`dbmdz <https://github.com/dbmdz>`__.

.. code-block::

    >>> from transformers import pipeline

    >>> nlp = pipeline("ner")

    >>> sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very"
    ...            "close to the Manhattan Bridge which is visible from the window."


This outputs a list of all words that have been identified as an entity from the 9 classes defined above. Here is the
expected results:

.. code-block::

    print(nlp(sequence))

    [
        {'word': 'Hu', 'score': 0.9995632767677307, 'entity': 'I-ORG'},
        {'word': '##gging', 'score': 0.9915938973426819, 'entity': 'I-ORG'},
        {'word': 'Face', 'score': 0.9982671737670898, 'entity': 'I-ORG'},
        {'word': 'Inc', 'score': 0.9994403719902039, 'entity': 'I-ORG'},
        {'word': 'New', 'score': 0.9994346499443054, 'entity': 'I-LOC'},
        {'word': 'York', 'score': 0.9993270635604858, 'entity': 'I-LOC'},
        {'word': 'City', 'score': 0.9993864893913269, 'entity': 'I-LOC'},
        {'word': 'D', 'score': 0.9825621843338013, 'entity': 'I-LOC'},
        {'word': '##UM', 'score': 0.936983048915863, 'entity': 'I-LOC'},
        {'word': '##BO', 'score': 0.8987102508544922, 'entity': 'I-LOC'},
        {'word': 'Manhattan', 'score': 0.9758241176605225, 'entity': 'I-LOC'},
        {'word': 'Bridge', 'score': 0.990249514579773, 'entity': 'I-LOC'}
    ]

Note how the words "Hugging Face" have been identified as an organisation, and "New York City", "DUMBO" and
"Manhattan Bridge" have been identified as locations.

Here is an example doing named entity recognition using a model and a tokenizer. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and
  loads it with the weights stored in the checkpoint.
- Define the label list with which the model was trained on.
- Define a sequence with known entities, such as "Hugging Face" as an organisation and "New York City" as a location.
- Split words into tokens so that they can be mapped to the predictions. We use a small hack by firstly completely
  encoding and decoding the sequence, so that we're left with a string that contains the special tokens.
- Encode that sequence into IDs (special tokens are added automatically).
- Retrieve the predictions by passing the input to the model and getting the first output. This results in a
  distribution over the 9 possible classes for each token. We take the argmax to retrieve the most likely class
  for each token.
- Zip together each token with its prediction and print it.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoModelForTokenClassification, AutoTokenizer
    >>> import torch

    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    >>> label_list = [
    ...     "O",       # Outside of a named entity
    ...     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    ...     "I-MISC",  # Miscellaneous entity
    ...     "B-PER",   # Beginning of a person's name right after another person's name
    ...     "I-PER",   # Person's name
    ...     "B-ORG",   # Beginning of an organisation right after another organisation
    ...     "I-ORG",   # Organisation
    ...     "B-LOC",   # Beginning of a location right after another location
    ...     "I-LOC"    # Location
    ... ]

    >>> sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
    ...            "close to the Manhattan Bridge."

    >>> # Bit of a hack to get the tokens with the special tokens
    >>> tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    >>> inputs = tokenizer.encode(sequence, return_tensors="pt")

    >>> outputs = model(inputs)[0]
    >>> predictions = torch.argmax(outputs, dim=2)
    >>> ## TENSORFLOW CODE
    >>> from transformers import TFAutoModelForTokenClassification, AutoTokenizer
    >>> import tensorflow as tf

    >>> model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    >>> label_list = [
    ...     "O",       # Outside of a named entity
    ...     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    ...     "I-MISC",  # Miscellaneous entity
    ...     "B-PER",   # Beginning of a person's name right after another person's name
    ...     "I-PER",   # Person's name
    ...     "B-ORG",   # Beginning of an organisation right after another organisation
    ...     "I-ORG",   # Organisation
    ...     "B-LOC",   # Beginning of a location right after another location
    ...     "I-LOC"    # Location
    ... ]

    >>> sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
    ...            "close to the Manhattan Bridge."

    >>> # Bit of a hack to get the tokens with the special tokens
    >>> tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    >>> inputs = tokenizer.encode(sequence, return_tensors="tf")

    >>> outputs = model(inputs)[0]
    >>> predictions = tf.argmax(outputs, axis=2)


This outputs a list of each token mapped to their prediction. Differently from the pipeline, here every token has
a prediction as we didn't remove the "0" class which means that no particular entity was found on that token. The
following array should be the output:

.. code-block::

    >>> print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
    [('[CLS]', 'O'), ('Hu', 'I-ORG'), ('##gging', 'I-ORG'), ('Face', 'I-ORG'), ('Inc', 'I-ORG'), ('.', 'O'), ('is', 'O'), ('a', 'O'), ('company', 'O'), ('based', 'O'), ('in', 'O'), ('New', 'I-LOC'), ('York', 'I-LOC'), ('City', 'I-LOC'), ('.', 'O'), ('Its', 'O'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'), ('D', 'I-LOC'), ('##UM', 'I-LOC'), ('##BO', 'I-LOC'), (',', 'O'), ('therefore', 'O'), ('very', 'O'), ('##c', 'O'), ('##lose', 'O'), ('to', 'O'), ('the', 'O'), ('Manhattan', 'I-LOC'), ('Bridge', 'I-LOC'), ('.', 'O'), ('[SEP]', 'O')]

Summarization
----------------------------------------------------

Summarization is the task of summarizing a text / an article into a shorter text.

An example of a summarization dataset is the CNN / Daily Mail dataset, which consists of long news articles and was created for the task of summarization.
If you would like to fine-tune a model on a summarization task, you may leverage the ``examples/summarization/bart/run_train.sh`` (leveraging pytorch-lightning) script.

Here is an example using the pipelines do to summarization. 
It leverages a Bart model that was fine-tuned on the CNN / Daily Mail data set.

.. code-block::

    >>> from transformers import pipeline

    >>> summarizer = pipeline("summarization")

    >>> ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    ... A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    ... Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    ... In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    ... Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    ... 2010 marriage license application, according to court documents.
    ... Prosecutors said the marriages were part of an immigration scam.
    ... On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    ... After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    ... Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    ... All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    ... Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    ... Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    ... The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    ... Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    ... Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    ... If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    ... """

Because the summarization pipeline depends on the ``PretrainedModel.generate()`` method, we can override the default arguments 
of ``PretrainedModel.generate()`` directly in the pipeline as is shown for ``max_length`` and ``min_length`` above.
This outputs the following summary:

.. code-block::

    >>> print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
    [{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]

Here is an example doing summarization using a model and a tokenizer. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. Summarization is usually done using an encoder-decoder model, such as ``Bart`` or ``T5``.
- Define the article that should be summarizaed.
- Leverage the ``PretrainedModel.generate()`` method.
- Add the T5 specific prefix "summarize: ".

Here Google`s T5 model is used that was only pre-trained on a multi-task mixed data set (including CNN / Daily Mail), but nevertheless yields very good results.
.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoModelWithLMHead, AutoTokenizer

    >>> model = AutoModelWithLMHead.from_pretrained("t5-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")

    >>> # T5 uses a max_length of 512 so we cut the article to 512 tokens.
    >>> inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
    >>> outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    >>> ## TENSORFLOW CODE
    >>> from transformers import TFAutoModelWithLMHead, AutoTokenizer

    >>> model = TFAutoModelWithLMHead.from_pretrained("t5-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")

    >>> # T5 uses a max_length of 512 so we cut the article to 512 tokens.
    >>> inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="tf", max_length=512)
    >>> outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

Translation
----------------------------------------------------

Translation is the task of translating a text from one language to another.

An example of a translation dataset is the WMT English to German dataset, which has English sentences as the input data 
and German sentences as the target data.

Here is an example using the pipelines do to translation. 
It leverages a T5 model that was only pre-trained on a multi-task mixture dataset (including WMT), but yields impressive 
translation results nevertheless.

.. code-block::

    >>> from transformers import pipeline

    >>> translator = pipeline("translation_en_to_de")
    >>> print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))
    [{'translation_text': 'Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.'}]

Because the translation pipeline depends on the ``PretrainedModel.generate()`` method, we can override the default arguments 
of ``PretrainedModel.generate()`` directly in the pipeline as is shown for ``max_length`` above.
This outputs the following translation into German:

::

  Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.
  
Here is an example doing translation using a model and a tokenizer. The process is the following:

- Instantiate a tokenizer and a model from the checkpoint name. Summarization is usually done using an encoder-decoder model, such as ``Bart`` or ``T5``.
- Define the article that should be summarizaed.
- Leverage the ``PretrainedModel.generate()`` method.
- Add the T5 specific prefix "translate English to German: "

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoModelWithLMHead, AutoTokenizer

    >>> model = AutoModelWithLMHead.from_pretrained("t5-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")

    >>> inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
    >>> outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

    >>> print(outputs)
    tensor([[    0, 11560,  3896,  8881,   229,   236,     3, 14366, 15377,   181,
             11216,    16,   368,  1060,    64,  1919,     5]])
    >>> ## TENSORFLOW CODE
    >>> from transformers import TFAutoModelWithLMHead, AutoTokenizer

    >>> model = TFAutoModelWithLMHead.from_pretrained("t5-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")

    >>> inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="tf")
    >>> outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

    >>> print(outputs)
    tf.Tensor(
    [[    0 11560  3896  8881   229   236     3 14366 15377   181 11216    16
        368  1060    64  1919     5]], shape=(1, 17), dtype=int32)
