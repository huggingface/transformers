Quick tour
==========

Let's have a quick look at the 🤗 Transformers library features. The library downloads pretrained models for
Natural Language Understanding (NLU), like analyzing the sentiment of a text, and Natural Language Generation (NLG), 
like completing a prompt with new text or translating in another language.

First we will see how to easily leverage the pipeline API to 

.. _pipeline:

Getting started on a task with a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to use a pretrained model on a given task is to use :func:`pipeline`. Let's try this for sentiment
analysis:

::

    from transformers import pipeline
    pipe = pipeline('sentiment-analysis')

When typing this command for the first time, a pretrained model and its tokenizer are downloaded and cached. We will
look at both later on, but as an introduction the tokenizer's job is to preprocess the text for the model, which is
then responsible for making predictions. The pipeline groups all of that together, and post-process the predictions to
make them readable. For instance

::

    pipe('We are very happy to show you the Transformers library.')

will return something like:

::

    [{'label': 'POSITIVE', 'score': 0.999799370765686}]

That's encouraging! You can use it on a list of sentences, which will be preprocessed then fed to the model as a 
`batch`, returning a list of dictionaries like this one:

::

    pipe(["We are very happy to show you the Transformers library.",
          "We hope you don't hate it."])

should return

::

    [{'label': 'POSITIVE', 'score': 0.999799370765686},
     {'label': 'NEGATIVE', 'score': 0.5308589935302734}]

You can see the second sentence has been classified as negative (it needs to be positive or negative) but its score is
fairly neutral.

By default, the model downloaded for this pipeline is called "distilbert-base-uncased-finetuned-sst-2-english". We can
look at its `model page <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__ to get more
information about it. It uses the :doc:`DistilBERT architecture </model_doc/distilbert>` and has been fine-tuned on a
dataset called SST-2 for the sentiment analysis task.

Let's say we want to use another model; for instance, one that has been trained on French data, we can search through
the `community uploaded models <https://huggingface.co/models>`__. Applying the tags "French" and "text-classification"
gives back a suggestion "nlptown/bert-base-multilingual-uncased-sentiment". Let's see how we can use it.

We will need two classes for this. The first is :class:`AutoTokenizer`, which we will use to download the tokenizer
associated to the model we picked and instantiates it. The second is :class:`AutoModelForSequenceClassification`, which
we will use to download the model itself. Note that if we were using the library for another task, the class of the
model would change. The :doc:`task summary </task_summary>` tutorial summarizes which class is used for which task.

::

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

Now, to download the models and tokenizer we found previously, we just have to use the ``from_pretrained`` method 
(feel free to replace ``model_name`` by any other model):

::

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

This pipeline can now deal with texts in English, French, but also Dutch, German, Italian and Spanish.

.. _pretrained-model:

Using the model directly
~~~~~~~~~~~~~~~~~~~~~~~~

