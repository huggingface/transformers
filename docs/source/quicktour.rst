.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Quick tour
=======================================================================================================================

Let's have a quick look at the 🤗 Transformers library features. The library downloads pretrained models for Natural
Language Understanding (NLU) tasks, such as analyzing the sentiment of a text, and Natural Language Generation (NLG),
such as completing a prompt with new text or translating in another language.

First we will see how to easily leverage the pipeline API to quickly use those pretrained models at inference. Then, we
will dig a little bit more and see how the library gives you access to those models and helps you preprocess your data.

.. note::

    All code examples presented in the documentation have a switch on the top left for Pytorch versus TensorFlow. If
    not, the code is expected to work for both backends without any change needed.

Getting started on a task with a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to use a pretrained model on a given task is to use :func:`~transformers.pipeline`.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/tiZFewofSLM" title="YouTube video player"
   frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
   picture-in-picture" allowfullscreen></iframe>

🤗 Transformers provides the following tasks out of the box:

- Sentiment analysis: is a text positive or negative?
- Text generation (in English): provide a prompt and the model will generate what follows.
- Name entity recognition (NER): in an input sentence, label each word with the entity it represents (person, place,
  etc.)
- Question answering: provide the model with some context and a question, extract the answer from the context.
- Filling masked text: given a text with masked words (e.g., replaced by ``[MASK]``), fill the blanks.
- Summarization: generate a summary of a long text.
- Translation: translate a text in another language.
- Feature extraction: return a tensor representation of the text.

Let's see how this work for sentiment analysis (the other tasks are all covered in the :doc:`task summary
</task_summary>`):

.. code-block::

    >>> from transformers import pipeline
    >>> classifier = pipeline('sentiment-analysis')

When typing this command for the first time, a pretrained model and its tokenizer are downloaded and cached. We will
look at both later on, but as an introduction the tokenizer's job is to preprocess the text for the model, which is
then responsible for making predictions. The pipeline groups all of that together, and post-process the predictions to
make them readable. For instance:


.. code-block::

    >>> classifier('We are very happy to show you the 🤗 Transformers library.')
    [{'label': 'POSITIVE', 'score': 0.9997795224189758}]

That's encouraging! You can use it on a list of sentences, which will be preprocessed then fed to the model as a
`batch`, returning a list of dictionaries like this one:

.. code-block::

    >>> results = classifier(["We are very happy to show you the 🤗 Transformers library.",
    ...            "We hope you don't hate it."])
    >>> for result in results:
    ...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    label: POSITIVE, with score: 0.9998
    label: NEGATIVE, with score: 0.5309

You can see the second sentence has been classified as negative (it needs to be positive or negative) but its score is
fairly neutral.

By default, the model downloaded for this pipeline is called "distilbert-base-uncased-finetuned-sst-2-english". We can
look at its `model page <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__ to get more
information about it. It uses the :doc:`DistilBERT architecture </model_doc/distilbert>` and has been fine-tuned on a
dataset called SST-2 for the sentiment analysis task.

Let's say we want to use another model; for instance, one that has been trained on French data. We can search through
the `model hub <https://huggingface.co/models>`__ that gathers models pretrained on a lot of data by research labs, but
also community models (usually fine-tuned versions of those big models on a specific dataset). Applying the tags
"French" and "text-classification" gives back a suggestion "nlptown/bert-base-multilingual-uncased-sentiment". Let's
see how we can use it.

You can directly pass the name of the model to use to :func:`~transformers.pipeline`:

.. code-block::

    >>> classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

This classifier can now deal with texts in English, French, but also Dutch, German, Italian and Spanish! You can also
replace that name by a local folder where you have saved a pretrained model (see below). You can also pass a model
object and its associated tokenizer.

We will need two classes for this. The first is :class:`~transformers.AutoTokenizer`, which we will use to download the
tokenizer associated to the model we picked and instantiate it. The second is
:class:`~transformers.AutoModelForSequenceClassification` (or
:class:`~transformers.TFAutoModelForSequenceClassification` if you are using TensorFlow), which we will use to download
the model itself. Note that if we were using the library on an other task, the class of the model would change. The
:doc:`task summary </task_summary>` tutorial summarizes which class is used for which task.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
    >>> ## TENSORFLOW CODE
    >>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

Now, to download the models and tokenizer we found previously, we just have to use the
:func:`~transformers.AutoModelForSequenceClassification.from_pretrained` method (feel free to replace ``model_name`` by
any other model from the model hub):

.. code-block::

    >>> ## PYTORCH CODE
    >>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    >>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    >>> ## TENSORFLOW CODE
    >>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    >>> # This model only exists in PyTorch, so we use the `from_pt` flag to import that model in TensorFlow.
    >>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

If you don't find a model that has been pretrained on some data similar to yours, you will need to fine-tune a
pretrained model on your data. We provide :doc:`example scripts </examples>` to do so. Once you're done, don't forget
to share your fine-tuned model on the hub with the community, using :doc:`this tutorial </model_sharing>`.

.. _pretrained-model:

Under the hood: pretrained models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's now see what happens beneath the hood when using those pipelines.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/AhChOFRegn4" title="YouTube video player"
   frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
   picture-in-picture" allowfullscreen></iframe>

As we saw, the model and tokenizer are created using the :obj:`from_pretrained` method:

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
    >>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    >>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> ## TENSORFLOW CODE
    >>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    >>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    >>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)

Using the tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We mentioned the tokenizer is responsible for the preprocessing of your texts. First, it will split a given text in
words (or part of words, punctuation symbols, etc.) usually called `tokens`. There are multiple rules that can govern
that process (you can learn more about them in the :doc:`tokenizer summary <tokenizer_summary>`), which is why we need
to instantiate the tokenizer using the name of the model, to make sure we use the same rules as when the model was
pretrained.

The second step is to convert those `tokens` into numbers, to be able to build a tensor out of them and feed them to
the model. To do this, the tokenizer has a `vocab`, which is the part we download when we instantiate it with the
:obj:`from_pretrained` method, since we need to use the same `vocab` as when the model was pretrained.

To apply these steps on a given text, we can just feed it to our tokenizer:

.. code-block::

    >>> inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")

This returns a dictionary string to list of ints. It contains the `ids of the tokens <glossary.html#input-ids>`__, as
mentioned before, but also additional arguments that will be useful to the model. Here for instance, we also have an
`attention mask <glossary.html#attention-mask>`__ that the model will use to have a better understanding of the
sequence:


.. code-block::

    >>> print(inputs)
    {'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

You can pass a list of sentences directly to your tokenizer. If your goal is to send them through your model as a
batch, you probably want to pad them all to the same length, truncate them to the maximum length the model can accept
and get tensors back. You can specify all of that to the tokenizer:

.. code-block::

    >>> ## PYTORCH CODE
    >>> pt_batch = tokenizer(
    ...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    ...     padding=True,
    ...     truncation=True,
    ...     max_length=512,
    ...     return_tensors="pt"
    ... )
    >>> ## TENSORFLOW CODE
    >>> tf_batch = tokenizer(
    ...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    ...     padding=True,
    ...     truncation=True,
    ...     max_length=512,
    ...     return_tensors="tf"
    ... )

The padding is automatically applied on the side expected by the model (in this case, on the right), with the padding
token the model was pretrained with. The attention mask is also adapted to take the padding into account:

.. code-block::

    >>> ## PYTORCH CODE
    >>> for key, value in pt_batch.items():
    ...     print(f"{key}: {value.numpy().tolist()}")
    input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
    attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
    >>> ## TENSORFLOW CODE
    >>> for key, value in tf_batch.items():
    ...     print(f"{key}: {value.numpy().tolist()}")
    input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
    attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]

You can learn more about tokenizers :doc:`here <preprocessing>`.

Using the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once your input has been preprocessed by the tokenizer, you can send it directly to the model. As we mentioned, it will
contain all the relevant information the model needs. If you're using a TensorFlow model, you can pass the dictionary
keys directly to tensors, for a PyTorch model, you need to unpack the dictionary by adding :obj:`**`.

.. code-block::

    >>> ## PYTORCH CODE
    >>> pt_outputs = pt_model(**pt_batch)
    >>> ## TENSORFLOW CODE
    >>> tf_outputs = tf_model(tf_batch)

In 🤗 Transformers, all outputs are objects that contain the model's final activations along with other metadata. These
objects are described in greater detail :doc:`here <main_classes/output>`. For now, let's inspect the output ourselves:

.. code-block::

    >>> ## PYTORCH CODE
    >>> print(pt_outputs)
    SequenceClassifierOutput(loss=None, logits=tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
    >>> ## TENSORFLOW CODE
    >>> print(tf_outputs)
    TFSequenceClassifierOutput(loss=None, logits=<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[-4.0832963 ,  4.3364143 ],
           [ 0.081807  , -0.04178282]], dtype=float32)>, hidden_states=None, attentions=None)

Notice how the output object has a ``logits`` attribute. You can use this to access the model's final activations.

.. note::

    All 🤗 Transformers models (PyTorch or TensorFlow) return the activations of the model *before* the final activation
    function (like SoftMax) since this final activation function is often fused with the loss.

Let's apply the SoftMax activation to get predictions.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from torch import nn
    >>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
    >>> ## TENSORFLOW CODE
    >>> import tensorflow as tf
    >>> tf.nn.softmax(tf_outputs.logits, axis=-1)

We can see we get the numbers from before:

.. code-block::

    >>> ## TENSORFLOW CODE
    >>> print(tf_predictions)
    tf.Tensor(
    [[2.2042994e-04 9.9977952e-01]
     [5.3086340e-01 4.6913657e-01]], shape=(2, 2), dtype=float32)
    >>> ## PYTORCH CODE
    >>> print(pt_predictions)
    tensor([[2.2043e-04, 9.9978e-01],
            [5.3086e-01, 4.6914e-01]], grad_fn=<SoftmaxBackward>)

If you provide the model with labels in addition to inputs, the model output object will also contain a ``loss``
attribute:

.. code-block::

    >>> ## PYTORCH CODE
    >>> import torch
    >>> pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
    >>> print(pt_outputs)
    SequenceClassifierOutput(loss=tensor(0.3167, grad_fn=<NllLossBackward>), logits=tensor([[-4.0833,  4.3364],
    [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
    >>> ## TENSORFLOW CODE
    >>> import tensorflow as tf
    >>> tf_outputs = tf_model(tf_batch, labels = tf.constant([1, 0]))
    >>> print(tf_outputs)
    TFSequenceClassifierOutput(loss=<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2.2051287e-04, 6.3326043e-01], dtype=float32)>, logits=<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[-4.0832963 ,  4.3364143 ],
           [ 0.081807  , -0.04178282]], dtype=float32)>, hidden_states=None, attentions=None)

Models are standard `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ or `tf.keras.Model
<https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ so you can use them in your usual training loop. 🤗
Transformers also provides a :class:`~transformers.Trainer` (or :class:`~transformers.TFTrainer` if you are using
TensorFlow) class to help with your training (taking care of things such as distributed training, mixed precision,
etc.). See the :doc:`training tutorial <training>` for more details.

.. note::

    Pytorch model outputs are special dataclasses so that you can get autocompletion for their attributes in an IDE.
    They also behave like a tuple or a dictionary (e.g., you can index with an integer, a slice or a string) in which
    case the attributes not set (that have :obj:`None` values) are ignored.

Once your model is fine-tuned, you can save it with its tokenizer in the following way:

.. code-block::

    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

You can then load this model back using the :func:`~transformers.AutoModel.from_pretrained` method by passing the
directory name instead of the model name. One cool feature of 🤗 Transformers is that you can easily switch between
PyTorch and TensorFlow: any model saved as before can be loaded back either in PyTorch or TensorFlow. If you are
loading a saved PyTorch model in a TensorFlow model, use :func:`~transformers.TFAutoModel.from_pretrained` like this:

.. code-block::

    from transformers import TFAutoModel
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = TFAutoModel.from_pretrained(save_directory, from_pt=True)

and if you are loading a saved TensorFlow model in a PyTorch model, you should use the following code:

.. code-block::

    from transformers import AutoModel
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModel.from_pretrained(save_directory, from_tf=True)

Lastly, you can also ask the model to return all hidden states and all attention weights if you need them:


.. code-block::

    >>> ## PYTORCH CODE
    >>> pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
    >>> all_hidden_states  = pt_outputs.hidden_states 
    >>> all_attentions = pt_outputs.attentions
    >>> ## TENSORFLOW CODE
    >>> tf_outputs = tf_model(tf_batch, output_hidden_states=True, output_attentions=True)
    >>> all_hidden_states =  tf_outputs.hidden_states
    >>> all_attentions = tf_outputs.attentions

Accessing the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :obj:`AutoModel` and :obj:`AutoTokenizer` classes are just shortcuts that will automatically work with any
pretrained model. Behind the scenes, the library has one model class per combination of architecture plus class, so the
code is easy to access and tweak if you need to.

In our previous example, the model was called "distilbert-base-uncased-finetuned-sst-2-english", which means it's using
the :doc:`DistilBERT </model_doc/distilbert>` architecture. As
:class:`~transformers.AutoModelForSequenceClassification` (or
:class:`~transformers.TFAutoModelForSequenceClassification` if you are using TensorFlow) was used, the model
automatically created is then a :class:`~transformers.DistilBertForSequenceClassification`. You can look at its
documentation for all details relevant to that specific model, or browse the source code. This is how you would
directly instantiate model and tokenizer without the auto magic:

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    >>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    >>> model = DistilBertForSequenceClassification.from_pretrained(model_name)
    >>> tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    >>> ## TENSORFLOW CODE
    >>> from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
    >>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    >>> model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
    >>> tokenizer = DistilBertTokenizer.from_pretrained(model_name)

Customizing the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to change how the model itself is built, you can define a custom configuration class. Each architecture
comes with its own relevant configuration. For example, :class:`~transformers.DistilBertConfig` allows you to specify
parameters such as the hidden dimension, dropout rate, etc for DistilBERT. If you do core modifications, like changing
the hidden size, you won't be able to use a pretrained model anymore and will need to train from scratch. You would
then instantiate the model directly from this configuration.

Below, we load a predefined vocabulary for a tokenizer with the
:func:`~transformers.DistilBertTokenizer.from_pretrained` method. However, unlike the tokenizer, we wish to initialize
the model from scratch. Therefore, we instantiate the model from a configuration instead of using the
:func:`~transformers.DistilBertForSequenceClassification.from_pretrained` method.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
    >>> config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForSequenceClassification(config)
    >>> ## TENSORFLOW CODE
    >>> from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
    >>> config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = TFDistilBertForSequenceClassification(config)

For something that only changes the head of the model (for instance, the number of labels), you can still use a
pretrained model for the body. For instance, let's define a classifier for 10 different labels using a pretrained body.
Instead of creating a new configuration with all the default values just to change the number of labels, we can instead
pass any argument a configuration would take to the :func:`from_pretrained` method and it will update the default
configuration appropriately:

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
    >>> model_name = "distilbert-base-uncased"
    >>> model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
    >>> tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    >>> ## TENSORFLOW CODE
    >>> from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
    >>> model_name = "distilbert-base-uncased"
    >>> model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
    >>> tokenizer = DistilBertTokenizer.from_pretrained(model_name)
