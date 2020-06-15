Training and fine-tuning
========================

Model classes in Transformers are designed to be compatible with native
PyTorch and TensorFlow 2 and can be used seemlessly with either. In this
quickstart, we will show how to fine-tune (or train from scratch) a model
using the standard training tools available in either framework. We will also
show how to use our included :func:`~transformers.Trainer` class which
handles much of the complexity of training for you.

This guide assume that you are already familiar with loading and use our
models for inference; otherwise, see `Usage <./usage.html>`_. We also assume
that you are familiar with training deep neural networks in either PyTorch or
TF2, and focus specifically on the nuances and tools for training models in
Transformers.

Sections:

  * :ref:`pytorch`
  * :ref:`tensorflow`
  * :ref:`trainer`
  * :ref:`additional-resources`

.. _pytorch:

Fine-tuning in native PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model classes in Transformers that don't begin with ``TF`` are
`PyTorch Modules <https://pytorch.org/docs/master/generated/torch.nn.Module.html>`_,
meaning that you can use them just as you would any model in PyTorch for
both inference and optimization.

Let's consider the common task of fine-tuning a masked language model like
BERT on a sequence classification task. When we instantiate a model with
:func:`~transformers.PreTrainedModel.from_pretrained`, the model
configuration and pre-trained weights
of the specified model are used to initialize the model. The
library also includes a number of task-specific final layers or 'heads' whose
weights are instantiated randomly when not present in the specified
pre-trained model. For example, instantiating a model with
``BertForSequenceClassification.from_pretrained('bert-base-uncased', num_classes=2)``
will create a BERT model instance with encoder weights copied from the
``bert-base-uncased`` model and a randomly initialized sequence
classification head on top of the encoder with an output size of 2.

This is useful because it allows us to make use of the pre-trained BERT
encoder and easily train it on whatever sequence classification dataset we
choose. Models are initialized in ``eval`` mode by default. To train, be sure
to first call ``model.train()``. When we call the model with the target labels
as a tensor, the first returned element is the Cross Entropy loss between the
classifier predictions and the passed labels. The following demonstrates
how you could instantiate a pre-trained model and perform a single update on
a dummy sentiment classification batch:

.. code-block:: python

    from transformers import BertTokenizer, BertForSequenceClassification, AdamW

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5) # any PT optimizer works

    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    labels = torch.tensor([1,0]).unsqueeze(0)
    encoding = tokenizer.batch_encode_plus(text_batch, return_tensors='pt',
                                           pad_to_max_length=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    loss.backward()
    optimizer.step()

Of course, you can train on GPU by calling ``to('cuda')`` on the model and
inputs as usual.

We also provide a few tools to make it easier to apply more complex
fine-tuning procedures. Here's an example using weight decay and learning
rate scheduling (omitting the data preparation steps for brevity):

.. code-block:: python

    from transformers import get_linear_schedule_with_warmup

    device = torch.device('cuda')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    model.train()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    # iterate over batches of preprocessed training data
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()

We highly recommend using :func:`~transformers.Trainer`, discussed below, which conveniently
handles the moving parts of training Transformers models with features like
mixed precision and easy tensorboard logging.


Freezing the encoder
--------------------

In some cases, you might be interested in keeping the weights of the
pre-trained encoder frozen and optimizing only the weights of the head
layers. To do so, simply set the ``requires_grad`` attribute to ``False`` on
the encoder parameters, which can be accessed with the ``base_model``
submodule:

.. code-block:: python
   
    for param in model.base_model.parameters():
        param.requires_grad = False


.. _tensorflow:

Fine-tuning in native TensorFlow 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow models can also be trained natively in TF2. Just as with PyTorch,
TensorFlow models can be instantiated with
:func:`~transformers.PreTrainedModel.from_pretrained` to load the weights of
the encoder from a pretrained model. The model can then be trained using the
Keras ``fit`` method. Here's an example of fine-tuning a sequence classifier
with a pre-trained encoder on MRPC:

.. code-block:: python
    
    import tensorflow as tf
    import tensorflow_datasets
    from transformers import TFBertForSequenceClassification, BertTokenizer, glue_convert_examples_to_features

    # Load dataset, tokenizer, model from pretrained model/vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
    data = tensorflow_datasets.load('glue/mrpc')

    # Prepare dataset for GLUE as a tf.data.Dataset instance
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
    train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
    valid_dataset = valid_dataset.batch(64)

    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)

    # Train and evaluate using tf.keras.Model.fit()
    history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                        validation_data=valid_dataset, validation_steps=7)

This example uses the built-in
:func:`~transformers.data.processors.glue.glue_convert_examples_to_features`
to tokenize MRPC and convert it to a TF ``Dataset`` object which can be
passed to ``model.fit()``.

With the tight interoperability between TensorFlow and PyTorch models, you
can even save the model and then reload it as a PyTorch model:

.. code-block:: python

    model.save_pretrained('./save/')
    pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)


.. _trainer:

Trainer
^^^^^^^

We also provide a simple but feature-complete training and evaluation
interface through :func:`~transformers.Trainer` and
:func:`~transformers.TFTrainer`. You can train, fine-tune,
and evaluate any Transformers model with a wide range of training options and
with built-in features like logging, gradient accumulation, and mixed
precision.

Using the trainer requires first creating a model and defining a
:func:`~transformers.data.DataCollator`, which is responsible for taking in a batch and preparing
them to be fed into the model.  The Trainer will expect the collated batches
to be a dictionary with keys corresponding to the kwargs that will be used to
call our model. Below, we define a collator in the case where a batch
consists of a list of dicts and returns a single dict with the examples
concatenated.

.. code-block:: python

    from transformers import DataCollator

    class MyCollator(DataCollator):
        def collate_batch(self, batch):
            input_ids = torch.stack([example['input_ids'] for example in batch])
            attention_mask = torch.stack([example['attention_mask'] for example in batch])
            labels = torch.stack([example['label'] for example in batch])

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

Now, you can simply define a :func:`~transformers.Trainer` and
:func:`~transformers.TrainingArguments` and the all
of the underlying work is done for you.

.. code-block:: python

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                     # the instantiated Transformers model to be trained
        args=training_args,              # training arguments, defined above
        data_collator=MyCollator(),      # instance of your defined data collator (see above)
        train_dataset=train_dataset,     # training dataset
        eval_dataset=test_dataset        # evaluation dataset
    )

Now simply call ``trainer.train()`` to train and ``trainer.evaluate()`` to
evaluate. You can view the results by launching tensorboard in your specified
``logging_dir`` directory.

You can use your own module as well, but the first argument returned from
``forward`` must be the loss which you wish to optimize.

.. _additional-resources:

Additional resources
^^^^^^^^^^^^^^^^^^^^

    * `A lightweight colab demo
      <https://colab.research.google.com/drive/1-JIJlao4dI-Ilww_NnTc0rxtp-ymgDgM?usp=sharing>`_
      which uses ``Trainer`` for IMDb sentiment classification.

    * `Transformers Examples <https://github.com/huggingface/transformers/tree/master/examples>`_
      including scripts for training and fine-tuning on GLUE, SQuAD, and
      several other tasks.

    * `How to train a language model
      <https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb>`_,
      a detailed colab notebook which uses ``Trainer`` to train a masked
      language model from scratch on Esperanto.

    * `Transformers Notebooks <./notebooks.html>`_ which contain dozens
      of example notebooks from the community for training and using
      Transformers on a variety of tasks.