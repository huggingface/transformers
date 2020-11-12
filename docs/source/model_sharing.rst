Model sharing and uploading
=======================================================================================================================

In this page, we will show you how to share a model you have trained or fine-tuned on new data with the community on
the `model hub <https://huggingface.co/models>`__.

.. note::

    You will need to create an account on `huggingface.co <https://huggingface.co/join>`__ for this.

    Optionally, you can join an existing organization or create a new one.

Prepare your model for uploading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen in the :doc:`training tutorial <training>`: how to fine-tune a model on a given task. You have probably
done something similar on your task, either using the model directly in your own training loop or using the
:class:`~.transformers.Trainer`/:class:`~.transformers.TFTrainer` class. Let's see how you can share the result on the
`model hub <https://huggingface.co/models>`__.

Model versioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since version v3.5.0, the model hub has built-in model versioning based on git and git-lfs. It is based on the paradigm
that one model *is* one repo.

This allows:

- built-in versioning
- access control
- scalability

This is built around *revisions*, which is a way to pin a specific version of a model, using a commit hash, tag or
branch.

For instance:

.. code-block::

    >>> tokenizer = AutoTokenizer.from_pretrained(
    >>>   "julien-c/EsperBERTo-small",
    >>>   revision="v2.0.1" # tag name, or branch name, or commit hash
    >>> )

Basic steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to upload a model, you'll need to first create a git repo. This repo will live on the model hub, allowing
users to clone it and you (and your organization members) to push to it. First, you should ensure you are logged in the
``transformers-cli``:

Go in a terminal and run the following command. It should be in the virtual environment where you installed 🤗
Transformers, since that command :obj:`transformers-cli` comes from the library.

.. code-block::

    transformers-cli login


Once you are logged in with your model hub credentials, you can start building your repositories. To create a repo:

.. code-block::

    transformers-cli repo create your-model-name

This creates a repo on the model hub, which can be cloned. You can then add/remove from that repo as you would with any
other git repo.

.. code-block::

    git clone https://huggingface.co/username/your-model-name

    # Then commit as usual
    cd your-model-name
    echo "hello" >> README.md
    git add . && git commit -m "Update from $USER"

We are intentionally not wrapping git too much, so as to stay intuitive and easy-to-use.


Make your model work on all frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. 
    TODO Sylvain: make this automatic during the upload

You probably have your favorite framework, but so will other users! That's why it's best to upload your model with both
PyTorch `and` TensorFlow checkpoints to make it easier to use (if you skip this step, users will still be able to load
your model in another framework, but it will be slower, as it will have to be converted on the fly). Don't worry, it's
super easy to do (and in a future version, it will all be automatic). You will need to install both PyTorch and
TensorFlow for this step, but you don't need to worry about the GPU, so it should be very easy. Check the `TensorFlow
installation page <https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available>`__ and/or the `PyTorch
installation page <https://pytorch.org/get-started/locally/#start-locally>`__ to see how.

First check that your model class exists in the other framework, that is try to import the same model by either adding
or removing TF. For instance, if you trained a :class:`~transformers.DistilBertForSequenceClassification`, try to type

.. code-block::

    >>> from transformers import TFDistilBertForSequenceClassification

and if you trained a :class:`~transformers.TFDistilBertForSequenceClassification`, try to type

.. code-block::

    >>> from transformers import DistilBertForSequenceClassification

This will give back an error if your model does not exist in the other framework (something that should be pretty rare
since we're aiming for full parity between the two frameworks). In this case, skip this and go to the next step.

Now, if you trained your model in PyTorch and have to create a TensorFlow version, adapt the following code to your
model class:

.. code-block::

    >>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
    >>> tf_model.save_pretrained("path/to/awesome-name-you-picked")

and if you trained your model in TensorFlow and have to create a PyTorch version, adapt the following code to your
model class:

.. code-block::

    >>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
    >>> pt_model.save_pretrained("path/to/awesome-name-you-picked")

That's all there is to it!

Check the directory before pushing to the model hub.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure there are no garbage files in the directory you'll upload. It should only have:

- a `config.json` file, which saves the :doc:`configuration <main_classes/configuration>` of your model ;
- a `pytorch_model.bin` file, which is the PyTorch checkpoint (unless you can't have it for some reason) ;
- a `tf_model.h5` file, which is the TensorFlow checkpoint (unless you can't have it for some reason) ;
- a `special_tokens_map.json`, which is part of your :doc:`tokenizer <main_classes/tokenizer>` save;
- a `tokenizer_config.json`, which is part of your :doc:`tokenizer <main_classes/tokenizer>` save;
- files named `vocab.json`, `vocab.txt`, `merges.txt`, or similar, which contain the vocabulary of your tokenizer, part
  of your :doc:`tokenizer <main_classes/tokenizer>` save;
- maybe a `added_tokens.json`, which is part of your :doc:`tokenizer <main_classes/tokenizer>` save.

Other files can safely be deleted.


Uploading your files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the repo is cloned, you can add the model, configuration and tokenizer files. For instance, saving the model and
tokenizer files:

.. code-block::

    >>> model.save_pretrained("path/to/repo/clone/your-model-name")
    >>> tokenizer.save_pretrained("path/to/repo/clone/your-model-name")

Or, if you're using the Trainer API

.. code-block::

    >>> trainer.save_model("path/to/awesome-name-you-picked")

You can then add these files to the staging environment and verify that they have been correctly staged with the ``git
status`` command:

.. code-block::

    git add --all
    git status

Finally, the files should be comitted:

.. code-block::

    git commit -m "First version of the your-model-name model and tokenizer."

And pushed to the remote:

.. code-block::

    git push

This will upload the folder containing the weights, tokenizer and configuration we have just prepared.


Add a model card
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make sure everyone knows what your model can do, what its limitations and potential bias or ethetical
considerations, please add a README.md model card to the 🤗 Transformers repo under `model_cards/`. It should then be
placed in a subfolder with your username or organization, then another subfolder named like your model
(`awesome-name-you-picked`). Or just click on the "Create a model card on GitHub" button on the model page, it will get
you directly to the right location. If you need one, `here <https://github.com/huggingface/model_card>`__ is a model
card template (meta-suggestions are welcome).

If your model is fine-tuned from another model coming from the model hub (all 🤗 Transformers pretrained models do),
don't forget to link to its model card so that people can fully trace how your model was built.

If you have never made a pull request to the 🤗 Transformers repo, look at the :doc:`contributing guide <contributing>`
to see the steps to follow.

.. Note::

    You can also send your model card in the folder you uploaded with the CLI by placing it in a `README.md` file
    inside `path/to/awesome-name-you-picked/`.

Using your model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your model now has a page on huggingface.co/models 🔥

Anyone can load it from code:

.. code-block::

    >>> tokenizer = AutoTokenizer.from_pretrained("namespace/awesome-name-you-picked")
    >>> model = AutoModel.from_pretrained("namespace/awesome-name-you-picked")


You may specify a revision by using the ``revision`` flag in the ``from_pretrained`` method:

.. code-block::

    >>> tokenizer = AutoTokenizer.from_pretrained(
    >>>   "julien-c/EsperBERTo-small",
    >>>   revision="v2.0.1" # tag name, or branch name, or commit hash
    >>> )
