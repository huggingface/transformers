Model sharing and uploading
===========================

In this page, we will show you how to share a model you have trained or fine-tuned on new data with the community on
the `model hub <https://huggingface.co/models>`__.

.. note::

    You will need to create an account on `huggingface.co <https://huggingface.co/join>`__ for this.

    Optionally, you can join an existing organization or create a new one.

Prepare your model for uploading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen in the :doc:`training tutorial <training>`: how to fine-tune a model on a given task. You have probably
done something similar on your task, either using the model directly in your own training loop or using the
:class:`~.transformers.Trainer`/:class:`~.transformers.TFTrainer` class. Let's see how you can share the result on
the `model hub <https://huggingface.co/models>`__.

Basic steps
^^^^^^^^^^^

.. 
    When #5258 is merged, we can remove the need to create the directory.

First, pick a directory with the name you want your model to have on the model hub (its full name will then be
`username/awesome-name-you-picked` or `organization/awesome-name-you-picked`) and create it with either

::

    mkdir path/to/awesome-name-you-picked

or in python

::

    import os
    os.makedirs("path/to/awesome-name-you-picked")

then you can save your model and tokenizer with:

::

    model.save_pretrained("path/to/awesome-name-you-picked")
    tokenizer.save_pretrained("path/to/awesome-name-you-picked")

Or, if you're using the Trainer API

::

    trainer.save_model("path/to/awesome-name-you-picked")
    tokenizer.save_pretrained("path/to/awesome-name-you-picked")

Make your model work on all frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. 
    TODO Sylvain: make this automatic during the upload

You probably have your favorite framework, but so will other users! That's why it's best to upload your model with both
PyTorch `and` TensorFlow checkpoints to make it easier to use (if you skip this step, users will still be able to load
your model in another framework, but it will be slower, as it will have to be converted on the fly). Don't worry, it's super easy to do (and in a future version,
it will all be automatic). You will need to install both PyTorch and TensorFlow for this step, but you don't need to
worry about the GPU, so it should be very easy. Check the
`TensorFlow installation page <https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available>`__ 
and/or the `PyTorch installation page <https://pytorch.org/get-started/locally/#start-locally>`__ to see how.

First check that your model class exists in the other framework, that is try to import the same model by either adding
or removing TF. For instance, if you trained a :class:`~transformers.DistilBertForSequenceClassification`, try to
type

::

    from transformers import TFDistilBertForSequenceClassification

and if you trained a :class:`~transformers.TFDistilBertForSequenceClassification`, try to
type

::

    from transformers import DistilBertForSequenceClassification

This will give back an error if your model does not exist in the other framework (something that should be pretty rare
since we're aiming for full parity between the two frameworks). In this case, skip this and go to the next step.

Now, if you trained your model in PyTorch and have to create a TensorFlow version, adapt the following code to your
model class:

::

    tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
    tf_model.save_pretrained("path/to/awesome-name-you-picked")

and if you trained your model in TensorFlow and have to create a PyTorch version, adapt the following code to your
model class:

::

    pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
    pt_model.save_pretrained("path/to/awesome-name-you-picked")

That's all there is to it!

Check the directory before uploading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure there are no garbage files in the directory you'll upload. It should only have:

- a `config.json` file, which saves the :doc:`configuration <main_classes/configuration>` of your model ;
- a `pytorch_model.bin` file, which is the PyTorch checkpoint (unless you can't have it for some reason) ;
- a `tf_model.h5` file, which is the TensorFlow checkpoint (unless you can't have it for some reason) ;
- a `special_tokens_map.json`, which is part of your :doc:`tokenizer <main_classes/tokenizer>` save;
- a `tokenizer_config.json`, which is part of your :doc:`tokenizer <main_classes/tokenizer>` save;
- a `vocab.txt`, which is the vocabulary of your tokenizer, part of your :doc:`tokenizer <main_classes/tokenizer>`
  save;
- maybe a `added_tokens.json`, which is part of your :doc:`tokenizer <main_classes/tokenizer>` save.

Other files can safely be deleted.

Upload your model with the CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now go in a terminal and run the following command. It should be in the virtual enviromnent where you installed 🤗
Transformers, since that command :obj:`transformers-cli` comes from the library.

::

    transformers-cli login

Then log in using the same credentials as on huggingface.co. To upload your model, just type

::

    transformers-cli upload path/to/awesome-name-you-picked/

This will upload the folder containing the weights, tokenizer and configuration we prepared in the previous section.

If you want to upload a single file (a new version of your model, or the other framework checkpoint you want to add),
just type:

::

    transformers-cli upload path/to/awesome-name-you-picked/that-file 

or

::

   transformers-cli upload path/to/awesome-name-you-picked/that-file --filename awesome-name-you-picked/new_name

if you want to change its filename.

This uploads the model to your personal account. If you want your model to be namespaced by your organization name
rather than your username, add the following flag to any command:

::

    --organization organization_name

so for instance:

::

    transformers-cli upload path/to/awesome-name-you-picked/ --organization organization_name

Your model will then be accessible through its identifier, which is, as we saw above,
`username/awesome-name-you-picked` or `organization/awesome-name-you-picked`.

Add a model card
^^^^^^^^^^^^^^^^

To make sure everyone knows what your model can do, what its limitations and potential bias or ethetical
considerations, please add a README.md model card to the 🤗 Transformers repo under `model_cards/`. It should then be
placed in a subfolder with your username or organization, then another subfolder named like your model
(`awesome-name-you-picked`). Or just click on the "Create a model card on GitHub" button on the model page, it will
get you directly to the right location. If you need one, `here <https://github.com/huggingface/model_card>`__ is a
model card template (meta-suggestions are welcome).

If your model is fine-tuned from another model coming from the model hub (all 🤗 Transformers pretrained models do),
don't forget to link to its model card so that people can fully trace how your model was built.

If you have never made a pull request to the 🤗 Transformers repo, look at the
:doc:`contributing guide <contributing>` to see the steps to follow.

.. Note::

    You can also send your model card in the folder you uploaded with the CLI by placing it in a `README.md` file
    inside `path/to/awesome-name-you-picked/`.

Using your model
^^^^^^^^^^^^^^^^

Your model now has a page on huggingface.co/models 🔥

Anyone can load it from code:

::

    tokenizer = AutoTokenizer.from_pretrained("namespace/awesome-name-you-picked")
    model = AutoModel.from_pretrained("namespace/awesome-name-you-picked")

Additional commands
^^^^^^^^^^^^^^^^^^^

You can list all the files you uploaded on the hub like this:

::

    transformers-cli s3 ls

You can also delete unneeded files with

::

    transformers-cli s3 rm awesome-name-you-picked/filename

