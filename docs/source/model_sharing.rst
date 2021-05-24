.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Model sharing and uploading
=======================================================================================================================

In this page, we will show you how to share a model you have trained or fine-tuned on new data with the community on
the `model hub <https://huggingface.co/models>`__.

.. note::

    You will need to create an account on `huggingface.co <https://huggingface.co/join>`__ for this.

    Optionally, you can join an existing organization or create a new one.


We have seen in the :doc:`training tutorial <training>`: how to fine-tune a model on a given task. You have probably
done something similar on your task, either using the model directly in your own training loop or using the
:class:`~.transformers.Trainer`/:class:`~.transformers.TFTrainer` class. Let's see how you can share the result on the
`model hub <https://huggingface.co/models>`__.

Model versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    >>> model = AutoModel.from_pretrained(
    >>>   "julien-c/EsperBERTo-small",
    >>>   revision="v2.0.1" # tag name, or branch name, or commit hash
    >>> )


Push your model from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to make sure your credentials to the hub are stored somewhere. This can be done in two ways. If you
have access to a terminal, you cam just run the following command in the virtual environment where you installed 🤗
Transformers:

.. code-block:: bash

    transformers-cli login

It will store your access token in the Hugging Face cache folder (by default :obj:`~/.cache/`).

If you don't have an easy access to a terminal (for instance in a Colab session), you can find a token linked to your
acount by going on `huggingface.co <https://huggingface.co/>`, click on your avatar on the top left corner, then on
`Edit profile` on the left, just beneath your profile picture. In the submenu `API Tokens`, you will find your API
token that you can just copy.

Directly push your model to the hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have an API token (either stored in the cache or copied and pasted in your notebook), you can directly push a
finetuned model you saved in :obj:`save_drectory` by calling:

.. code-block:: python

    finetuned_model.push_to_hub("my-awesome-model")

If you have your API token not stored in the cache, you will need to pass it with :obj:`use_auth_token=your_token`.
This is also be the case for all the examples below, so we won't mention it again.

This will create a repository in your namespace name :obj:`my-awesome-model`, so anyone can now run:

.. code-block:: python

    from transformers import AutoModel

    model = AutoModel.from_pretrained("your_username/my-awesome-model")

Even better, you can combine this push to the hub with the call to :obj:`save_pretrained`:

.. code-block:: python

    finetuned_model.save_pretrained(save_directory, push_to_hub=True, repo_name="my-awesome-model")

If you are a premium user and want your model to be private, just add :obj:`private=True` to this call.

If you are a member of an organization and want to push it inside the namespace of the organization instead of yours,
just add :obj:`organization=my_amazing_org`.

Add new files to your model repo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have pushed your model to the hub, you might want to add the tokenizer, or a version of your model for another
framework (TensorFlow, PyTorch, Flax). This is super easy to do! Let's begin with the tokenizer. You can add it to the
repo you created before like this

.. code-block:: python

    tokenizer.push_to_hub("my-awesome-model")

If you know its URL (it should be :obj:`https://huggingface.co/username/repo_name`), you can also do:

.. code-block:: python

    tokenizer.push_to_hub(repo_url=my_repo_url)

And that's all there is to it! It's also a very easy way to fix a mistake if one of the files online had a bug.

To add a model for another backend, it's also super easy. Let's say you have fine-tuned a TensorFlow model and want to
add the pytorch model files to your model repo, so that anyone in the community can use it. The following allows you to
directly create a PyTorch version of your TensorFlow model:

.. code-block:: python

    from transformers import AutoModel

    model = AutoModel.from_pretrained(save_directory, from_tf=True)

You can also replace :obj:`save_directory` by the identifier of your model (:obj:`username/repo_name`) if you don't
have a local save of it anymore. Then, just do the same as before:

.. code-block:: python

    model.push_to_hub("my-awesome-model")

or

.. code-block:: python

    model.push_to_hub(repo_url=my_repo_url)


Use your terminal and git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to upload a model, you'll need to first create a git repo. This repo will live on the model hub, allowing
users to clone it and you (and your organization members) to push to it.

You can create a model repo directly from `the /new page on the website <https://huggingface.co/new>`__.

Alternatively, you can use the ``transformers-cli``. The next steps describe that process:

Go to a terminal and run the following command. It should be in the virtual environment where you installed 🤗
Transformers, since that command :obj:`transformers-cli` comes from the library.

.. code-block:: bash

    transformers-cli login


Once you are logged in with your model hub credentials, you can start building your repositories. To create a repo:

.. code-block:: bash

    transformers-cli repo create your-model-name

If you want to create a repo under a specific organization, you should add a `--organization` flag:

.. code-block:: bash

    transformers-cli repo create your-model-name --organization your-org-name

This creates a repo on the model hub, which can be cloned.

.. code-block:: bash

    # Make sure you have git-lfs installed
    # (https://git-lfs.github.com/)
    git lfs install

    git clone https://huggingface.co/username/your-model-name

When you have your local clone of your repo and lfs installed, you can then add/remove from that clone as you would
with any other git repo.

.. code-block:: bash

    # Commit as usual
    cd your-model-name
    echo "hello" >> README.md
    git add . && git commit -m "Update from $USER"

We are intentionally not wrapping git too much, so that you can go on with the workflow you're used to and the tools
you already know.

The only learning curve you might have compared to regular git is the one for git-lfs. The documentation at
`git-lfs.github.com <https://git-lfs.github.com/>`__ is decent, but we'll work on a tutorial with some tips and tricks
in the coming weeks!

Additionally, if you want to change multiple repos at once, the `change_config.py script
<https://github.com/huggingface/efficient_scripts/blob/main/change_config.py>`__ can probably save you some time.

Make your model work on all frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. 
    TODO Sylvain: make this automatic during the upload

You probably have your favorite framework, but so will other users! That's why it's best to upload your model with both
PyTorch `and` TensorFlow checkpoints to make it easier to use (if you skip this step, users will still be able to load
your model in another framework, but it will be slower, as it will have to be converted on the fly). Don't worry, it's
super easy to do (and in a future version, it might all be automatic). You will need to install both PyTorch and
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
    >>> tokenizer.save_pretrained("path/to/repo/clone/your-model-name")

You can then add these files to the staging environment and verify that they have been correctly staged with the ``git
status`` command:

.. code-block:: bash

    git add --all
    git status

Finally, the files should be committed:

.. code-block:: bash

    git commit -m "First version of the your-model-name model and tokenizer."

And pushed to the remote:

.. code-block:: bash

    git push

This will upload the folder containing the weights, tokenizer and configuration we have just prepared.


Add a model card
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make sure everyone knows what your model can do, what its limitations, potential bias or ethical considerations are,
please add a README.md model card to your model repo. You can just create it, or there's also a convenient button
titled "Add a README.md" on your model page. A model card template can be found `here
<https://github.com/huggingface/model_card>`__ (meta-suggestions are welcome). model card template (meta-suggestions
are welcome).

.. note::

    Model cards used to live in the 🤗 Transformers repo under `model_cards/`, but for consistency and scalability we
    migrated every model card from the repo to its corresponding huggingface.co model repo.

If your model is fine-tuned from another model coming from the model hub (all 🤗 Transformers pretrained models do),
don't forget to link to its model card so that people can fully trace how your model was built.


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

Workflow in a Colab notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're in a Colab notebook (or similar) with no direct access to a terminal, here is the workflow you can use to
upload your model. You can execute each one of them in a cell by adding a ! at the beginning.

First you need to install `git-lfs` in the environment used by the notebook:

.. code-block:: bash

    sudo apt-get install git-lfs

Then you can use either create a repo directly from `huggingface.co <https://huggingface.co/>`__ , or use the
:obj:`transformers-cli` to create it:


.. code-block:: bash

    transformers-cli login
    transformers-cli repo create your-model-name

Once it's created, you can clone it and configure it (replace username by your username on huggingface.co):

.. code-block:: bash

    git lfs install

    git clone https://username:password@huggingface.co/username/your-model-name
    # Alternatively if you have a token,
    # you can use it instead of your password
    git clone https://username:token@huggingface.co/username/your-model-name

    cd your-model-name
    git config --global user.email "email@example.com"
    # Tip: using the same email than for your huggingface.co account will link your commits to your profile
    git config --global user.name "Your name"

Once you've saved your model inside, and your clone is setup with the right remote URL, you can add it and push it with
usual git commands.

.. code-block:: bash

    git add .
    git commit -m "Initial commit"
    git push
