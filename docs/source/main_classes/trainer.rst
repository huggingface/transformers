..
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Trainer
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.Trainer` and :class:`~transformers.TFTrainer` classes provide an API for feature-complete
training in most standard use cases. It's used in most of the :doc:`example scripts <../examples>`.

Before instantiating your :class:`~transformers.Trainer`/:class:`~transformers.TFTrainer`, create a
:class:`~transformers.TrainingArguments`/:class:`~transformers.TFTrainingArguments` to access all the points of
customization during training.

The API supports distributed training on multiple GPUs/TPUs, mixed precision through `NVIDIA Apex
<https://github.com/NVIDIA/apex>`__ and Native AMP for PyTorch and :obj:`tf.keras.mixed_precision` for TensorFlow.

Both :class:`~transformers.Trainer` and :class:`~transformers.TFTrainer` contain the basic training loop which supports
the above features. To inject custom behavior you can subclass them and override the following methods:

- **get_train_dataloader**/**get_train_tfdataset** -- Creates the training DataLoader (PyTorch) or TF Dataset.
- **get_eval_dataloader**/**get_eval_tfdataset** -- Creates the evaluation DataLoader (PyTorch) or TF Dataset.
- **get_test_dataloader**/**get_test_tfdataset** -- Creates the test DataLoader (PyTorch) or TF Dataset.
- **log** -- Logs information on the various objects watching training.
- **create_optimizer_and_scheduler** -- Sets up the optimizer and learning rate scheduler if they were not passed at
  init. Note, that you can also subclass or override the ``create_optimizer`` and ``create_scheduler`` methods
  separately.
- **create_optimizer** -- Sets up the optimizer if it wasn't passed at init.
- **create_scheduler** -- Sets up the learning rate scheduler if it wasn't passed at init.
- **compute_loss** - Computes the loss on a batch of training inputs.
- **training_step** -- Performs a training step.
- **prediction_step** -- Performs an evaluation/test step.
- **run_model** (TensorFlow only) -- Basic pass through the model.
- **evaluate** -- Runs an evaluation loop and returns metrics.
- **predict** -- Returns predictions (with metrics if labels are available) on a test set.

.. warning::

    The :class:`~transformers.Trainer` class is optimized for 🤗 Transformers models and can have surprising behaviors
    when you use it on other models. When using it on your own model, make sure:

    - your model always return tuples or subclasses of :class:`~transformers.file_utils.ModelOutput`.
    - your model can compute the loss if a :obj:`labels` argument is provided and that loss is returned as the first
      element of the tuple (if your model returns tuples)
    - your model can accept multiple label arguments (use the :obj:`label_names` in your
      :class:`~transformers.TrainingArguments` to indicate their name to the :class:`~transformers.Trainer`) but none
      of them should be named :obj:`"label"`.

Here is an example of how to customize :class:`~transformers.Trainer` using a custom loss function for multi-label
classification:

.. code-block:: python

    from torch import nn
    from transformers import Trainer

    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.float().view(-1, self.model.config.num_labels))
            return (loss, outputs) if return_outputs else loss

Another way to customize the training loop behavior for the PyTorch :class:`~transformers.Trainer` is to use
:doc:`callbacks <callback>` that can inspect the training loop state (for progress reporting, logging on TensorBoard or
other ML platforms...) and take decisions (like early stopping).


Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Trainer
    :members:


Seq2SeqTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Seq2SeqTrainer
    :members: evaluate, predict


TFTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTrainer
    :members:


TrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TrainingArguments
    :members:


Seq2SeqTrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Seq2SeqTrainingArguments
    :members:


TFTrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTrainingArguments
    :members:


Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default :class:`~transformers.Trainer` will use ``logging.INFO`` for the main process and ``logging.WARNING`` for
the replicas if any.

These defaults can be overridden to use any of the 5 ``logging`` levels with :class:`~transformers.TrainingArguments`'s
arguments:

- ``log_level`` - for the main process
- ``log_level_replica`` - for the replicas

Further, if :class:`~transformers.TrainingArguments`'s ``log_on_each_node`` is set to ``False`` only the main node will
use the log level settings for its main process, all other nodes will use the log level settings for replicas.

Note that :class:`~transformers.Trainer` is going to set ``transformers``'s log level separately for each node in its
:meth:`~transformers.Trainer.__init__`. So you may want to set this sooner (see the next example) if you tap into other
``transformers`` functionality before creating the :class:`~transformers.Trainer` object.

Here is an example of how this can be used in an application:

.. code-block:: python

    [...]
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    trainer = Trainer(...)

And then if you only want to see warnings on the main node and all other nodes to not print any most likely duplicated
warnings you could run it as:

.. code-block:: bash

    my_app.py ... --log_level warning --log_level_replica error

In the multi-node environment if you also don't want the logs to repeat for each node's main process, you will want to
change the above to:

.. code-block:: bash

    my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

and then only the main process of the first node will log at the "warning" level, and all other processes on the main
node and all processes on other nodes will log at the "error" level.

If you need your application to be as quiet as possible you could do:

.. code-block:: bash

    my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0

(add ``--log_on_each_node 0`` if on multi-node environment)



Randomness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When resuming from a checkpoint generated by :class:`~transformers.Trainer` all efforts are made to restore the
`python`, `numpy` and `pytorch` RNG states to the same states as they were at the moment of saving that checkpoint,
which should make the "stop and resume" style of training as close as possible to non-stop training.

However, due to various default non-deterministic pytorch settings this might not fully work. If you want full
determinism please refer to `Controlling sources of randomness
<https://pytorch.org/docs/stable/notes/randomness.html>`__. As explained in the document, that some of those settings
that make things determinstic (.e.g., ``torch.backends.cudnn.deterministic``) may slow things down, therefore this
can't be done by default, but you can enable those yourself if needed.


Trainer Integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The :class:`~transformers.Trainer` has been extended to support libraries that may dramatically improve your training
time and fit much bigger models.

Currently it supports third party solutions, `DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ and `FairScale
<https://github.com/facebookresearch/fairscale/>`__, which implement parts of the paper `ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models, by Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
<https://arxiv.org/abs/1910.02054>`__.

This provided support is new and experimental as of this writing.

.. _zero-install-notes:

CUDA Extension Installation Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of this writing, both FairScale and Deepspeed require compilation of CUDA C++ code, before they can be used.

While all installation issues should be dealt with through the corresponding GitHub Issues of `FairScale
<https://github.com/facebookresearch/fairscale/issues>`__ and `Deepspeed
<https://github.com/microsoft/DeepSpeed/issues>`__, there are a few common issues that one may encounter while building
any PyTorch extension that needs to build CUDA extensions.

Therefore, if you encounter a CUDA-related build issue while doing one of the following or both:

.. code-block:: bash

    pip install fairscale
    pip install deepspeed

please, read the following notes first.

In these notes we give examples for what to do when ``pytorch`` has been built with CUDA ``10.2``. If your situation is
different remember to adjust the version number to the one you are after.

Possible problem #1
=======================================================================================================================

While, Pytorch comes with its own CUDA toolkit, to build these two projects you must have an identical version of CUDA
installed system-wide.

For example, if you installed ``pytorch`` with ``cudatoolkit==10.2`` in the Python environment, you also need to have
CUDA ``10.2`` installed system-wide.

The exact location may vary from system to system, but ``/usr/local/cuda-10.2`` is the most common location on many
Unix systems. When CUDA is correctly set up and added to the ``PATH`` environment variable, one can find the
installation location by doing:

.. code-block:: bash

    which nvcc

If you don't have CUDA installed system-wide, install it first. You will find the instructions by using your favorite
search engine. For example, if you're on Ubuntu you may want to search for: `ubuntu cuda 10.2 install
<https://www.google.com/search?q=ubuntu+cuda+10.2+install>`__.

Possible problem #2
=======================================================================================================================

Another possible common problem is that you may have more than one CUDA toolkit installed system-wide. For example you
may have:

.. code-block:: bash

    /usr/local/cuda-10.2
    /usr/local/cuda-11.0

Now, in this situation you need to make sure that your ``PATH`` and ``LD_LIBRARY_PATH`` environment variables contain
the correct paths to the desired CUDA version. Typically, package installers will set these to contain whatever the
last version was installed. If you encounter the problem, where the package build fails because it can't find the right
CUDA version despite you having it installed system-wide, it means that you need to adjust the 2 aforementioned
environment variables.

First, you may look at their contents:

.. code-block:: bash

    echo $PATH
    echo $LD_LIBRARY_PATH

so you get an idea of what is inside.

It's possible that ``LD_LIBRARY_PATH`` is empty.

``PATH`` lists the locations of where executables can be found and ``LD_LIBRARY_PATH`` is for where shared libraries
are to looked for. In both cases, earlier entries have priority over the later ones. ``:`` is used to separate multiple
entries.

Now, to tell the build program where to find the specific CUDA toolkit, insert the desired paths to be listed first by
doing:

.. code-block:: bash

    export PATH=/usr/local/cuda-10.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

Note that we aren't overwriting the existing values, but prepending instead.

Of course, adjust the version number, the full path if need be. Check that the directories you assign actually do
exist. ``lib64`` sub-directory is where the various CUDA ``.so`` objects, like ``libcudart.so`` reside, it's unlikely
that your system will have it named differently, but if it is adjust it to reflect your reality.


Possible problem #3
=======================================================================================================================

Some older CUDA versions may refuse to build with newer compilers. For example, you my have ``gcc-9`` but it wants
``gcc-7``.

There are various ways to go about it.

If you can install the latest CUDA toolkit it typically should support the newer compiler.

Alternatively, you could install the lower version of the compiler in addition to the one you already have, or you may
already have it but it's not the default one, so the build system can't see it. If you have ``gcc-7`` installed but the
build system complains it can't find it, the following might do the trick:

.. code-block:: bash

    sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
    sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++


Here, we are making a symlink to ``gcc-7`` from ``/usr/local/cuda-10.2/bin/gcc`` and since
``/usr/local/cuda-10.2/bin/`` should be in the ``PATH`` environment variable (see the previous problem's solution), it
should find ``gcc-7`` (and ``g++7``) and then the build will succeed.

As always make sure to edit the paths in the example to match your situation.

FairScale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By integrating `FairScale <https://github.com/facebookresearch/fairscale/>`__ the :class:`~transformers.Trainer`
provides support for the following features from `the ZeRO paper <https://arxiv.org/abs/1910.02054>`__:

1. Optimizer State Sharding
2. Gradient Sharding
3. Model Parameters Sharding (new and very experimental)
4. CPU offload (new and very experimental)

You will need at least two GPUs to use this feature.


**Installation**:

Install the library via pypi:

.. code-block:: bash

    pip install fairscale

or via ``transformers``' ``extras``:

.. code-block:: bash

    pip install transformers[fairscale]

(will become available starting from ``transformers==4.6.0``)

or find more details on `the FairScale's GitHub page <https://github.com/facebookresearch/fairscale/#installation>`__.

If you're still struggling with the build, first make sure to read :ref:`zero-install-notes`.

If it's still not resolved the build issue, here are a few more ideas.

``fairscale`` seems to have an issue with the recently introduced by pip build isolation feature. If you have a problem
with it, you may want to try one of:

.. code-block:: bash

    pip install fairscale --no-build-isolation .

or:

.. code-block:: bash

    git clone https://github.com/facebookresearch/fairscale/
    cd fairscale
    rm -r dist build
    python setup.py bdist_wheel
    pip uninstall -y fairscale
    pip install dist/fairscale-*.whl

``fairscale`` also has issues with building against pytorch-nightly, so if you use it you may have to try one of:

.. code-block:: bash

    pip uninstall -y fairscale; pip install fairscale --pre \
    -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html \
    --no-cache --no-build-isolation

or:

.. code-block:: bash

    pip install -v --disable-pip-version-check . \
    -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html --pre

Of course, adjust the urls to match the cuda version you use.

If after trying everything suggested you still encounter build issues, please, proceed with the GitHub Issue of
`FairScale <https://github.com/facebookresearch/fairscale/issues>`__.



**Usage**:

To use the first version of Sharded data-parallelism, add ``--sharded_ddp simple`` to the command line arguments, and
make sure you have added the distributed launcher ``-m torch.distributed.launch
--nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`` if you haven't been using it already.

For example here is how you could use it for ``run_translation.py`` with 2 GPUs:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro \
    --fp16 --sharded_ddp simple

Notes:

- This feature requires distributed training (so multiple GPUs).
- It is not implemented for TPUs.
- It works with ``--fp16`` too, to make things even faster.
- One of the main benefits of enabling ``--sharded_ddp simple`` is that it uses a lot less GPU memory, so you should be
  able to use significantly larger batch sizes using the same hardware (e.g. 3x and even bigger) which should lead to
  significantly shorter training time.

3. To use the second version of Sharded data-parallelism, add ``--sharded_ddp zero_dp_2`` or ``--sharded_ddp
   zero_dp_3`` to the command line arguments, and make sure you have added the distributed launcher ``-m
   torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`` if you haven't been using it already.

For example here is how you could use it for ``run_translation.py`` with 2 GPUs:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro \
    --fp16 --sharded_ddp zero_dp_2

:obj:`zero_dp_2` is an optimized version of the simple wrapper, while :obj:`zero_dp_3` fully shards model weights,
gradients and optimizer states.

Both are compatible with adding :obj:`cpu_offload` to enable ZeRO-offload (activate it like this: :obj:`--sharded_ddp
"zero_dp_2 cpu_offload"`).

Notes:

- This feature requires distributed training (so multiple GPUs).
- It is not implemented for TPUs.
- It works with ``--fp16`` too, to make things even faster.
- The ``cpu_offload`` additional option requires ``--fp16``.
- This is an area of active development, so make sure you have a source install of fairscale to use this feature as
  some bugs you encounter may have been fixed there already.

Known caveats:

- This feature is incompatible with :obj:`--predict_with_generate` in the `run_translation.py` script.
- Using :obj:`--sharded_ddp zero_dp_3` requires wrapping each layer of the model in the special container
  :obj:`FullyShardedDataParallelism` of fairscale. It should be used with the option :obj:`auto_wrap` if you are not
  doing this yourself: :obj:`--sharded_ddp "zero_dp_3 auto_wrap"`.


DeepSpeed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Moved to :ref:`deepspeed-trainer-integration`.


Installation
=======================================================================================================================

Moved to :ref:`deepspeed-installation`.


Deployment with multiple GPUs
=======================================================================================================================

Moved to :ref:`deepspeed-multi-gpu`.


Deployment with one GPU
=======================================================================================================================

Moved to :ref:`deepspeed-one-gpu`.


Deployment in Notebooks
=======================================================================================================================

Moved to :ref:`deepspeed-notebook`.


Configuration
=======================================================================================================================

Moved to :ref:`deepspeed-config`.


Passing Configuration
=======================================================================================================================

Moved to :ref:`deepspeed-config-passing`.


Shared Configuration
=======================================================================================================================

Moved to :ref:`deepspeed-config-shared`.

ZeRO
=======================================================================================================================

Moved to :ref:`deepspeed-zero`.

ZeRO-2 Config
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-zero2-config`.

ZeRO-3 Config
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-zero3-config`.


NVMe Support
=======================================================================================================================

Moved to :ref:`deepspeed-nvme`.

ZeRO-2 vs ZeRO-3 Performance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-zero2-zero3-performance`.

ZeRO-2 Example
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-zero2-example`.

ZeRO-3 Example
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-zero3-example`.

Optimizer and Scheduler
=======================================================================================================================



Optimizer
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-optimizer`.


Scheduler
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Moved to :ref:`deepspeed-scheduler`.

fp32 Precision
=======================================================================================================================

Moved to :ref:`deepspeed-fp32`.

Automatic Mixed Precision
=======================================================================================================================

Moved to :ref:`deepspeed-amp`.

Batch Size
=======================================================================================================================

Moved to :ref:`deepspeed-bs`.

Gradient Accumulation
=======================================================================================================================

Moved to :ref:`deepspeed-grad-acc`.


Gradient Clipping
=======================================================================================================================

Moved to :ref:`deepspeed-grad-clip`.


Getting The Model Weights Out
=======================================================================================================================

Moved to :ref:`deepspeed-weight-extraction`.
