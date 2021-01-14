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
<https://github.com/NVIDIA/apex>`__ for PyTorch and :obj:`tf.keras.mixed_precision` for TensorFlow.

Both :class:`~transformers.Trainer` and :class:`~transformers.TFTrainer` contain the basic training loop supporting the
previous features. To inject custom behavior you can subclass them and override the following methods:

- **get_train_dataloader**/**get_train_tfdataset** -- Creates the training DataLoader (PyTorch) or TF Dataset.
- **get_eval_dataloader**/**get_eval_tfdataset** -- Creates the evaluation DataLoader (PyTorch) or TF Dataset.
- **get_test_dataloader**/**get_test_tfdataset** -- Creates the test DataLoader (PyTorch) or TF Dataset.
- **log** -- Logs information on the various objects watching training.
- **create_optimizer_and_scheduler** -- Setups the optimizer and learning rate scheduler if they were not passed at
  init.
- **compute_loss** - Computes the loss on a batch of training inputs.
- **training_step** -- Performs a training step.
- **prediction_step** -- Performs an evaluation/test step.
- **run_model** (TensorFlow only) -- Basic pass through the model.
- **evaluate** -- Runs an evaluation loop and returns metrics.
- **predict** -- Returns predictions (with metrics if labels are available) on a test set.

Here is an example of how to customize :class:`~transformers.Trainer` using a custom loss function:

.. code-block:: python

    from transformers import Trainer
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs[0]
            return my_custom_loss(logits, labels)

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


Trainer Integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The :class:`~transformers.Trainer` has been extended to support libraries that may dramatically improve your training
time and fit much bigger models.

Currently it supports third party solutions, `DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ and `FairScale
<https://github.com/facebookresearch/fairscale/>`__, which implement parts of the paper `ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models, by Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
<https://arxiv.org/abs/1910.02054>`__.

This provided support is new and experimental as of this writing.

You will need at least 2 GPUs to benefit from these features.

FairScale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By integrating `FairScale <https://github.com/facebookresearch/fairscale/>`__ the :class:`~transformers.Trainer`
provides support for the following features from `the ZeRO paper <https://arxiv.org/abs/1910.02054>`__:

1. Optimizer State Sharding
2. Gradient Sharding

To deploy this feature:

1. Install the library via pypi:

   .. code-block:: bash

       pip install fairscale

   or find more details on `the FairScale's github page
   <https://github.com/facebookresearch/fairscale/#installation>`__.

2. Add ``--sharded_ddp`` to the command line arguments, and make sure you have added the distributed launcher ``-m
   torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`` if you haven't been using it already.

For example here is how you could use it for ``finetune_trainer.py`` with 2 GPUs:

.. code-block:: bash

    cd examples/seq2seq
    python -m torch.distributed.launch --nproc_per_node=2 ./finetune_trainer.py \
    --model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --data_dir wmt_en_ro \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --n_train 500 --num_train_epochs 1 \
    --per_device_train_batch_size 1  --freeze_embeds \
    --src_lang en_XX --tgt_lang ro_RO --task translation \
    --fp16 --sharded_ddp

Notes:

- This feature requires distributed training (so multiple GPUs).
- It is not implemented for TPUs.
- It works with ``--fp16`` too, to make things even faster.
- One of the main benefits of enabling ``--sharded_ddp`` is that it uses a lot less GPU memory, so you should be able
  to use significantly larger batch sizes using the same hardware (e.g. 3x and even bigger) which should lead to
  significantly shorter training time.


DeepSpeed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ implements everything described in the `ZeRO paper
<https://arxiv.org/abs/1910.02054>`__, except ZeRO's stage 3. "Parameter Partitioning (Pos+g+p)". Currently it provides
full support for:

1. Optimizer State Partitioning (ZeRO stage 1)
2. Add Gradient Partitioning (ZeRO stage 2)

To deploy this feature:

1. Install the library via pypi:

   .. code-block:: bash

       pip install deepspeed

   or find more details on `the DeepSpeed's github page <https://github.com/microsoft/deepspeed#installation>`__.

2. Adjust the :class:`~transformers.Trainer` command line arguments as following:

   1. replace ``python -m torch.distributed.launch`` with ``deepspeed``.
   2. add a new argument ``--deepspeed ds_config.json``, where ``ds_config.json`` is the DeepSpeed configuration file
      as documented `here <https://www.deepspeed.ai/docs/config-json/>`__. The file naming is up to you.

   Therefore, if your original command line looked as following:

   .. code-block:: bash

       python -m torch.distributed.launch --nproc_per_node=2 your_program.py <normal cl args>

   Now it should be:

   .. code-block:: bash

       deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json

   Unlike, ``torch.distributed.launch`` where you have to specify how many GPUs to use with ``--nproc_per_node``, with
   the ``deepspeed`` launcher you don't have to use the corresponding ``--num_gpus`` if you want all of your GPUs used.
   The full details on how to configure various nodes and GPUs can be found `here
   <https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node>`__.

   Here is an example of running ``finetune_trainer.py`` under DeepSpeed deploying all available GPUs:

   .. code-block:: bash

       cd examples/seq2seq
       deepspeed ./finetune_trainer.py --deepspeed ds_config.json \
       --model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --data_dir wmt_en_ro \
       --output_dir output_dir --overwrite_output_dir \
       --do_train --n_train 500 --num_train_epochs 1 \
       --per_device_train_batch_size 1  --freeze_embeds \
       --src_lang en_XX --tgt_lang ro_RO --task translation

   Note that in the DeepSpeed documentation you are likely to see ``--deepspeed --deepspeed_config ds_config.json`` -
   i.e. two DeepSpeed-related arguments, but for the sake of simplicity, and since there are already so many arguments
   to deal with, we combined the two into a single argument.

Before you can deploy DeepSpeed, let's discuss its configuration.

**Configuration:**

For the complete guide to the DeepSpeed configuration options that can be used in its configuration file please refer
to the `following documentation <https://www.deepspeed.ai/docs/config-json/>`__.

While you always have to supply the DeepSpeed configuration file, you can configure the DeepSpeed integration in
several ways:

1. Supply most of the configuration inside the file, and just use a few required command line arguments. This is the
   recommended way as it puts most of the configuration params in one place.
2. Supply just the ZeRO configuration params inside the file, and configure the rest using the normal
   :class:`~transformers.Trainer` command line arguments.
3. Any variation of the first two ways.

To get an idea of what DeepSpeed configuration file looks like, here is one that activates ZeRO stage 2 features,
enables FP16, uses AdamW optimizer and WarmupLR scheduler:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

       "zero_optimization": {
           "stage": 2,
           "allgather_partitions": true,
           "allgather_bucket_size": 5e8,
           "overlap_comm": true,
           "reduce_scatter": true,
           "reduce_bucket_size": 5e8,
           "contiguous_gradients": true,
           "cpu_offload": true
       },

       "optimizer": {
         "type": "AdamW",
         "params": {
           "lr": 3e-5,
           "betas": [ 0.8, 0.999 ],
           "eps": 1e-8,
           "weight_decay": 3e-7
         }
       },
       "zero_allow_untested_optimizer": true,

       "scheduler": {
         "type": "WarmupLR",
         "params": {
           "warmup_min_lr": 0,
           "warmup_max_lr": 3e-5,
           "warmup_num_steps": 500
         }
       }
    }

If you already have a command line that you have been using with :class:`transformers.Trainer` args, you can continue
using those and the :class:`~transformers.Trainer` will automatically convert them into the corresponding DeepSpeed
configuration at run time. For example, you could use the following configuration file:

.. code-block:: json

    {
       "zero_optimization": {
           "stage": 2,
           "allgather_partitions": true,
           "allgather_bucket_size": 5e8,
           "overlap_comm": true,
           "reduce_scatter": true,
           "reduce_bucket_size": 5e8,
           "contiguous_gradients": true,
           "cpu_offload": true
       }
    }

and the following command line arguments:

.. code-block:: bash

    --learning_rate 3e-5 --warmup_steps 500 --adam_beta1 0.8 --adam_beta2 0.999 --adam_epsilon 1e-8 \
    --weight_decay 3e-7 --lr_scheduler_type constant_with_warmup --fp16 --fp16_backend amp

to achieve the same configuration as provided by the longer json file in the first example.

When you execute the program, DeepSpeed will log the configuration it received from the :class:`~transformers.Trainer`
to the console, so you can see exactly what the final configuration was passed to it.

**Shared Configuration:**

Some configuration information is required by both the :class:`~transformers.Trainer` and DeepSpeed to function
correctly, therefore, to prevent conflicting definitions, which could lead to hard to detect errors, we chose to
configure those via the :class:`~transformers.Trainer` command line arguments.

Therefore, the following DeepSpeed configuration params shouldn't be used with the :class:`~transformers.Trainer`:

* ``train_batch_size``
* ``train_micro_batch_size_per_gpu``
* ``gradient_accumulation_steps``

as these will be automatically derived from the run time environment and the following 2 command line arguments:

.. code-block:: bash

    --per_device_train_batch_size 8 --gradient_accumulation_steps 2

which are always required to be supplied.

Of course, you will need to adjust the values in this example to your situation.



**ZeRO:**

The ``zero_optimization`` section of the configuration file is the most important part (`docs
<https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`__), since that is where you define
which ZeRO stages you want to enable and how to configure them.

.. code-block:: json

    {
       "zero_optimization": {
           "stage": 2,
           "allgather_partitions": true,
           "allgather_bucket_size": 5e8,
           "overlap_comm": true,
           "reduce_scatter": true,
           "reduce_bucket_size": 5e8,
           "contiguous_gradients": true,
           "cpu_offload": true
       }
    }

Notes:

- enabling ``cpu_offload`` should reduce GPU RAM usage (it requires ``"stage": 2``)
- ``"overlap_comm": true`` trades off increased GPU RAM usage to lower all-reduce latency. ``overlap_comm`` uses 4.5x
  the ``allgather_bucket_size`` and ``reduce_bucket_size`` values. So if they are set to 5e8, this requires a 9GB
  footprint (``5e8 x 2Bytes x 2 x 4.5``). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting
  OOM-errors you will need to reduce those parameters to about ``2e8``, which would require 3.6GB.

This section has to be configured exclusively via DeepSpeed configuration - the :class:`~transformers.Trainer` provides
no equivalent command line arguments.



**Optimizer:**


DeepSpeed's main optimizers are Adam, OneBitAdam, and Lamb. These have been thoroughly tested with ZeRO and are thus
recommended to be used. It, however, can import other optimizers from ``torch``. The full documentation is `here
<https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`__.

If you don't configure the ``optimizer`` entry in the configuration file, the :class:`~transformers.Trainer` will
automatically set it to ``AdamW`` and will use the supplied values or the defaults for the following command line
arguments: ``--learning_rate``, ``--adam_beta1``, ``--adam_beta2``, ``--adam_epsilon`` and ``--weight_decay``.

Here is an example of the pre-configured ``optimizer`` entry for AdamW:

.. code-block:: json

    {
       "zero_allow_untested_optimizer": true,
       "optimizer": {
           "type": "AdamW",
           "params": {
             "lr": 0.001,
             "betas": [0.8, 0.999],
             "eps": 1e-8,
             "weight_decay": 3e-7
           }
         }
    }

Since AdamW isn't on the list of tested with DeepSpeed/ZeRO optimizers, we have to add
``zero_allow_untested_optimizer`` flag.

If you want to use one of the officially supported optimizers, configure them explicitly in the configuration file, and
make sure to adjust the values. e.g. if use Adam you will want ``weight_decay`` around ``0.01``.


**Scheduler:**

DeepSpeed supports LRRangeTest, OneCycle, WarmupLR and WarmupDecayLR LR schedulers. The full documentation is `here
<https://www.deepspeed.ai/docs/config-json/#scheduler-parameters>`__.

If you don't configure the ``scheduler`` entry in the configuration file, the :class:`~transformers.Trainer` will use
the value of ``--lr_scheduler_type`` to configure it. Currently the :class:`~transformers.Trainer` supports only 2 LR
schedulers that are also supported by DeepSpeed:

* ``WarmupLR`` via ``--lr_scheduler_type constant_with_warmup``
* ``WarmupDecayLR`` via ``--lr_scheduler_type linear``. This is also the default value for ``--lr_scheduler_type``,
  therefore, if you don't configure the scheduler this is scheduler that will get configured by default.

In either case, the values of ``--learning_rate`` and ``--warmup_steps`` will be used for the configuration.

In other words, if you don't use the configuration file to set the ``scheduler`` entry, provide either:

.. code-block:: bash

    --lr_scheduler_type constant_with_warmup --learning_rate 3e-5 --warmup_steps 500

or

.. code-block:: bash

    --lr_scheduler_type linear --learning_rate 3e-5 --warmup_steps 500

with the desired values. If you don't pass these arguments, reasonable default values will be used instead.

In the case of WarmupDecayLR ``total_num_steps`` gets set either via the ``--max_steps`` command line argument, or if
it is not provided, derived automatically at run time based on the environment and the size of the dataset and other
command line arguments.

Here is an example of the pre-configured ``scheduler`` entry for WarmupLR (``constant_with_warmup`` in the
:class:`~transformers.Trainer` API):

.. code-block:: json

    {
       "scheduler": {
             "type": "WarmupLR",
             "params": {
                 "warmup_min_lr": 0,
                 "warmup_max_lr": 0.001,
                 "warmup_num_steps": 1000
             }
         }
    }

**Automatic Mixed Precision:**

You can work with FP16 in one of the following ways:

1. Pytorch native amp, as documented `here <https://www.deepspeed.ai/docs/config-json/#fp16-training-options>`__.
2. NVIDIA's apex, as documented `here
   <https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options>`__.

If you want to use an equivalent of the pytorch native amp, you can either configure the ``fp16`` entry in the
configuration file, or use the following command line arguments: ``--fp16 --fp16_backend amp``.

Here is an example of the ``fp16`` configuration:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
    }

If you want to use NVIDIA's apex instead, you can can either configure the ``amp`` entry in the configuration file, or
use the following command line arguments: ``--fp16 --fp16_backend apex --fp16_opt_level 01``.

Here is an example of the ``amp`` configuration:

.. code-block:: json

    {
        "amp": {
            "enabled": true,
            "opt_level": "O1"
        }
    }



**Gradient Clipping:**

If you don't configure the ``gradient_clipping`` entry in the configuration file, the :class:`~transformers.Trainer`
will use the value of the ``--max_grad_norm`` command line argument to set it.

Here is an example of the ``gradient_clipping`` configuration:

.. code-block:: json

    {
        "gradient_clipping": 1.0,
    }



**Notes:**

* DeepSpeed works with the PyTorch :class:`~transformers.Trainer` but not TF :class:`~transformers.TFTrainer`.
* While DeepSpeed has a pip installable PyPI package, it is highly recommended that it gets installed from `source
  <https://github.com/microsoft/deepspeed#installation>`__ to best match your hardware and also if you need to enable
  certain features, like 1-bit Adam, which aren't available in the pypi distribution.
* You don't have to use the :class:`~transformers.Trainer` to use DeepSpeed with HuggingFace ``transformers`` - you can
  use any model with your own trainer, and you will have to adapt the latter according to `the DeepSpeed integration
  instructions <https://www.deepspeed.ai/getting-started/#writing-deepspeed-models>`__.

**Main DeepSpeed Resources:**

- `github <https://github.com/microsoft/deepspeed>`__
- `Usage docs <https://www.deepspeed.ai/getting-started/>`__
- `API docs <https://deepspeed.readthedocs.io/en/latest/index.html>`__

Finally, please, remember that, HuggingFace :class:`~transformers.Trainer` only integrates DeepSpeed, therefore if you
have any problems or questions with regards to DeepSpeed usage, please, file an issue with `DeepSpeed github
<https://github.com/microsoft/DeepSpeed/issues>`__.
