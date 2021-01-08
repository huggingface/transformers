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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The trainer is being extended to support experimental libraries that may dramatically improve your training time and
fit bigger models.

The main part that is being integrated at the moment is based on the paper `ZeRO: Memory Optimizations Toward Training
Trillion Parameter Models, by Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
<https://arxiv.org/abs/1910.02054>`__.

These parts are supported by FairScale and DeepSpeed that have been integrated into ``transformers``' Trainer.

FairScale
-----------------------------------------------------------------------------------------------------------------------

You can already deploy the following features from this paper:

* Optimizer State Sharding
* Gradient Sharding

using the `--sharded_ddp` trainer argument. This is implemented via `fairscale
<https://github.com/facebookresearch/fairscale/>`__, so you will have to install this library.

This feature requires distributed training (so multiple GPUs) and is not implemented for TPUs.

For example here is how you could use it for ``finetune_trainer.py``:

.. code-block:: bash

    cd examples/seq2seq
    python -m torch.distributed.launch --nproc_per_node=2 ./finetune_trainer.py \
    --model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --data_dir wmt_en_ro \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --n_train 500 --num_train_epochs 1 \
    --per_device_train_batch_size 1  --freeze_embeds \
    --src_lang en_XX --tgt_lang ro_RO --task translation \
    --fp16 --sharded_ddp

Note that it works with `--fp16` too, to make things even faster.

One of the main benefits of enabling `--sharded_ddp` is that it uses a lot less GPU memory, so you should be able to
use significantly larger batch sizes using the same hardware (e.g. 3x or bigger).

DeepSpeed
-----------------------------------------------------------------------------------------------------------------------

The other important third party component that this trainer supports is `DeepSpeed
<https://github.com/microsoft/DeepSpeed>`__.

It implements almost everything described in the `ZeRO paper <https://arxiv.org/abs/1910.02054>`__. As of this writing
it is still missing ZeRO's stage 3. "Parameter Partitioning (Pos+g+p )", but it fully supports:

1. Optimizer State Partitioning (stage 1)
2. Add Gradient Partitioning (stage 2)

To enable DeepSpeed you need to first install DeepSpeed following `the instructions
<https://github.com/microsoft/deepspeed#installation>`__.

And when the installation has been completed you need to adjust the command line arguments as following:

1. replace ``python -m torch.distributed.launch`` with ``deepspeed``
2. add a new argument ``--deepspeed ds_config.json`` where ``ds_config.json`` is the DeepSpeed configuration file as
   documented at https://www.deepspeed.ai/docs/config-json/

Therefore if your original program looked as following:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 your_program.py <your program\'s normal args>

Now it becomes:

.. code-block:: bash

    deepspeed --num_gpus=2 your_program.py <your program\'s normal args> --deepspeed ds_config.json

Unlike, `torch.distributed.launch` where you have to specify how many gpus to use with `--nproc_per_node`, with the
`deepspeed` launcher you don't have to use the corresponding `--num_gpus` if you want all of your GPUs used. The full
details on how to configure various nodes and GPUs can be found
`here <https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node>`__.

Here is an example of running ``finetune_trainer.py`` under DeepSpeed deploying all available GPUs:

.. code-block:: bash

    cd examples/seq2seq
    deepspeed ./finetune_trainer.py --deepspeed ds_config.json \
    --model_name_or_path sshleifer/distill-mbart-en-ro-12-4 --data_dir wmt_en_ro \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --n_train 500 --num_train_epochs 1 \
    --per_device_train_batch_size 1  --freeze_embeds \
    --src_lang en_XX --tgt_lang ro_RO --task translation

Of course, you can name the DeepSpeed configuration file in any way you want, just adjust its name when you specify it
on the command line.

Note that in the DeepSpeed documentation you are likely to see ``--deepspeed --deepspeed_config ds_config.json`` - i.e.
2 DeepSpeed-related arguments, but for simplicity-sake, and since there are already so many arguments to deal with, we
combined the two into a single argument.

You can configure DeepSpeed integration in 2 ways:

1. supply most of the configuration inside ``ds_config.json``
2. configure it using the normal trainer arguments

For example here is an example of a ``ds_config.json`` configuration file which activates ZeRO stage 2 features,
enables fp16, uses Adam optimizer and WarmupLR scheduler:

.. code-block:: json

    {
        "steps_per_print": 2000,

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
           "allgather_bucket_size": 200000000,
           "overlap_comm": true,
           "reduce_scatter": true,
           "reduce_bucket_size": 200000000,
           "contiguous_gradients": true,
           "cpu_offload": true
       },

       "optimizer": {
         "type": "Adam",
         "params": {
           "lr": 3e-5,
           "betas": [
             0.8,
             0.999
           ],
           "eps": 1e-8,
           "weight_decay": 3e-7
         }
       },
       "scheduler": {
         "type": "WarmupLR",
         "params": {
           "warmup_min_lr": 0,
           "warmup_max_lr": 3e-5,
           "warmup_num_steps": 500
         }
       }
    }

If you already have a command line that you have been using with HF Trainer args, you can continue using those and
Trainer will automatically convert them into the corresponding DeepSpeed configuration file. So for example you could
use the following ``ds_config.json`` configuration file:

.. code-block:: json

    {
       "steps_per_print": 2000,
       "zero_optimization": {
           "stage": 2,
           "allgather_partitions": true,
           "allgather_bucket_size": 200000000,
           "overlap_comm": true,
           "reduce_scatter": true,
           "reduce_bucket_size": 200000000,
           "contiguous_gradients": true,
           "cpu_offload": true
       }
    }

and the following command line arguments:

.. code-block:: bash

    --learning_rate 3e-5 --warmup_steps 500 --adam_beta1 0.8 --adam_beta2 0.999 --adam_epsilon 1e-8 \
    --weight_decay 3e-7 --lr_scheduler_type constant_with_warmup --fp16 --fp16_backend amp

to achieve the same as the much longer json file in the first example.

You always have to supply the following arguments specific to both DeepSpeed configuration and are also needed by the
Trainer:

* ``--per_device_train_batch_size``
* ``--gradient_accumulation_steps``

For all the DeepSpeed configuration options that can be used in its configuration file please refer to the `following
documentation <https://www.deepspeed.ai/docs/config-json/>`__

The ``zero_optimization`` section of the configuration file is the most important part (`docs
<https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`__). For example, here is where you
define which ZeRO stages you want to enable.

Note the buffer sizes `allgather_bucket_size`, which `reduce_bucket_size` in this example are set to a relatively small
size which most cards should handle - if you have a large GPU consider raising those to ``500000000`` to get better
performance.

``transformers`` trainer only integrates DeepSpeed, therefore if you have any questions with regards to its usage
please file an issue with `DeepSpeed github <https://github.com/microsoft/deepspeed>`__.

Miscellaneous notes:

* DeepSpeed works with the PyTorch Trainer but not TF Trainer.
* While DeepSpeed has a pip installable PyPI package, it is highly recommended that it be `installed from source
  <https://github.com/microsoft/deepspeed#installation>`__ to best match your hardware and also to enable features like
  1-bit Adam, which aren't available in the pypi distribution.

Main DeepSpeed resources:

- `github <https://github.com/microsoft/deepspeed>`__
- `Usage docs <https://www.deepspeed.ai/getting-started/>`__
- `API docs <https://deepspeed.readthedocs.io/en/latest/index.html>`__
