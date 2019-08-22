examples.rst

Examples
================================================

.. list-table::
   :header-rows: 1

   * - Sub-section
     - Description
   * - `Training large models: introduction, tools and examples <#introduction>`_
     - How to use gradient-accumulation, multi-gpu training, distributed training, optimize on CPU and 16-bits training to train Bert models
   * - `Fine-tuning with BERT: running the examples <#fine-tuning-bert-examples>`_
     - Running the examples in `examples <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples>`_\ : ``extract_classif.py``\ , ``run_bert_classifier.py``\ , ``run_bert_squad.py`` and ``run_lm_finetuning.py``
   * - `Fine-tuning with OpenAI GPT, Transformer-XL and GPT-2 <#fine-tuning>`_
     - Running the examples in `examples <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples>`_\ : ``run_openai_gpt.py``\ , ``run_transfo_xl.py`` and ``run_gpt2.py``
   * - `Fine-tuning BERT-large on GPUs <#fine-tuning-bert-large>`_
     - How to fine tune ``BERT large``


.. _introduction:

Training large models: introduction, tools and examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BERT-base and BERT-large are respectively 110M and 340M parameters models and it can be difficult to fine-tune them on a single GPU with the recommended batch size for good performance (in most case a batch size of 32).

To help with fine-tuning these models, we have included several techniques that you can activate in the fine-tuning scripts `run_bert_classifier.py <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/run_bert_classifier.py>`_ and `run_bert_squad.py <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/run_bert_squad.py>`_\ : gradient-accumulation, multi-gpu training, distributed training and 16-bits training . For more details on how to use these techniques you can read `the tips on training large batches in PyTorch <https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255>`_ that I published earlier this year.

Here is how to use these techniques in our scripts:


* **Gradient Accumulation**\ : Gradient accumulation can be used by supplying a integer greater than 1 to the ``--gradient_accumulation_steps`` argument. The batch at each step will be divided by this integer and gradient will be accumulated over ``gradient_accumulation_steps`` steps.
* **Multi-GPU**\ : Multi-GPU is automatically activated when several GPUs are detected and the batches are splitted over the GPUs.
* **Distributed training**\ : Distributed training can be activated by supplying an integer greater or equal to 0 to the ``--local_rank`` argument (see below).
* **16-bits training**\ : 16-bits training, also called mixed-precision training, can reduce the memory requirement of your model on the GPU by using half-precision training, basically allowing to double the batch size. If you have a recent GPU (starting from NVIDIA Volta architecture) you should see no decrease in speed. A good introduction to Mixed precision training can be found `here <https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/>`__ and a full documentation is `here <https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html>`__. In our scripts, this option can be activated by setting the ``--fp16`` flag and you can play with loss scaling using the ``--loss_scale`` flag (see the previously linked documentation for details on loss scaling). The loss scale can be zero in which case the scale is dynamically adjusted or a positive power of two in which case the scaling is static.

To use 16-bits training and distributed training, you need to install NVIDIA's apex extension `as detailed here <https://github.com/nvidia/apex>`__. You will find more information regarding the internals of ``apex`` and how to use ``apex`` in `the doc and the associated repository <https://github.com/nvidia/apex>`_. The results of the tests performed on pytorch-BERT by the NVIDIA team (and my trials at reproducing them) can be consulted in `the relevant PR of the present repository <https://github.com/huggingface/pytorch-pretrained-BERT/pull/116>`_.

Note: To use *Distributed Training*\ , you will need to run one training script on each of your machines. This can be done for example by running the following command on each server (see `the above mentioned blog post <https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255>`_\ ) for more details):

.. code-block:: bash

    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=$THIS_MACHINE_INDEX \
        --master_addr="192.168.1.1" \
        --master_port=1234 run_bert_classifier.py \
        (--arg1 --arg2 --arg3 and all other arguments of the run_classifier script)

Where ``$THIS_MACHINE_INDEX`` is an sequential index assigned to each of your machine (0, 1, 2...) and the machine with rank 0 has an IP address ``192.168.1.1`` and an open port ``1234``.

.. _fine-tuning-bert-examples:

Fine-tuning with BERT: running the examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We showcase several fine-tuning examples based on (and extended from) `the original implementation <https://github.com/google-research/bert/>`_\ :


* a *sequence-level classifier* on nine different GLUE tasks,
* a *token-level classifier* on the question answering dataset SQuAD, and
* a *sequence-level multiple-choice classifier* on the SWAG classification corpus.
* a *BERT language model* on another target corpus

GLUE results on dev set
~~~~~~~~~~~~~~~~~~~~~~~

We get the following results on the dev set of GLUE benchmark with an uncased BERT base
model (`bert-base-uncased`). All experiments ran on 8 V100 GPUs with a total train batch size of 24. Some of 
these tasks have a small dataset and training can lead to high variance in the results between different runs.
We report the median on 5 runs (with different seeds) for each of the metrics.

.. list-table::
   :header-rows: 1

   * - Task
     - Metric
     - Result
   * - CoLA
     - Matthew's corr.
     - 55.75
   * - SST-2
     - accuracy
     - 92.09
   * - MRPC
     - F1/accuracy
     - 90.48/86.27
   * - STS-B
     - Pearson/Spearman corr.
     - 89.03/88.64
   * - QQP
     - accuracy/F1
     - 90.92/87.72
   * - MNLI
     - matched acc./mismatched acc.
     - 83.74/84.06
   * - QNLI
     - accuracy
     - 91.07
   * - RTE
     - accuracy
     - 68.59
   * - WNLI
     - accuracy
     - 43.66


Some of these results are significantly different from the ones reported on the test set
of GLUE benchmark on the website. For QQP and WNLI, please refer to `FAQ #12 <https://gluebenchmark.com/faq>`_ on the webite.

Before running anyone of these GLUE tasks you should download the
`GLUE data <https://gluebenchmark.com/tasks>`_ by running
`this script <https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e>`_
and unpack it to some directory ``$GLUE_DIR``.

.. code-block:: shell

   export GLUE_DIR=/path/to/glue
   export TASK_NAME=MRPC

   python run_bert_classifier.py \
     --task_name $TASK_NAME \
     --do_train \
     --do_eval \
     --do_lower_case \
     --data_dir $GLUE_DIR/$TASK_NAME \
     --bert_model bert-base-uncased \
     --max_seq_length 128 \
     --train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3.0 \
     --output_dir /tmp/$TASK_NAME/

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

The code has not been tested with half-precision training with apex on any GLUE task apart from MRPC, MNLI, CoLA, SST-2. The following section provides details on how to run half-precision training with MRPC. With that being said, there shouldn't be any issues in running half-precision training with the remaining GLUE tasks as well, since the data processor for each task inherits from the base class DataProcessor.

MRPC
~~~~

This example code fine-tunes BERT on the Microsoft Research Paraphrase
Corpus (MRPC) corpus and runs in less than 10 minutes on a single K-80 and in 27 seconds (!) on single tesla V100 16GB with apex installed.

Before running this example you should download the
`GLUE data <https://gluebenchmark.com/tasks>`_ by running
`this script <https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e>`_
and unpack it to some directory ``$GLUE_DIR``.

.. code-block:: shell

   export GLUE_DIR=/path/to/glue

   python run_bert_classifier.py \
     --task_name MRPC \
     --do_train \
     --do_eval \
     --do_lower_case \
     --data_dir $GLUE_DIR/MRPC/ \
     --bert_model bert-base-uncased \
     --max_seq_length 128 \
     --train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3.0 \
     --output_dir /tmp/mrpc_output/

Our test ran on a few seeds with `the original implementation hyper-parameters <https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks>`__ gave evaluation results between 84% and 88%.

**Fast run with apex and 16 bit precision: fine-tuning on MRPC in 27 seconds!**
First install apex as indicated `here <https://github.com/NVIDIA/apex>`__.
Then run

.. code-block:: shell

   export GLUE_DIR=/path/to/glue

   python run_bert_classifier.py \
     --task_name MRPC \
     --do_train \
     --do_eval \
     --do_lower_case \
     --data_dir $GLUE_DIR/MRPC/ \
     --bert_model bert-base-uncased \
     --max_seq_length 128 \
     --train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3.0 \
     --output_dir /tmp/mrpc_output/ \
     --fp16

**Distributed training**
Here is an example using distributed training on 8 V100 GPUs and Bert Whole Word Masking model to reach a F1 > 92 on MRPC:

.. code-block:: bash

    python -m torch.distributed.launch \
        --nproc_per_node 8 run_bert_classifier.py \
        --bert_model bert-large-uncased-whole-word-masking \
        --task_name MRPC \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/MRPC/ \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
         --output_dir /tmp/mrpc_output/

Training with these hyper-parameters gave us the following results:

.. code-block:: bash

     acc = 0.8823529411764706
     acc_and_f1 = 0.901702786377709
     eval_loss = 0.3418912578906332
     f1 = 0.9210526315789473
     global_step = 174
     loss = 0.07231863956341798

Here is an example on MNLI:

.. code-block:: bash

    python -m torch.distributed.launch \
        --nproc_per_node 8 run_bert_classifier.py \
        --bert_model bert-large-uncased-whole-word-masking \
        --task_name mnli \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir /datadrive/bert_data/glue_data//MNLI/ \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir ../models/wwm-uncased-finetuned-mnli/ \
        --overwrite_output_dir

.. code-block:: bash

   ***** Eval results *****
     acc = 0.8679706601466992
     eval_loss = 0.4911287787382479
     global_step = 18408
     loss = 0.04755385363816904

   ***** Eval results *****
     acc = 0.8747965825874695
     eval_loss = 0.45516540421714036
     global_step = 18408
     loss = 0.04755385363816904

This is the example of the ``bert-large-uncased-whole-word-masking-finetuned-mnli`` model

SQuAD
~~~~~

This example code fine-tunes BERT on the SQuAD dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large) on a single tesla V100 16GB.

The data for SQuAD can be downloaded with the following links and should be saved in a ``$SQUAD_DIR`` directory.


* `train-v1.1.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json>`_
* `dev-v1.1.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json>`_
* `evaluate-v1.1.py <https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py>`_

.. code-block:: shell

   export SQUAD_DIR=/path/to/SQUAD

   python run_bert_squad.py \
     --bert_model bert-base-uncased \
     --do_train \
     --do_predict \
     --do_lower_case \
     --train_file $SQUAD_DIR/train-v1.1.json \
     --predict_file $SQUAD_DIR/dev-v1.1.json \
     --train_batch_size 12 \
     --learning_rate 3e-5 \
     --num_train_epochs 2.0 \
     --max_seq_length 384 \
     --doc_stride 128 \
     --output_dir /tmp/debug_squad/

Training with the previous hyper-parameters gave us the following results:

.. code-block:: bash

   python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /tmp/debug_squad/predictions.json
   {"f1": 88.52381567990474, "exact_match": 81.22043519394512}

**distributed training**

Here is an example using distributed training on 8 V100 GPUs and Bert Whole Word Masking uncased model to reach a F1 > 93 on SQuAD:

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=8 \
    run_bert_squad.py \
    --bert_model bert-large-uncased-whole-word-masking  \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../models/wwm_uncased_finetuned_squad/ \
    --train_batch_size 24 \
    --gradient_accumulation_steps 12

Training with these hyper-parameters gave us the following results:

.. code-block:: bash

   python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ../models/wwm_uncased_finetuned_squad/predictions.json
   {"exact_match": 86.91579943235573, "f1": 93.1532499015869}

This is the model provided as ``bert-large-uncased-whole-word-masking-finetuned-squad``.

And here is the model provided as ``bert-large-cased-whole-word-masking-finetuned-squad``\ :

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=8  run_bert_squad.py \
        --bert_model bert-large-cased-whole-word-masking \
        --do_train \
        --do_predict \
        --do_lower_case \
        --train_file $SQUAD_DIR/train-v1.1.json \
        --predict_file $SQUAD_DIR/dev-v1.1.json \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ../models/wwm_cased_finetuned_squad/ \
        --train_batch_size 24 \
        --gradient_accumulation_steps 12

Training with these hyper-parameters gave us the following results:

.. code-block:: bash

   python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ../models/wwm_uncased_finetuned_squad/predictions.json
   {"exact_match": 84.18164616840113, "f1": 91.58645594850135}

SWAG
~~~~

The data for SWAG can be downloaded by cloning the following `repository <https://github.com/rowanz/swagaf>`_

.. code-block:: shell

   export SWAG_DIR=/path/to/SWAG

   python run_bert_swag.py \
     --bert_model bert-base-uncased \
     --do_train \
     --do_lower_case \
     --do_eval \
     --data_dir $SWAG_DIR/data \
     --train_batch_size 16 \
     --learning_rate 2e-5 \
     --num_train_epochs 3.0 \
     --max_seq_length 80 \
     --output_dir /tmp/swag_output/ \
     --gradient_accumulation_steps 4

Training with the previous hyper-parameters on a single GPU gave us the following results:

.. code-block::

   eval_accuracy = 0.8062081375587323
   eval_loss = 0.5966546792367169
   global_step = 13788
   loss = 0.06423990014260186

LM Fine-tuning
~~~~~~~~~~~~~~

The data should be a text file in the same format as `sample_text.txt <./samples/sample_text.txt>`_  (one sentence per line, docs separated by empty line).
You can download an `exemplary training corpus <https://ext-bert-sample.obs.eu-de.otc.t-systems.com/small_wiki_sentence_corpus.txt>`_ generated from wikipedia articles and split into ~500k sentences with spaCy.
Training one epoch on this corpus takes about 1:20h on 4 x NVIDIA Tesla P100 with ``train_batch_size=200`` and ``max_seq_length=128``\ :

Thank to the work of @Rocketknight1 and @tholor there are now **several scripts** that can be used to fine-tune BERT using the pretraining objective (combination of masked-language modeling and next sentence prediction loss). These scripts are detailed in the `README <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning/README.md>`_ of the `examples/lm_finetuning/ <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning/>`_ folder.

.. _fine-tuning:

OpenAI GPT, Transformer-XL and GPT-2: running the examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide three examples of scripts for OpenAI GPT, Transformer-XL and OpenAI GPT-2 based on (and extended from) the respective original implementations:


* fine-tuning OpenAI GPT on the ROCStories dataset
* evaluating Transformer-XL on Wikitext 103
* unconditional and conditional generation from a pre-trained OpenAI GPT-2 model

Fine-tuning OpenAI GPT on the RocStories dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example code fine-tunes OpenAI GPT on the RocStories dataset.

Before running this example you should download the
`RocStories dataset <https://github.com/snigdhac/StoryComprehension_EMNLP/tree/master/Dataset/RoCStories>`_ and unpack it to some directory ``$ROC_STORIES_DIR``.

.. code-block:: shell

   export ROC_STORIES_DIR=/path/to/RocStories

   python run_openai_gpt.py \
     --model_name openai-gpt \
     --do_train \
     --do_eval \
     --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
     --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
     --output_dir ../log \
     --train_batch_size 16 \

This command runs in about 10 min on a single K-80 an gives an evaluation accuracy of about 87.7% (the authors report a median accuracy with the TensorFlow code of 85.8% and the OpenAI GPT paper reports a best single run accuracy of 86.5%).

Evaluating the pre-trained Transformer-XL on the WikiText 103 dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example code evaluate the pre-trained Transformer-XL on the WikiText 103 dataset.
This command will download a pre-processed version of the WikiText 103 dataset in which the vocabulary has been computed.

.. code-block:: shell

   python run_transfo_xl.py --work_dir ../log

This command runs in about 1 min on a V100 and gives an evaluation perplexity of 18.22 on WikiText-103 (the authors report a perplexity of about 18.3 on this dataset with the TensorFlow code).

Unconditional and conditional generation from OpenAI's GPT-2 model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example code is identical to the original unconditional and conditional generation codes.

Conditional generation:

.. code-block:: shell

   python run_gpt2.py

Unconditional generation:

.. code-block:: shell

   python run_gpt2.py --unconditional

The same option as in the original scripts are provided, please refere to the code of the example and the original repository of OpenAI.

.. _fine-tuning-BERT-large:

Fine-tuning BERT-large on GPUs
------------------------------

The options we list above allow to fine-tune BERT-large rather easily on GPU(s) instead of the TPU used by the original implementation.

For example, fine-tuning BERT-large on SQuAD can be done on a server with 4 k-80 (these are pretty old now) in 18 hours. Our results are similar to the TensorFlow implementation results (actually slightly higher):

.. code-block:: bash

   {"exact_match": 84.56953642384106, "f1": 91.04028647786927}

To get these results we used a combination of:


* multi-GPU training (automatically activated on a multi-GPU server),
* 2 steps of gradient accumulation and
* perform the optimization step on CPU to store Adam's averages in RAM.

Here is the full list of hyper-parameters for this run:

.. code-block:: bash

   export SQUAD_DIR=/path/to/SQUAD

   python ./run_bert_squad.py \
     --bert_model bert-large-uncased \
     --do_train \
     --do_predict \
     --do_lower_case \
     --train_file $SQUAD_DIR/train-v1.1.json \
     --predict_file $SQUAD_DIR/dev-v1.1.json \
     --learning_rate 3e-5 \
     --num_train_epochs 2 \
     --max_seq_length 384 \
     --doc_stride 128 \
     --output_dir /tmp/debug_squad/ \
     --train_batch_size 24 \
     --gradient_accumulation_steps 2

If you have a recent GPU (starting from NVIDIA Volta series), you should try **16-bit fine-tuning** (FP16).

Here is an example of hyper-parameters for a FP16 run we tried:

.. code-block:: bash

   export SQUAD_DIR=/path/to/SQUAD

   python ./run_bert_squad.py \
     --bert_model bert-large-uncased \
     --do_train \
     --do_predict \
     --do_lower_case \
     --train_file $SQUAD_DIR/train-v1.1.json \
     --predict_file $SQUAD_DIR/dev-v1.1.json \
     --learning_rate 3e-5 \
     --num_train_epochs 2 \
     --max_seq_length 384 \
     --doc_stride 128 \
     --output_dir /tmp/debug_squad/ \
     --train_batch_size 24 \
     --fp16 \
     --loss_scale 128

The results were similar to the above FP32 results (actually slightly higher):

.. code-block:: bash

   {"exact_match": 84.65468306527909, "f1": 91.238669287002}

Here is an example with the recent ``bert-large-uncased-whole-word-masking``\ :

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=8 \
     run_bert_squad.py \
     --bert_model bert-large-uncased-whole-word-masking \
     --do_train \
     --do_predict \
     --do_lower_case \
     --train_file $SQUAD_DIR/train-v1.1.json \
     --predict_file $SQUAD_DIR/dev-v1.1.json \
     --learning_rate 3e-5 \
     --num_train_epochs 2 \
     --max_seq_length 384 \
     --doc_stride 128 \
     --output_dir /tmp/debug_squad/ \
     --train_batch_size 24 \
     --gradient_accumulation_steps 2

Fine-tuning XLNet
-----------------

STS-B
~~~~~

This example code fine-tunes XLNet on the STS-B corpus.

Before running this example you should download the
`GLUE data <https://gluebenchmark.com/tasks>`_ by running
`this script <https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e>`_
and unpack it to some directory ``$GLUE_DIR``.

.. code-block:: shell

   export GLUE_DIR=/path/to/glue

   python run_xlnet_classifier.py \
    --task_name STS-B \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/STS-B/ \
    --max_seq_length 128 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mrpc_output/

Our test ran on a few seeds with `the original implementation hyper-parameters <https://github.com/zihangdai/xlnet#1-sts-b-sentence-pair-relevance-regression-with-gpus>`__ gave evaluation results between 84% and 88%.

**Distributed training**
Here is an example using distributed training on 8 V100 GPUs to reach XXXX:

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node 8 \
    run_xlnet_classifier.py \
    --task_name STS-B \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/STS-B/ \
    --max_seq_length 128 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mrpc_output/

Training with these hyper-parameters gave us the following results:

.. code-block:: bash

     acc = 0.8823529411764706
     acc_and_f1 = 0.901702786377709
     eval_loss = 0.3418912578906332
     f1 = 0.9210526315789473
     global_step = 174
     loss = 0.07231863956341798

Here is an example on MNLI:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node 8 run_bert_classifier.py \
        --bert_model bert-large-uncased-whole-word-masking \
        --task_name mnli \
        --do_train \
        --do_eval \
        --data_dir /datadrive/bert_data/glue_data//MNLI/ \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir ../models/wwm-uncased-finetuned-mnli/ \
        --overwrite_output_dir

.. code-block:: bash

   ***** Eval results *****
     acc = 0.8679706601466992
     eval_loss = 0.4911287787382479
     global_step = 18408
     loss = 0.04755385363816904

   ***** Eval results *****
     acc = 0.8747965825874695
     eval_loss = 0.45516540421714036
     global_step = 18408
     loss = 0.04755385363816904

This is the example of the ``bert-large-uncased-whole-word-masking-finetuned-mnli`` model.
