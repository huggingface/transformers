.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Pipelines
-----------------------------------------------------------------------------------------------------------------------

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of
the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the
:doc:`task summary <../task_summary>` for examples of use.

There are two categories of pipeline abstractions to be aware about:

- The :func:`~transformers.pipeline` which is the most powerful object encapsulating all other pipelines.
- The other task-specific pipelines:

    - :class:`~transformers.AudioClassificationPipeline`
    - :class:`~transformers.AutomaticSpeechRecognitionPipeline`
    - :class:`~transformers.ConversationalPipeline`
    - :class:`~transformers.FeatureExtractionPipeline`
    - :class:`~transformers.FillMaskPipeline`
    - :class:`~transformers.ImageClassificationPipeline`
    - :class:`~transformers.ImageSegmentationPipeline`
    - :class:`~transformers.ObjectDetectionPipeline`
    - :class:`~transformers.QuestionAnsweringPipeline`
    - :class:`~transformers.SummarizationPipeline`
    - :class:`~transformers.TableQuestionAnsweringPipeline`
    - :class:`~transformers.TextClassificationPipeline`
    - :class:`~transformers.TextGenerationPipeline`
    - :class:`~transformers.Text2TextGenerationPipeline`
    - :class:`~transformers.TokenClassificationPipeline`
    - :class:`~transformers.TranslationPipeline`
    - :class:`~transformers.ZeroShotClassificationPipeline`

The pipeline abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `pipeline` abstraction is a wrapper around all the other available pipelines. It is instantiated as any other
pipeline but can provide additional quality of life.

Simple call on one item:

.. code-block::

    >>> pipe = pipeline("text-classification")
    >>> pipe("This restaurant is awesome")
    [{'label': 'POSITIVE', 'score': 0.9998743534088135}]

If you want to use a specific model from the `hub <https://huggingface.co>`__ you can ignore the task if the model on
the hub already defines it:

.. code-block::

    >>> pipe = pipeline(model="roberta-large-mnli")
    >>> pipe("This restaurant is awesome")
    [{'label': 'POSITIVE', 'score': 0.9998743534088135}]

To call a pipeline on many items, you can either call with a `list`.

.. code-block::

    >>> pipe = pipeline("text-classification")
    >>> pipe(["This restaurant is awesome", "This restaurant is aweful"])
    [{'label': 'POSITIVE', 'score': 0.9998743534088135},
     {'label': 'NEGATIVE', 'score': 0.9996669292449951}]


To iterate of full datasets it is recommended to use a :obj:`dataset` directly. This means you don't need to allocate
the whole dataset at once, nor do you need to do batching yourself. This should work just as fast as custom loops on
GPU. If it doesn't don't hesitate to create an issue.

.. code-block::

    import datasets
    from transformers import pipeline
    from transformers.pipelines.base import KeyDataset
    import tqdm

    pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
    dataset = datasets.load_dataset("superb", name="asr", split="test")

    # KeyDataset (only `pt`) will simply return the item in the dict returned by the dataset item
    # as we're not interested in the `target` part of the dataset.
    for out in tqdm.tqdm(pipe(KeyDataset(dataset, "file"))):
        print(out)
        # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
        # {"text": ....}
        # ....


.. autofunction:: transformers.pipeline

Pipeline batching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All pipelines (except `zero-shot-classification` and `question-answering` currently) can use batching. This will work
whenever the pipeline uses its streaming ability (so when passing lists or :obj:`Dataset`).

.. code-block::

    from transformers import pipeline                                                   
    from transformers.pipelines.base import KeyDataset
    import datasets
    import tqdm                                                                         

    dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
    pipe = pipeline("text-classification", device=0)
    for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
        print(out)
        # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
        # Exactly the same output as before, but the content are passed
        # as batches to the model


.. warning::

    However, this is not automatically a win for performance. It can be either a 10x speedup or 5x slowdown depending
    on hardware, data and the actual model being used.

    Example where it's most a speedup:


.. code-block::

    from transformers import pipeline                                                   
    from torch.utils.data import Dataset                                                
    import tqdm                                                                         


    pipe = pipeline("text-classification", device=0)                                    


    class MyDataset(Dataset):                                                           
        def __len__(self):                                                              
            return 5000                                                                 

        def __getitem__(self, i):                                                       
            return "This is a test"                                                     


    dataset = MyDataset()   

    for batch_size in [1, 8, 64, 256]:
        print("-" * 30)                                                                     
        print(f"Streaming batch_size={batch_size}")    
        for out in tqdm.tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):              
            pass


.. code-block::

    # On GTX 970
    ------------------------------
    Streaming no batching
    100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
    ------------------------------
    Streaming batch_size=8
    100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
    ------------------------------
    Streaming batch_size=64
    100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
    ------------------------------
    Streaming batch_size=256
    100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
    (diminishing returns, saturated the GPU)


Example where it's most a slowdown:

.. code-block::

    class MyDataset(Dataset):                                                           
        def __len__(self):                                                              
            return 5000                                                                 

        def __getitem__(self, i):                                                       
            if i % 64 == 0:                                                          
                n = 100                                                              
            else:                                                                    
                n = 1                                                                
            return "This is a test" * n

This is a occasional very long sentence compared to the other. In that case, the **whole** batch will need to be 400
tokens long, so the whole batch will be [64, 400] instead of [64, 4], leading to the high slowdown. Even worse, on
bigger batches, the program simply crashes.


.. code-block::

    ------------------------------
    Streaming no batching
    100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
    ------------------------------
    Streaming batch_size=8
    100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
    ------------------------------
    Streaming batch_size=64
    100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
    ------------------------------
    Streaming batch_size=256
      0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
    Traceback (most recent call last):
      File "/home/nicolas/src/transformers/test.py", line 42, in <module>
        for out in tqdm.tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
    ....
        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)


There are no good (general) solutions for this problem, and your mileage may vary depending on your use cases. Rule of
thumb:

For users, a rule of thumb is:

- **Measure performance on your load, with your hardware. Measure, measure, and keep measuring. Real numbers are the
  only way to go.**
- If you are latency constrained (live product doing inference), don't batch
- If you are using CPU, don't batch.
- If you are using throughput (you want to run your model on a bunch of static data), on GPU, then:

      - If you have no clue about the size of the sequence_length ("natural" data), by default don't batch, measure and
        try tentatively to add it, add OOM checks to recover when it will fail (and it will at some point if you don't
        control the sequence_length.)
      - If your sequence_length is super regular, then batching is more likely to be VERY interesting, measure and push
        it until you get OOMs.
      - The larger the GPU the more likely batching is going to be more interesting
- As soon as you enable batching, make sure you can handle OOMs nicely.

Pipeline custom code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to override a specific pipeline.

Don't hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most
cases, so :obj:`transformers` could maybe support your use case.


If you want to try simply you can:

- Subclass your pipeline of choice

.. code-block::

    class MyPipeline(TextClassificationPipeline):
        def postprocess(...):
            ...
            scores = scores * 100
            ...

    my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
    # or if you use `pipeline` function, then:
    my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)

That should enable you to do all the custom code you want.


Implementing a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`Implementing a new pipeline <../add_new_pipeline>`

The task specific pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


AudioClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.AudioClassificationPipeline
    :special-members: __call__
    :members:

AutomaticSpeechRecognitionPipeline
=======================================================================================================================

.. autoclass:: transformers.AutomaticSpeechRecognitionPipeline
    :special-members: __call__
    :members:

ConversationalPipeline
=======================================================================================================================

.. autoclass:: transformers.Conversation

.. autoclass:: transformers.ConversationalPipeline
    :special-members: __call__
    :members:

FeatureExtractionPipeline
=======================================================================================================================

.. autoclass:: transformers.FeatureExtractionPipeline
    :special-members: __call__
    :members:

FillMaskPipeline
=======================================================================================================================

.. autoclass:: transformers.FillMaskPipeline
    :special-members: __call__
    :members:

ImageClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.ImageClassificationPipeline
    :special-members: __call__
    :members:

ImageSegmentationPipeline
=======================================================================================================================

.. autoclass:: transformers.ImageSegmentationPipeline
    :special-members: __call__
    :members:

NerPipeline
=======================================================================================================================

.. autoclass:: transformers.NerPipeline

See :class:`~transformers.TokenClassificationPipeline` for all details.

ObjectDetectionPipeline
=======================================================================================================================

.. autoclass:: transformers.ObjectDetectionPipeline
    :special-members: __call__
    :members:

QuestionAnsweringPipeline
=======================================================================================================================

.. autoclass:: transformers.QuestionAnsweringPipeline
    :special-members: __call__
    :members:

SummarizationPipeline
=======================================================================================================================

.. autoclass:: transformers.SummarizationPipeline
    :special-members: __call__
    :members:

TableQuestionAnsweringPipeline
=======================================================================================================================

.. autoclass:: transformers.TableQuestionAnsweringPipeline
    :special-members: __call__


TextClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.TextClassificationPipeline
    :special-members: __call__
    :members:

TextGenerationPipeline
=======================================================================================================================

.. autoclass:: transformers.TextGenerationPipeline
    :special-members: __call__
    :members:

Text2TextGenerationPipeline
=======================================================================================================================

.. autoclass:: transformers.Text2TextGenerationPipeline
    :special-members: __call__
    :members:

TokenClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.TokenClassificationPipeline
    :special-members: __call__
    :members:

TranslationPipeline
=======================================================================================================================

.. autoclass:: transformers.TranslationPipeline
    :special-members: __call__
    :members:

ZeroShotClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.ZeroShotClassificationPipeline
    :special-members: __call__
    :members:

Parent class: :obj:`Pipeline`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Pipeline
    :members:
