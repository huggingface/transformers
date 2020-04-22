BERTology
---------

There is a growing field of study concerned with investigating the inner working of large-scale transformers like BERT (that some call "BERTology"). Some good examples of this field are:


* BERT Rediscovers the Classical NLP Pipeline by Ian Tenney, Dipanjan Das, Ellie Pavlick: https://arxiv.org/abs/1905.05950
* Are Sixteen Heads Really Better than One? by Paul Michel, Omer Levy, Graham Neubig: https://arxiv.org/abs/1905.10650
* What Does BERT Look At? An Analysis of BERT's Attention by Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning: https://arxiv.org/abs/1906.04341

In order to help this new field develop, we have included a few additional features in the BERT/GPT/GPT-2 models to help people access the inner representations, mainly adapted from the great work of Paul Michel (https://arxiv.org/abs/1905.10650):


* accessing all the hidden-states of BERT/GPT/GPT-2,
* accessing all the attention weights for each head of BERT/GPT/GPT-2,
* retrieving heads output values and gradients to be able to compute head importance score and prune head as explained in https://arxiv.org/abs/1905.10650.

To help you understand and use these features, we have added a specific example script: `bertology.py <https://github.com/huggingface/transformers/blob/master/examples/run_bertology.py>`_ while extract information and prune a model pre-trained on GLUE.
