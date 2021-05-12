# Intro
Authors: @patrickvonplaten and @lhoestq


The original RAG implementation is able to train the question encoder and the generator in an end-to-end fashion. 
This extension enables complete end-to-end training of RAG including the context encoder in the retriever component. 
Please read the [accompanying blog post](https://shamanesiri.medium.com/how-to-finetune-the-entire-rag-architecture-including-dpr-retriever-4b4385322552) for details on this implementation.

The original RAG code has also been modified to work with the latest versions of pytorch lightning (version 1.2.10) and RAY (version 1.3.0). All other implementation details remain the same as the [original RAG code](https://github.com/huggingface/transformers/tree/master/examples/research_projects/rag).
Read more about RAG  at https://arxiv.org/abs/2005.11401.

This code can be modified to experiment with other research on retrival augmented models that include training of the retriever such as [REALM](https://arxiv.org/abs/2002.08909) and [MARGE](https://arxiv.org/abs/2006.15020). 

To start training, use the bash script (finetune_rag_ray_end2end.sh) in this folder. 


Contributors: @shamanez and @rivinduw
