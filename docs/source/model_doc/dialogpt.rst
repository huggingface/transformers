DialoGPT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

DialoGPT was proposed in
`DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation <https://arxiv.org/abs/1911.00536>`_
by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
It's a GPT2 Model trained on 147M conversation-like exchanges extracted from Reddit.

The abstract from the paper is the following:

*We present a large, tunable neural conversational response generation model, DialoGPT (dialogue generative pre-trained transformer). 
Trained on 147M conversation-like exchanges extracted from Reddit comment chains over a period spanning from 2005 through 2017, DialoGPT extends the Hugging Face PyTorch transformer to attain a performance close to human both in terms of automatic and human evaluation in single-turn dialogue settings.
We show that conversational systems that leverage DialoGPT generate more relevant, contentful and context-consistent responses than strong baseline systems.
The pre-trained model and training pipeline are publicly released to facilitate research into neural response generation and the development of more intelligent open-domain dialogue systems.*

Tips:

- DialoGPT is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- DialoGPT was trained with a causal language modeling (CLM) objective on conversational data and is therefore powerful at response generation in open-domain dialogue systems.
- DialoGPT enables the user to create a chat bot in just 10 lines of code as shown on `DialoGPT's model card <https://huggingface.co/microsoft/DialoGPT-medium>`_.

Training:

In order to train or fine-tune DialoGPT, one can use causal language modeling training. 
To cite the official paper: 
*We follow the OpenAI GPT-2 to model a multiturn dialogue session 
as a long text and frame the generation task as language modeling. We first
concatenate all dialog turns within a dialogue session into a long text 
x_1,..., x_N (N is the sequence length), ended by the end-of-text token.* 
For more information please confer to the original paper.
    

DialoGPT's architecture is based on the GPT2 model, so one can refer to GPT2's `docstring <https://huggingface.co/transformers/model_doc/gpt2.html>`_.

The original code can be found `here <https://github.com/microsoft/DialoGPT>`_.
