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
- DialoGPT enables the user to create a chat bot in just 10 lines of code as shown on the `model card of DialoGPT <https://huggingface.co/microsoft/DialoGPT-medium>`_.
 

Since DialoGPT is a GPT2 model it makes use of `modeling_gpt2.py` and one can refer to GPT2's `docstring <https://huggingface.co/transformers/model_doc/gpt2.html>`_.

The original code can be found `here <https://github.com/microsoft/DialoGPT>`_.
