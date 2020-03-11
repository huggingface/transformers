## A State-of-the-Art Large-scale Pretrained Response generation model (DialoGPT)

DialoGPT is a SOTA large-scale pretrained dialogue response generation model for multiturn conversations. 
The [human evaluation results](https://github.com/dreasysnail/Dialogpt_dev#human-evaluation) indicate that the response generated from DialoGPT is comparable to human response quality under a single-turn conversation Turing test.
The model is trained on 147M multi-turn dialogue from Reddit discussion thread. 

* Multi-turn generation examples from an interactive environment:

|Role | Response |
|---------|--------|
|User | Does money buy happiness? |
| Bot |  Depends how much money you spend on it .|
|User | What is the best way to buy happiness ? |
| Bot | You just have to be a millionaire by your early 20s, then you can be happy . |
|User |This is so difficult ! |
| Bot | You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money |

Please find the information about preprocessing, training and full details of the DialoGPT in the [original DialoGPT repository](https://github.com/microsoft/DialoGPT)

ArXiv paper: [https://arxiv.org/abs/1911.00536](https://arxiv.org/abs/1911.00536)
