\# BERT Base Uncased



This is the BERT base model (uncased) as described in the paper:

\*\*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\*\*.



\- \*\*Layers\*\*: 12  

\- \*\*Hidden Size\*\*: 768  

\- \*\*Heads\*\*: 12  

\- \*\*Parameters\*\*: 110M  

\- \*\*Case\*\*: Uncased (lowercased input text)  



\## Usage

Example:

```python

from transformers import BertTokenizer, BertModel



tokenizer = BertTokenizer.from\_pretrained("bert-base-uncased")

model = BertModel.from\_pretrained("bert-base-uncased")



inputs = tokenizer("Hello, how are you?", return\_tensors="pt")

outputs = model(\*\*inputs)



