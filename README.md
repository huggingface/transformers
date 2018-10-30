# pytorch-pretrained-BERT
A PyTorch version of Google's pretrained BERT model as described in

No bells and whitles, just:
- [one class](bert_model.py) with a clean commented version of Google's BERT model that can load the weights pre-trained by Google's authors,
- [another class](data_processor.py) with all you need to pre- and post-process text data for the model (tokenize and encode),
- and [a script](download_weigths.sh) to download Google's pre-trained weights.

Here is how to use these:

```python
from .bert_model import BERT
from .data_processor import DataProcessor

bert_model = BERT(bert_model_path='.')
data_processor = DataProcessor(bert_vocab_path='.')

input_sentence = "We are playing with the BERT model."

tensor_input = data_processor.encode(input_sentence)
tensor_output = bert_model(prepared_input)
output_sentence = data_processor.decode(tensor_output)
```
