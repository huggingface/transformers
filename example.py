"""
Show how to use HuggingFace's PyTorch implementation of Google's BERT Model.
"""
from .bert_model import BERT
from .prepare_inputs import DataPreprocessor

bert_model = BERT()
bert_model.load_from('.')
data_processor = DataProcessor(encoder_file_path='.')

input_sentence = "We are playing with the BERT model."
print("BERT inputs: {}".format(input_sentence))

tensor_input = data_processor.encode(input_sentence)
tensor_output = bert_model(prepared_input)
output_sentence = data_processor.decode(tensor_output)

print("BERT predicted: {}".format(output_sentence))
