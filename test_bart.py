# from src.transformers.models.bart import BartConfig, BartForSequenceClassification
# from src.transformers.models.bart import BartTokenizer, BartForSequenceClassification, BartModel, BartForTokenClassification
# import torch
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# model = BartModel.from_pretrained('facebook/bart-base')
# model = BartForTokenClassification.from_pretrained('facebook/bart-base')
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors='pt')
# # {'input_ids': DeviceArray([[    0, 31414,     6,   127,  2335,    16, 11962,     2]], dtype=int32), 'attention_mask': DeviceArray([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)}
# outputs = model(**inputs)
#
#
# labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
#
# print(outputs)

from tests.test_modeling_beit import BeitModelTest

b=BeitModelTest()
b.setUp()
b.test_training()