from transformers import pipeline
test_sentence = 'Do you [MASK] the muffin man?'

# for comparison
bert = pipeline('fill-mask', model = 'bert-base-uncased')
print('\n'.join([d['sequence'] for d in bert(test_sentence)]))


deberta = pipeline('fill-mask', model = 'microsoft/deberta-v3-large')
print('\n'.join([d['sequence'] for d in deberta(test_sentence)]))
