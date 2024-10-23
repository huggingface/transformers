import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForMaskedLM
import time

test_sentence = 'Do you [MASK] the muffin man?'

# for comparison
bert = pipeline('fill-mask', model = 'bert-base-uncased')
print('\n'.join([d['sequence'] for d in bert(test_sentence)]))


deberta = pipeline('fill-mask', model = 'microsoft/deberta-v3-base', model_kwargs={"legacy": False})
print('\n'.join([d['sequence'] for d in deberta(test_sentence)]))


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

tokenized_dict = tokenizer(
    ["Is this working",], ["Not yet",],
    return_tensors="pt"
)

deberta.model.forward = torch.compile(deberta.model.forward)
start=time.time()
deberta.model(**tokenized_dict)
end=time.time()
print(end-start)


start=time.time()
deberta.model(**tokenized_dict)
end=time.time()
print(end-start)


start=time.time()
deberta.model(**tokenized_dict)
end=time.time()
print(end-start)


model = AutoModel.from_pretrained('microsoft/deberta-base')
model.config.return_dict = False
model.config.output_hidden_states=False
input_tuple = (tokenized_dict['input_ids'], tokenized_dict['attention_mask'])


start=time.time()
traced_model = torch.jit.trace(model, input_tuple)
end=time.time()
print(end-start)


start=time.time()
traced_model(tokenized_dict['input_ids'], tokenized_dict['attention_mask'])
end=time.time()
print(end-start)


start=time.time()
traced_model(tokenized_dict['input_ids'], tokenized_dict['attention_mask'])
end=time.time()
print(end-start)


start=time.time()
traced_model(tokenized_dict['input_ids'], tokenized_dict['attention_mask'])
end=time.time()
print(end-start)


start=time.time()
traced_model(tokenized_dict['input_ids'], tokenized_dict['attention_mask'])
end=time.time()
print(end-start)


torch.jit.save(traced_model, "compiled_deberta.pt")



# my_script_module = torch.jit.script(model)
