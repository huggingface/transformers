from transformers import AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer, PreTrainedTokenizerFast
from huggingface_hub import login


access_token = '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'

login(access_token)

#hf-internal-testing/tiny-random-BertModel

from transformers import AutoTokenizer
#model = 'meta-llama/Llama-2-7b-chat-hf' #
#model = 'mistralai/Mistral-7B-Instruct-v0.2'
phrase = 'this is html <unk>'

# model = '01-ai/Yi-9B'
# slowtok_from_slow_false_prefix_true = AutoTokenizer.from_pretrained(model,use_fast=True,from_slow=False, legacy=True)
# yi = slowtok_from_slow_false_prefix_true.tokenize(phrase) #['this', 'Ġis', 'Ġhtml', 'Ġ<', 'unk', '>']
# print(yi)
# slowtok_from_slow_false_prefix_true = AutoTokenizer.from_pretrained(model,use_fast=True,from_slow=False, legacy=True, add_prefix_space=True)
# yi = slowtok_from_slow_false_prefix_true.tokenize(phrase) #['this', 'Ġis', 'Ġhtml', 'Ġ<', 'unk', '>']
# print(yi)

model = 'meta-llama/Llama-2-7b-chat-hf'
slowtok_from_slow_false = AutoTokenizer.from_pretrained(model,use_fast=True,from_slow=False, legacy=False)
llama = slowtok_from_slow_false.tokenize(phrase) #['this', 'Ġis', 'Ġhtml', 'Ġ<', 'unk', '>']
print(llama)
slowtok_from_slow_false = AutoTokenizer.from_pretrained(model,use_fast=True,from_slow=False, legacy=False, add_prefix_space=True)
llama = slowtok_from_slow_false.tokenize(phrase) #['this', 'Ġis', 'Ġhtml', 'Ġ<', 'unk', '>']
print(llama)

model = 'mistralai/Mistral-7B-Instruct-v0.2'
slowtok_from_slow_false_prefix_true = AutoTokenizer.from_pretrained(model,use_fast=True,from_slow=False, legacy=False)
mistral = slowtok_from_slow_false_prefix_true.tokenize(phrase) #['this', 'Ġis', 'Ġhtml', 'Ġ<', 'unk', '>']
print(mistral)
slowtok_from_slow_false_prefix_true = AutoTokenizer.from_pretrained(model,use_fast=True,from_slow=False, legacy=False, add_prefix_space=False)
mistral = slowtok_from_slow_false_prefix_true.tokenize(phrase) #['this', 'Ġis', 'Ġhtml', 'Ġ<', 'unk', '>']
print(mistral)

