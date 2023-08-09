from transformers import AutoTokenizer, AddedToken
tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast = False)
tokenizer_fast = AutoTokenizer.from_pretrained("t5-base", use_fast = False)

token = "<extra_id_x>"
tokenizer.add_tokens(AddedToken(token, single_word = True, lstrip = False, rstrip = False))

print(tokenizer.tokenize("Hey"+token+" hey"))
print(tokenizer.tokenize("Hey "+token+" hey"))
print(tokenizer.tokenize("Hey "+token+"hey"))
print(tokenizer.tokenize("Hey"+token+"hey"))
print(token)
token = "<extra_id_xx>"
tokenizer.add_tokens(AddedToken(token, single_word = True, lstrip = True, rstrip = False))

print(tokenizer.tokenize("Hey"+token+" hey"))
print(tokenizer.tokenize("Hey "+token+" hey"))
print(tokenizer.tokenize("Hey "+token+"hey"))
print(tokenizer.tokenize("Hey"+token+"hey"))
print(token)

token = "<extra_id_xxx>"
tokenizer.add_tokens(AddedToken(token, single_word = True, lstrip = True, rstrip = True))

print(tokenizer.tokenize("Hey"+token+" hey"))
print(tokenizer.tokenize("Hey "+token+" hey"))
print(tokenizer.tokenize("Hey "+token+"hey"))
print(tokenizer.tokenize("Hey"+token+"hey"))
print(token)

token = "<extra_id_xxx>"
tokenizer.add_tokens(AddedToken(token, single_word = True, lstrip = False, rstrip = True))

print(tokenizer.tokenize("Hey"+token+" hey"))
print(tokenizer.tokenize("Hey "+token+" hey"))
print(tokenizer.tokenize("Hey "+token+"hey"))
print(tokenizer.tokenize("Hey"+token+"hey"))
print(token)
