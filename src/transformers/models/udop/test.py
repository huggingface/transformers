from transformers import UdopTokenizer, UdopTokenizerFast


tokenizer = UdopTokenizer.from_pretrained("/Users/nielsrogge/Documents/UDOP/test")
fast_tokenizer = UdopTokenizerFast.from_pretrained("/Users/nielsrogge/Documents/UDOP/test")

output_ids = [0, 2233, 32942, 32959, 32836, 32958, 1]

print(tokenizer.decode(output_ids))
print(fast_tokenizer.decode(output_ids))
