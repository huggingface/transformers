from transformers import MarkupLMTokenizer

tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")

encoding = tokenizer(["hello", "world"], xpaths=["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"], 
                      padding="max_length", max_length=20, return_tensors="pt")

for k,v in encoding.items():
    print(k, v.shape)