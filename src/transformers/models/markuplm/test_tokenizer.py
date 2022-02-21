from transformers import MarkupLMTokenizer, MarkupLMTokenizerFast
import torch

slow_tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")

slow_encoding = slow_tokenizer(
    ["hello", "world"],
    xpaths=["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
    padding="max_length",
    max_length=20,
    return_tensors="pt",
)

fast_tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")

fast_encoding = fast_tokenizer(
    ["hello", "world"],
    xpaths=["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
    padding="max_length",
    max_length=20,
    return_tensors="pt",
)

for k in slow_encoding.keys():
    assert torch.allclose(slow_encoding[k], fast_encoding[k])
